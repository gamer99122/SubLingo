"""SubLingo 實測腳本：比較語音辨識模型與翻譯模型的速度與品質。

  uv run bench.py asr 影片.mp4 --start 600 --duration 600
  uv run bench.py mt bench_out/asr_kotoba.json --models gemma4:26b qwen3:30b-a3b
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ASR_MODELS = {
    "kotoba": "kotoba-tech/kotoba-whisper-v2.0-faster",
    "turbo": "large-v3-turbo",
    "large": "large-v3",
}
OLLAMA = "http://localhost:11434/api/chat"
SR = 16000


def ts(t):
    ms = round(t * 1000)
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02}:{m:02}:{s:02},{ms:03}"


def write_srt(path, items):
    """items: [(start, end, text)]，text 可含換行（雙語）。"""
    with open(path, "w", encoding="utf-8") as f:
        for i, (s, e, text) in enumerate(items, 1):
            f.write(f"{i}\n{ts(s)} --> {ts(e)}\n{text}\n\n")


# ---------------- 語音辨識 ----------------

def kotoba_dir():
    """kotoba 的 config 沿用 large-v3 的 alignment_heads（第 7–25 層），但它只有 2 層解碼器，
    逐字時間戳會越界崩潰。複製一份，改用最後一層（第 1 層）全部 20 個 head。"""
    import shutil
    from huggingface_hub import snapshot_download
    dst = Path("models/kotoba-whisper-v2.0-fixed")
    if not (dst / "config.json").exists():
        shutil.copytree(snapshot_download(ASR_MODELS["kotoba"]), dst, dirs_exist_ok=True)
    heads = [[1, h] for h in range(20)]
    cfg = json.load(open(dst / "config.json", encoding="utf-8"))
    if cfg.get("alignment_heads") != heads:
        cfg["alignment_heads"] = heads
        json.dump(cfg, open(dst / "config.json", "w", encoding="utf-8"), indent=1)
    return str(dst)


def run_asr(args):
    from faster_whisper import WhisperModel, decode_audio

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    audio = decode_audio(args.video, sampling_rate=SR)
    end = (args.start + args.duration) * SR if args.duration else len(audio)
    audio = audio[int(args.start * SR):int(end)]
    clip_sec = len(audio) / SR
    if clip_sec < 1:
        sys.exit(f"取到的片段是空的：--start {args.start:.0f} 秒可能超過影片長度，或影片沒有音軌")
    print(f"片段長度 {clip_sec:.0f} 秒，threads={args.threads}")

    rows = []
    for name in args.models:
        print(f"\n== {name} ==（第一次執行會下載模型）")
        path = kotoba_dir() if name == "kotoba" else ASR_MODELS.get(name, name)
        model = WhisperModel(path, device="cpu",
                             compute_type="int8", cpu_threads=args.threads)
        t0 = time.perf_counter()
        segs, info = model.transcribe(
            audio, language="ja", vad_filter=True, word_timestamps=True,
            condition_on_previous_text=False,  # 避免幻聽在後文一直重複
            chunk_length=15 if name == "kotoba" else None,  # kotoba 官方建議 15 秒
            beam_size=5)
        result = []
        for s in segs:
            o = args.start  # 存成原影片的絕對時間，SRT 才能直接對著 mp4 看
            print(f"[{ts(s.start + o)}] {s.text.strip()}")
            result.append({"start": s.start + o, "end": s.end + o, "text": s.text.strip(),
                           "words": [[w.start + o, w.end + o, w.word] for w in (s.words or [])]})
        sec = time.perf_counter() - t0
        rows.append((name, sec, sec / clip_sec, len(result)))
        safe = name.replace("/", "_").replace(":", "_")
        json.dump({"model": name, "clip_sec": clip_sec, "asr_sec": sec, "segments": result},
                  open(out / f"asr_{safe}.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)
        write_srt(out / f"asr_{safe}.srt", [(r["start"], r["end"], r["text"]) for r in result])
        del model

    print("\n模型        耗時(秒)   RTF    1小時影片估計   段數")
    for name, sec, rtf, n in rows:
        print(f"{name:<10} {sec:8.0f}  {rtf:5.2f}  {rtf * 60:8.0f} 分鐘   {n}")


# ---------------- 翻譯 ----------------

SYSTEM = """你是專業的日文→繁體中文（台灣）字幕譯者，內容是日本偶像的直播聊天。
規則：
1. 逐句翻譯，id 一一對應，不可合併、拆分或省略任何一句。
2. 口語、自然、像台灣人說話；保留語氣（撒嬌、吐槽、驚訝、害羞），語尾的ね、よ、じゃん等轉成自然的中文語氣。
3. 人名、團名、專有名詞依照名詞表翻譯。
4. 語音辨識可能有錯字，請依上下文推測原意，不要加註解或括號說明。
5. 只輸出 JSON。"""

SCHEMA = {
    "type": "object",
    "properties": {"lines": {"type": "array", "items": {
        "type": "object",
        "properties": {"id": {"type": "integer"}, "zh": {"type": "string"}},
        "required": ["id", "zh"]}}},
    "required": ["lines"],
}


def parse_batch(content, ids):
    """模型回傳的 JSON 必須剛好對應 ids，否則回傳 None 讓呼叫端重試。"""
    try:
        lines = json.loads(content)["lines"]
        got = {int(x["id"]): str(x["zh"]).strip() for x in lines}
    except (ValueError, KeyError, TypeError):
        return None
    if len(lines) != len(ids) or sorted(got) != sorted(ids) or not all(got.values()):
        return None
    return [got[i] for i in ids]


def chat(model, messages, use_think, threads):
    # Ollama 在 13500 上預設只用約 6 執行緒，實測 14 時生成速度約為兩倍
    body = {"model": model, "messages": messages, "stream": False, "format": SCHEMA,
            "options": {"temperature": 0.3, "num_ctx": 8192, "num_thread": threads}, "keep_alive": "10m"}
    if use_think:
        body["think"] = False  # 關閉思考模式，否則 CPU 上很慢
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.load(r)


def run_mt(args):
    from opencc import OpenCC
    cc = OpenCC("s2twp")
    data = json.load(open(args.asr_json, encoding="utf-8"))
    segs = data["segments"]
    glossary = Path(args.glossary).read_text(encoding="utf-8").strip() if args.glossary else "（無）"
    stem = Path(args.asr_json).stem

    rows = []
    for model in args.models:
        print(f"\n== {model} ==")
        zh_all, stats, use_think = [], {"eval": 0, "eval_ns": 0, "prompt": 0, "prompt_ns": 0}, True
        t0 = time.perf_counter()
        for b in range(0, len(segs), args.batch):
            batch = segs[b:b + args.batch]
            ids = list(range(b + 1, b + 1 + len(batch)))
            ctx = [f"{segs[i]['text']} → {zh_all[i]}" for i in range(max(0, b - 5), b)]
            user = (f"名詞表：\n{glossary}\n\n前文（僅供參考，不要翻譯）：\n" + ("\n".join(ctx) or "（無）")
                    + "\n\n請翻譯：\n" + json.dumps({"lines": [{"id": i, "ja": s["text"]} for i, s in zip(ids, batch)]},
                                                  ensure_ascii=False))
            msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
            zh = None
            for attempt in range(3):
                try:
                    r = chat(model, msgs, use_think, args.threads)
                except urllib.error.HTTPError as e:
                    if use_think and e.code == 400:  # 不支援 think 參數的模型
                        use_think = False
                        continue
                    raise
                stats["eval"] += r.get("eval_count", 0)
                stats["eval_ns"] += r.get("eval_duration", 0)
                stats["prompt"] += r.get("prompt_eval_count", 0)
                stats["prompt_ns"] += r.get("prompt_eval_duration", 0)
                zh = parse_batch(r["message"]["content"], ids)
                if zh:
                    break
                print(f"  第 {ids[0]}-{ids[-1]} 句格式不符，重試 {attempt + 1}")
            zh = [cc.convert(z) for z in zh] if zh else ["【翻譯失敗】"] * len(batch)
            zh_all += zh
            for s, z in zip(batch, zh):
                print(f"  {s['text']}\n  → {z}")

        sec = time.perf_counter() - t0
        gen = stats["eval"] / (stats["eval_ns"] / 1e9 or 1)
        pp = stats["prompt"] / (stats["prompt_ns"] / 1e9 or 1)
        fails = zh_all.count("【翻譯失敗】")
        rows.append((model, sec, gen, pp, sec / data["clip_sec"] * 60, fails))
        safe = model.replace("/", "_").replace(":", "_")
        write_srt(Path(args.asr_json).parent / f"mt_{stem}_{safe}.srt",
                  [(s["start"], s["end"], f"{z}\n{s['text']}") for s, z in zip(segs, zh_all)])

    print("\n模型                    耗時(秒)  生成tok/s  讀入tok/s  1小時影片估計  失敗句")
    for model, sec, gen, pp, est, fails in rows:
        print(f"{model:<22} {sec:8.0f}  {gen:8.1f}  {pp:8.1f}  {est:8.0f} 分鐘   {fails}")


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # 重導到檔案時 Windows 預設 cp950，遇到日文會崩
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(required=True)
    a = sub.add_parser("asr", help="語音辨識實測")
    a.add_argument("video")
    a.add_argument("--start", type=float, default=0, help="從第幾秒開始")
    a.add_argument("--duration", type=float, default=600, help="測幾秒，0=全部")
    a.add_argument("--models", nargs="+", default=["kotoba", "turbo", "large"])
    a.add_argument("--threads", type=int, default=14)
    a.add_argument("--out", default="bench_out")
    a.set_defaults(fn=run_asr)
    m = sub.add_parser("mt", help="翻譯實測")
    m.add_argument("asr_json")
    m.add_argument("--models", nargs="+", required=True)
    m.add_argument("--batch", type=int, default=30)
    m.add_argument("--glossary", help="名詞表文字檔，每行如：かのん=花音")
    m.add_argument("--threads", type=int, default=14)
    m.set_defaults(fn=run_mt)
    args = p.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
