"""SubLingo：日文影片 → 日文辨識 + 繁中翻譯 → 與影片同名的雙語 SRT。

  uv run sublingo.py 直播.mp4                 # large-v3 辨識（品質優先）
  uv run sublingo.py 直播.mp4 --fast          # turbo 辨識（快速模式）
  uv run sublingo.py 直播.mp4 --glossary 名詞表.txt

中間檔存在「影片名.sublingo/」，中斷後重跑會從上次進度接續。
"""
import argparse
import json
import sys
import time
import urllib.error
from pathlib import Path

from bench import ASR_MODELS, SYSTEM, chat, parse_batch, ts, write_srt

MAX_CHARS, MAX_SEC = 20, 6.0  # 每段字幕上限
GAP = 0.8                     # 字與字停頓超過這麼久就切段
MAX_WORD = 1.0                # 單一字最長秒數；超過代表後面的停頓被算進去
END_PUNCT = "。？！?!"
HALLUCINATIONS = ("ご視聴ありがとうございました", "ご清聴ありがとうございました", "チャンネル登録", "字幕")
FAILED = "【翻譯失敗】"


def load_json(path):
    return json.load(open(path, encoding="utf-8")) if path.exists() else None


def save_json(path, data):
    tmp = path.with_suffix(".tmp")  # 先寫暫存檔再改名，中斷時不會留下壞掉的 JSON
    json.dump(data, open(tmp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    tmp.replace(path)


def load_glossary(path):
    """回傳 (原始文字給翻譯用, 日文詞列表給語音辨識用)。"""
    if not path or not Path(path).exists():
        return "（無）", []
    text = Path(path).read_text(encoding="utf-8-sig").strip()
    ja = [line.split("=", 1)[0].strip() for line in text.splitlines() if "=" in line]
    return text, [w for w in ja if w]


# ---------------- ② 語音辨識 ----------------

def transcribe(video, model_name, hotwords, threads):
    from faster_whisper import WhisperModel
    print(f"載入語音辨識模型 {model_name}（第一次會下載）")
    model = WhisperModel(ASR_MODELS[model_name], device="cpu", compute_type="int8", cpu_threads=threads)
    segs, info = model.transcribe(
        str(video), language="ja", vad_filter=True, word_timestamps=True,
        condition_on_previous_text=False, beam_size=5,
        hotwords=" ".join(hotwords) or None)
    out, t0 = [], time.perf_counter()
    for s in segs:
        print(f"  [{ts(s.start)}] {s.text.strip()}   ({s.end / info.duration:.0%})")
        out.append({"start": s.start, "end": s.end, "text": s.text.strip(),
                    "no_speech": s.no_speech_prob, "logprob": s.avg_logprob,
                    "words": [[w.start, w.end, w.word] for w in (s.words or [])]})
    print(f"語音辨識完成，{time.perf_counter() - t0:.0f} 秒")
    return out


# ---------------- ③ 清理 + 重新斷句 ----------------

def is_hallucination(seg):
    # 只在 Whisper 自己也沒把握時才刪，避免刪掉直播主真的說的「ご視聴ありがとうございました」
    unsure = seg.get("no_speech", 0) > 0.5 or seg.get("logprob", 0) < -1.0
    return unsure and any(h in seg["text"] for h in HALLUCINATIONS)


def resegment(segments):
    out = []

    def flush(words):
        text = "".join(w[2] for w in words).strip(" 　、，,")
        if text:
            out.append({"start": words[0][0], "end": words[-1][1], "text": text})

    for seg in segments:
        if is_hallucination(seg):
            continue
        # [開始, 截短後結束, 字, 原始結束]；停頓判斷用原始結束，否則截短本身會製造假停頓
        words = [[s, min(e, s + MAX_WORD), w, e] for s, e, w in seg["words"] if w.strip()]
        if not words:
            if seg["text"]:
                out.append({"start": seg["start"], "end": seg["end"], "text": seg["text"]})
            continue
        cur = []
        for w in words:
            if cur and w[0] - cur[-1][3] > GAP:
                if len("".join(x[2] for x in cur)) < 3:
                    # 停頓前只剩一兩個字，多半是 Whisper 對位錯誤（例「期|4 秒|待」），併入下一段
                    cur = [[w[0], w[0], x[2], w[0]] for x in cur]
                else:
                    flush(cur)
                    cur = []
            elif cur and (len("".join(x[2] for x in cur) + w[2]) > MAX_CHARS or w[1] - cur[0][0] > MAX_SEC):
                # 太長：優先在逗號後切，其次在助詞/語尾後切，都沒有才硬切，避免把詞切成兩半
                cut = len(cur)
                for marks in ("、，,", "ねよさがはをにでてもと"):
                    # 不考慮最後一個字：它是否真的是助詞無法確定（例如「おも|ろい」），在那裡切等於硬切
                    ok = [i + 1 for i, x in enumerate(cur[:-1])
                          if x[2].rstrip()[-1:] in marks and len("".join(y[2] for y in cur[:i + 1])) >= 5]
                    if ok:
                        cut = ok[-1]
                        break
                flush(cur[:cut])
                cur = cur[cut:]
            cur.append(w)
            if w[2].rstrip()[-1:] in END_PUNCT:
                flush(cur)
                cur = []
        if cur:
            flush(cur)

    # 每段至少顯示 0.8 秒，但不蓋到下一段
    for i, s in enumerate(out):
        nxt = out[i + 1]["start"] if i + 1 < len(out) else float("inf")
        s["end"] = min(max(s["end"], s["start"] + 0.8), nxt)
    return out


# ---------------- ④ 翻譯 ----------------

_cc = None


def to_tw(text):
    """簡轉台灣繁體。LLM 輸出多半已是繁體，「只」會被 OpenCC 誤轉成「隻」（例：那是只有→那是隻有），所以保護起來。"""
    global _cc
    if _cc is None:
        from opencc import OpenCC
        _cc = OpenCC("s2twp")
    return _cc.convert(text.replace("只", "\0")).replace("\0", "只")


def translate(segs, path, model, glossary, threads, batch=40):
    """每批翻完就存檔；中斷後重跑會從 path 裡已完成的部分接續。"""
    zh_all = load_json(path) or []
    use_think, t0 = True, time.perf_counter()
    for b in range(len(zh_all), len(segs), batch):
        part = segs[b:b + batch]
        ids = list(range(b + 1, b + 1 + len(part)))
        ctx = [f"{segs[i]['text']} → {zh_all[i]}" for i in range(max(0, b - 5), b)]
        user = (f"名詞表：\n{glossary}\n\n前文（僅供參考，不要翻譯）：\n" + ("\n".join(ctx) or "（無）")
                + "\n\n請翻譯：\n" + json.dumps({"lines": [{"id": i, "ja": s["text"]} for i, s in zip(ids, part)]},
                                              ensure_ascii=False))
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
        zh = None
        for attempt in range(3):
            try:
                r = chat(model, msgs, use_think, threads)
            except urllib.error.HTTPError as e:
                if use_think and e.code == 400:  # 不支援 think 參數的模型
                    use_think = False
                    continue
                raise
            zh = parse_batch(r["message"]["content"], ids)
            if zh:
                break
            print(f"  第 {ids[0]}-{ids[-1]} 句格式不符，重試 {attempt + 1}")
        zh_all += [to_tw(z) for z in zh] if zh else [FAILED] * len(part)
        save_json(path, zh_all)
        print(f"  翻譯 {len(zh_all)}/{len(segs)} 句（{time.perf_counter() - t0:.0f} 秒）")
    return zh_all


# ---------------- 主流程 ----------------

def main():
    sys.stdout.reconfigure(encoding="utf-8")
    p = argparse.ArgumentParser(description="日文影片 → 雙語 SRT")
    p.add_argument("video")
    p.add_argument("--fast", action="store_true", help="用 large-v3-turbo 辨識，約快 2.7 倍但較不準")
    p.add_argument("--glossary", default="glossary.txt", help="名詞表（預設 glossary.txt，不存在就略過）")
    p.add_argument("--model", default="gemma4:26b", help="Ollama 翻譯模型")
    p.add_argument("--threads", type=int, default=14)
    p.add_argument("--redo", action="store_true", help="忽略中間檔，全部重做")
    args = p.parse_args()

    video = Path(args.video).resolve()
    if not video.exists():
        sys.exit(f"找不到影片：{video}")
    work = video.with_name(video.stem + ".sublingo")
    work.mkdir(exist_ok=True)
    asr_path, seg_path, zh_path = work / "asr.json", work / "segments.json", work / "translated.json"
    glossary, hotwords = load_glossary(args.glossary)
    asr_model = "turbo" if args.fast else "large"

    asr = None if args.redo else load_json(asr_path)
    if asr and asr["model"] == asr_model:
        print(f"沿用已完成的語音辨識（{asr_model}）")
    else:
        asr = {"model": asr_model, "segments": transcribe(video, asr_model, hotwords, args.threads)}
        save_json(asr_path, asr)

    segs = resegment(asr["segments"])
    if load_json(seg_path) != segs or args.redo:  # 斷句結果變了，舊翻譯就對不上
        save_json(seg_path, segs)
        zh_path.unlink(missing_ok=True)
    print(f"斷句完成：{len(asr['segments'])} 段 → {len(segs)} 段字幕")

    try:
        zh = translate(segs, zh_path, args.model, glossary, args.threads)
    except urllib.error.URLError as e:
        sys.exit(f"連不到 Ollama（{e.reason}）。請確認 Ollama 已啟動，重跑會從已翻譯的地方接續。")

    srt = video.with_suffix(".srt")
    if srt.exists():
        srt.replace(srt.with_suffix(".srt.bak"))  # 保留舊檔，以免蓋掉手動修改過的字幕
    write_srt(srt, [(s["start"], s["end"], f"{z}\n{s['text']}") for s, z in zip(segs, zh)])
    fails = zh.count(FAILED)
    print(f"\n完成：{srt}" + (f"（{fails} 句翻譯失敗，標記為 {FAILED}）" if fails else ""))


if __name__ == "__main__":
    main()
