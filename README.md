# SubLingo

本機日文影片 AI 字幕工具：辨識日文語音，翻譯成繁體中文（台灣用語），輸出雙語 SRT 字幕。

- 全程在本機 CPU 執行，不需要獨立顯卡，也不用付費雲端 API
- 主要針對日本偶像直播的聊天內容調整
- 輸出和影片同名的 `.srt`，放在影片旁邊，播放器（PotPlayer、VLC、MPC-HC）會自動載入

> **目前狀態：實測階段。** 本 repo 現在只有用來比較模型的實測腳本 `bench.py`，正式的一鍵處理工具與圖形介面尚在開發中。進度見 [PROGRESS.md](PROGRESS.md)。

## 處理流程

```
影片.mp4
  → ① 讀取音訊（PyAV）
  → ② 語音辨識（faster-whisper + VAD，CPU int8）
  → ③ 清理幻聽句、重新斷句            ← 開發中
  → ④ 翻譯（Ollama 本機 LLM，分批 + 句數檢查 + OpenCC 轉台灣繁體）
  → ⑤ 輸出雙語 影片.srt
```

## 技術選型

| 用途 | 採用 | 說明 |
|---|---|---|
| 語音辨識 | [faster-whisper](https://github.com/SYSTRAN/faster-whisper) + Whisper large-v3 | 準確度最高、有標點；`large-v3-turbo` 作為快速模式 |
| 翻譯 | [Ollama](https://ollama.com) + `gemma4:26b` | MoE 模型，CPU 上速度與品質較平衡 |
| 簡轉繁 | [OpenCC](https://github.com/BYVoid/OpenCC) `s2twp` | 修正簡體字與大陸用語 |

## 實測結果

測試環境：Intel Core i5-13500、64GB DDR4（實跑 2400）、無獨立顯卡、Windows 11。測試影片為 9 分鐘偶像直播。

**語音辨識**（RTF = 處理時間 ÷ 影片長度，越小越快）

| 模型 | RTF | 1 小時影片約 | 評語 |
|---|---|---|---|
| kotoba-whisper v2.0 | 0.36 | 22 分 | 錯字多、漏字約 13%，淘汰 |
| large-v3-turbo | 0.25 | 15 分 | 最快，部分句子聽錯 |
| **large-v3** | 0.68 | 41 分 | 最準，有標點 |

**翻譯**（Ollama `num_thread=14`）

| 模型 | 生成速度 | 1 小時影片約 | 評語 |
|---|---|---|---|
| **gemma4:26b** | 8.8 tok/s | 60 分 | 忠實、懂直播情境 |
| qwen3:30b-a3b | 6.0 tok/s* | 83 分* | 口語自然但偶爾誤解情境 |

\* qwen3 為 6 執行緒時的數據，未用 14 執行緒重測。

## 安裝

需要 [uv](https://docs.astral.sh/uv/) 與 [Ollama](https://ollama.com)。

```
winget install Ollama.Ollama
ollama pull gemma4:26b
uv sync
```

Whisper 模型會在第一次執行時自動下載（每個約 1.5–3 GB）。

## 使用實測腳本

```
# 語音辨識：比較三個模型（--duration 0 = 到片尾）
uv run bench.py asr 影片.mp4 --start 0 --duration 0

# 翻譯：用辨識結果比較翻譯模型
uv run bench.py mt bench_out/asr_large.json --models gemma4:26b --glossary glossary.txt
```

結果輸出到 `bench_out/`：每個模型一份 `.json`（含逐字時間戳）與 `.srt`。

**名詞表** `glossary.txt`（選用，UTF-8）：每行一組「日文=中文」，填入成員名、團名、粉絲用語，可明顯改善譯名一致性。

```
つしらみずき=津白瑞希
推し=本命
```

## 已知問題與處理

- **PyAV 19 與 faster-whisper 1.2.1 不相容**：`pyproject.toml` 已鎖定 `av<19`
- **kotoba-whisper 逐字時間戳會讓程式崩潰**：模型設定檔的 `alignment_heads` 指到不存在的解碼器層；`bench.py` 會自動複製一份修正版到 `models/`
- **Ollama 在混合架構 CPU 上預設執行緒偏少**：明確設 `num_thread=14`，生成速度約為預設的 1.7 倍
- **記憶體沒跑在標稱速度時翻譯會變慢**：DDR4-3200 若實跑 2400，請到 BIOS 開啟 XMP

## 開發計畫

- [ ] 正式處理流程 `sublingo.py`：一個指令從影片產生雙語 SRT
- [ ] 幻聽句過濾與重新斷句（每段 ≤ 6 秒、≤ 20 字）
- [ ] 名詞表同時用於語音辨識（`hotwords`）與翻譯
- [ ] Gradio 本機網頁介面
