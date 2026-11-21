# SubLingo 進度

本機日文影片 → 日文辨識 + 繁中翻譯 → 雙語軟字幕。內容：日本偶像直播聊天（mp4），只要軟字幕。

## 已定方向
- 核心 Python（uv 管理），每階段存中間檔：asr.json → 斷句 → 翻譯 json → 與影片同名的雙語 .srt（放影片旁，播放器自動載入）
- 不轉檔：不做 ASS、MKV 封裝、燒字幕，因此不需要 FFmpeg
- ASR：faster-whisper int8 CPU + VAD；候選 kotoba-whisper v2.0 / large-v3-turbo / large-v3
- 翻譯：Ollama + MoE 模型（候選 gemma4:26b、qwen3:30b-a3b），批次編號 JSON + 句數檢查 + OpenCC s2twp
- 介面：之後用 Gradio
- 暫不做：語者分離、預設人聲分離、WPF 介面、燒字幕

## 目前狀態（2026-11-20）
- [x] `bench.py`（asr / mt 實測）、`test_bench.py` 自檢通過，`uv sync` 完成
- [x] 安裝 Ollama；`gemma4:26b`、`qwen3:30b-a3b` 已下載（各 18 GB，名稱已用 `ollama list` 確認）
- [ ] 挑 10 分鐘直播片段跑 `bench.py asr`（含 `--threads 6` 對照）
- [ ] 準備 `glossary.txt`（成員名、團名、粉絲用語）跑 `bench.py mt`
- [x] 指令版 `sublingo.py`（斷句、幻聽過濾、名詞表、可接續、同名 SRT），`test_sublingo.py` 通過
- [x] 實跑 9 分鐘影片：191 段、0 失敗，翻譯 463 秒
- [ ] 使用者播放檢查字幕時間軸與翻譯
- [ ] 建立 `glossary.txt`
- [ ] Gradio 介面
- 決定：不認得的暱稱維持現狀（不改成 OOO）

## 實測結果（9 分鐘影片，2026-11-21）
- ASR：kotoba RTF 0.36 / turbo 0.25 / large-v3 0.68。large-v3 最準且有標點 → 暫定 large-v3，turbo 當快速模式；淘汰 kotoba
- 翻譯：gemma4:26b 105 分/小時、qwen3 83 分/小時（皆 6 執行緒，失敗 0 句）
- **使用者選定翻譯模型：gemma4:26b**（較忠實、懂直播情境）
- gemma4:26b 改 14 執行緒後：950 秒 → 544 秒（60 分/小時影片，生成 8.8 tok/s）
- 加速：Ollama num_thread=14（已加到 bench.py `--threads`）；記憶體 DDR4-3200 實跑 2400，建議 BIOS 開 XMP

## 待確認
- kotoba 的 words 逐字時間是否準確（斷句依賴它）
- 6 vs 14 threads 哪個快
- 實測尚未執行過 ASR 與 Ollama 路徑
