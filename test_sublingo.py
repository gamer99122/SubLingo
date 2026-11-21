from sublingo import is_hallucination, resegment, to_tw

assert to_tw("但那是只有那兩個人才能辦的") == "但那是只有那兩個人才能辦的"
assert to_tw("这个软件的视频") == "這個軟體的影片"
assert to_tw("一隻貓") == "一隻貓"


def seg(words, text=None, **kw):
    return {"start": words[0][0], "end": words[-1][1], "text": text or "".join(w[2] for w in words),
            "words": words, **kw}


# 句號切段
out = resegment([seg([[0, 0.5, "はい"], [0.5, 0.6, "。"], [0.7, 1.2, "そう"], [1.2, 1.5, "だね"]])])
assert [s["text"] for s in out] == ["はい。", "そうだね"], out

# 停頓超過 0.8 秒切段
out = resegment([seg([[0, 0.5, "そうなん"], [0.5, 0.8, "だ"], [2.0, 2.5, "でもさ"]])])
assert [s["text"] for s in out] == ["そうなんだ", "でもさ"], out

# 被拉長的單字截短後，不能因此產生假停頓而切段
out = resegment([seg([[4.0, 4.3, "みて"], [4.3, 7.8, "る"], [7.8, 8.0, "人"]])])
assert [s["text"] for s in out] == ["みてる人"], out

# 停頓前只剩一兩個字（Whisper 對位錯誤）時併入下一段，時間從下一段開始
out = resegment([seg([[124.32, 124.6, "期"], [128.92, 129.08, "待"], [129.08, 129.72, "しちゃダメだよ"]])])
assert [s["text"] for s in out] == ["期待しちゃダメだよ"] and out[0]["start"] == 128.92, out

# 單字被拉長時截成 1 秒
out = resegment([seg([[0, 0.5, "えっと"], [0.5, 3.5, "ね"]])])
assert out[0]["end"] == 1.5, out

# 超過 20 字切段
w = [[i * 0.2, i * 0.2 + 0.2, "あいう"] for i in range(10)]  # 30 字、2 秒
out = resegment([seg(w)])
assert [len(s["text"]) for s in out] == [18, 12], out

# 太長時優先在逗號後切
w = [[0, 0.5, "どちらかさ、"], [0.5, 1, "みずき、"], [1, 1.5, "つしらみずきって"], [1.5, 2, "言ってたのが"]]
out = resegment([seg(w)])
assert [s["text"] for s in out] == ["どちらかさ、みずき", "つしらみずきって言ってたのが"], out

# 沒有逗號時在助詞後切
w = [[0, 0.5, "つしらみずきって"], [0.5, 1, "言ってたのが"], [1, 1.5, "めっちゃおも"], [1.5, 2, "ろくてさ"]]
out = resegment([seg(w)])
assert [s["text"] for s in out] == ["つしらみずきって言ってたのが", "めっちゃおもろくてさ"], out

# 至少顯示 0.8 秒，但不蓋到下一段
out = resegment([seg([[0, 0.2, "え"], [0.2, 0.3, "？"], [0.5, 1.0, "うん"]])])
assert out[0]["end"] == 0.5 and out[1]["end"] == 1.3, out

# 幻聽：只有 Whisper 沒把握時才刪
h = [[0, 1, "ご視聴ありがとうございました"]]
assert is_hallucination(seg(h, no_speech=0.8, logprob=-0.3))
assert not is_hallucination(seg(h, no_speech=0.1, logprob=-0.3))
assert resegment([seg(h, no_speech=0.8, logprob=-0.3)]) == []

# 沒有逐字時間時保留原段落
assert resegment([{"start": 1, "end": 2, "text": "テスト", "words": []}])[0]["text"] == "テスト"
print("ok")
