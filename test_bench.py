from bench import parse_batch, ts

assert ts(0) == "00:00:00,000"
assert ts(3723.4567) == "01:02:03,457"

ok = '{"lines":[{"id":2,"zh":"嗨"},{"id":1,"zh":"你好"}]}'
assert parse_batch(ok, [1, 2]) == ["你好", "嗨"]
assert parse_batch('{"lines":[{"id":1,"zh":"你好"}]}', [1, 2]) is None  # 漏句
assert parse_batch('{"lines":[{"id":1,"zh":"a"},{"id":1,"zh":"b"}]}', [1, 2]) is None  # 重複 id
assert parse_batch('{"lines":[{"id":1,"zh":""},{"id":2,"zh":"b"}]}', [1, 2]) is None  # 空譯文
assert parse_batch("not json", [1]) is None

from opencc import OpenCC
assert OpenCC("s2twp").convert("这个软件的视频") == "這個軟體的影片"
print("ok")
