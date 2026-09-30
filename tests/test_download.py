"""download.py 离线单元测试：mock 网络层，验证重试/断点续跑/字节达标即停/双格式抽取/续传。

运行：python3 -m unittest discover -s tests -v
"""
import gzip
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "corpus"))
import download


def make_shard(dir_path: Path, name: str, docs: list[dict]) -> Path:
    """生成本地假分片，.gz 走 gzip 压缩。"""
    path = dir_path / name
    opener = gzip.open if name.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8") as f:
        for d in docs:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")
    return path


class ExtractTextTest(unittest.TestCase):
    def test_jsonl_and_gz(self):
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            s1 = make_shard(td, "a.jsonl", [{"text": "文本一"}, {"content": "WuDao格式"}, {"other": "无正文"}])
            s2 = make_shard(td, "b.jsonl.gz", [{"text": "压缩包文本"}])
            (td / "a.jsonl").write_text(  # 追加一行坏 JSON
                (td / "a.jsonl").read_text(encoding="utf-8") + "{bad json\n", encoding="utf-8")
            out = td / "out.jsonl"
            n = download.extract_text([s1, s2], out)
            texts = [json.loads(l)["text"] for l in out.read_text(encoding="utf-8").splitlines()]
            self.assertEqual(n, 3)
            self.assertEqual(texts, ["文本一", "WuDao格式", "压缩包文本"])


class DownloadShardTest(unittest.TestCase):
    def setUp(self):
        # 重定向下载目录到临时目录，避免污染 data/raw
        self._td = tempfile.TemporaryDirectory()
        self._patcher = mock.patch.object(download, "RAW_DIR", Path(self._td.name))
        self._patcher.start()

    def tearDown(self):
        self._patcher.stop()
        self._td.cleanup()

    def _run(self, shards, state, target, fetch):
        with mock.patch.object(download, "fetch_file", side_effect=fetch), \
             mock.patch("time.sleep"), \
             mock.patch.object(download, "save_state", lambda s: None):
            return list(download.download_shard(shards, state, target, "http://x", "r"))

    @staticmethod
    def _fake_fetch(url, dest):
        dest.write_bytes(b"x" * 100)  # 每个分片假定为 100 字节

    def test_skip_done_and_stop_at_target(self):
        # s1 已完成应跳过；目标 250 字节，下完 s2、s3 后达标即停，s4 不再下载
        state = {"done": ["s1.jsonl"], "bytes": 100}
        files = ["s1.jsonl", "s2.jsonl", "s3.jsonl", "s4.jsonl"]
        got = self._run(files, state, 250, self._fake_fetch)
        self.assertEqual(len(got), 2)
        self.assertEqual(state["done"], ["s1.jsonl", "s2.jsonl", "s3.jsonl"])
        self.assertEqual(state["bytes"], 300)

    def test_retry_then_success(self):
        state = {"done": [], "bytes": 0}
        calls = {"n": 0}

        def flaky_fetch(url, dest):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("x")  # 前两次失败
            self._fake_fetch(url, dest)     # 第三次成功

        got = self._run(["s.jsonl"], state, 100, flaky_fetch)
        self.assertEqual(len(got), 1)

    def test_give_up_after_max_retry(self):
        state = {"done": [], "bytes": 0}
        got = self._run(["s1.jsonl", "s2.jsonl"], state, 1000,
                        [ConnectionError("x")] * download.MAX_RETRY)
        self.assertEqual(got, [])       # 达到重试上限：放弃并中断，进度保留可续跑
        self.assertEqual(state["done"], [])


class FetchFileTest(unittest.TestCase):
    """fetch_file 的续传语义：用假 httpx.stream 验证 Range 头与写入模式。"""

    class _Resp:
        def __init__(self, status, data):
            self.status_code, self._data = status, data

        def raise_for_status(self):
            assert self.status_code in (200, 206)

        def iter_bytes(self, n):
            yield self._data

        def __enter__(self):
            return self

        def __exit__(self, *a):
            pass

    def _stream(self, status, data):
        captured = {}

        def fake_stream(method, url, headers=None, **kw):
            captured.update(headers or {})
            return self._Resp(status, data)

        return fake_stream, captured

    def test_resume_with_range(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "s.jsonl"
            dest.with_suffix(".jsonl.part").write_bytes(b"ab")  # 已有 2 字节进度
            fake, headers = self._stream(206, b"cdef")
            with mock.patch("httpx.stream", fake):
                download.fetch_file("http://x", dest)
            self.assertEqual(headers.get("Range"), "bytes=2-")
            self.assertEqual(dest.read_bytes(), b"abcdef")  # 追加写入

    def test_server_ignores_range(self):
        with tempfile.TemporaryDirectory() as td:
            dest = Path(td) / "s.jsonl"
            dest.with_suffix(".jsonl.part").write_bytes(b"ab")
            fake, _ = self._stream(200, b"full")  # 服务器回 200 全量
            with mock.patch("httpx.stream", fake):
                download.fetch_file("http://x", dest)
            self.assertEqual(dest.read_bytes(), b"full")    # 覆盖写入，未损坏


class StateTest(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(download, "STATE_FILE", Path(td) / "s.json"):
                self.assertEqual(download.load_state(), {"done": [], "bytes": 0})
                download.save_state({"done": ["a"], "bytes": 1})
                self.assertEqual(download.load_state(), {"done": ["a"], "bytes": 1})


if __name__ == "__main__":
    unittest.main()
