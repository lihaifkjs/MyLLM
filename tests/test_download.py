"""download.py 离线单元测试：mock 网络层，验证重试/断点续跑/字节达标即停/双格式抽取。

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
    def _run(self, repo_files, state, target, side_effect):
        with mock.patch("huggingface_hub.hf_hub_download") as m, \
             mock.patch("os.path.getsize", return_value=100), \
             mock.patch("time.sleep"), \
             mock.patch.object(download, "save_state", lambda s: None):
            m.side_effect = side_effect
            return list(download.download_shard(repo_files, state, target))

    def test_skip_done_and_stop_at_target(self):
        # s1 已完成应跳过；目标 250 字节，下完 s2、s3 后达标即停，s4 不再下载
        state = {"done": ["s1.jsonl"], "bytes": 100}
        files = ["s1.jsonl", "s2.jsonl", "s3.jsonl", "s4.jsonl"]
        got = self._run(files, state, 250, ["/fake/s2", "/fake/s3"])
        self.assertEqual(len(got), 2)
        self.assertEqual(state["done"], ["s1.jsonl", "s2.jsonl", "s3.jsonl"])
        self.assertEqual(state["bytes"], 300)

    def test_retry_then_success(self):
        state = {"done": [], "bytes": 0}
        got = self._run(["s.jsonl"], state, 100,
                        [ConnectionError("x"), ConnectionError("x"), "/fake/s"])
        self.assertEqual(len(got), 1)  # 前两次失败重试，第三次成功

    def test_give_up_after_max_retry(self):
        state = {"done": [], "bytes": 0}
        got = self._run(["s1.jsonl", "s2.jsonl"], state, 1000,
                        [ConnectionError("x")] * download.MAX_RETRY)
        self.assertEqual(got, [])       # 达到重试上限：放弃并中断，进度保留可续跑
        self.assertEqual(state["done"], [])


class StateTest(unittest.TestCase):
    def test_roundtrip(self):
        with tempfile.TemporaryDirectory() as td:
            with mock.patch.object(download, "STATE_FILE", Path(td) / "s.json"):
                self.assertEqual(download.load_state(), {"done": [], "bytes": 0})
                download.save_state({"done": ["a"], "bytes": 1})
                self.assertEqual(download.load_state(), {"done": ["a"], "bytes": 1})


if __name__ == "__main__":
    unittest.main()
