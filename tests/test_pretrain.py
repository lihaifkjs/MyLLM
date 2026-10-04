"""模块4 预训练冒烟测试：合成小数据，CPU 跑数步，校验 ckpt 与日志契约。"""
import json

import numpy as np
import pytest
import torch

from src.model import LlamaModel, ModelConfig
from src.pretrain.dataset import TokenData
from src.pretrain.encode_corpus import encode_corpus
from src.pretrain.train import main as train_main

SEQ_LEN = 32
NUM_TOKENS = 20000


@pytest.fixture()
def tokens_bin(tmp_path):
    """合成随机 token 流（uint16），模拟 encode_corpus 的产物。"""
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 32000, size=NUM_TOKENS, dtype=np.uint16)
    bin_path = tmp_path / "tokens.bin"
    arr.tofile(bin_path)
    bin_path.with_suffix(".meta.json").write_text(
        json.dumps({"num_tokens": NUM_TOKENS, "dtype": "uint16", "eos_id": 3}))
    return str(bin_path)


def test_encode_corpus(tmp_path):
    corpus = tmp_path / "corpus.txt"
    corpus.write_text("今天天气不错\n\n我们去看电影吧\n", encoding="utf-8")
    out = tmp_path / "tokens.bin"
    n = encode_corpus(str(corpus), str(out))
    meta = json.loads(out.with_suffix(".meta.json").read_text())
    assert meta["num_tokens"] == n == len(np.fromfile(out, dtype=np.uint16))
    assert n > 2  # 两行非空，各至少 1 个 token + eos


def test_dataset_window(tokens_bin):
    data = TokenData(tokens_bin, val_ratio=0.01)
    rng = np.random.default_rng(42)
    x, y = data.random_batch(4, SEQ_LEN, rng)
    assert x.shape == y.shape == (4, SEQ_LEN)
    assert torch.equal(x[:, 1:], y[:, :-1])  # y 是 x 右移一位（next-token 对齐）
    batches = list(data.val_batches(2, SEQ_LEN, 3))
    assert 1 <= len(batches) <= 3


def test_train_smoke(tokens_bin, tmp_path):
    ckpt_dir = tmp_path / "ckpt"
    log = tmp_path / "log.jsonl"
    train_main([
        "--data", tokens_bin, "--ckpt-dir", str(ckpt_dir), "--log", str(log),
        "--seq-len", str(SEQ_LEN), "--micro-batch", "2", "--grad-accum", "1",
        "--total-steps", "3", "--warmup-steps", "1",
        "--eval-interval", "2", "--eval-batches", "2",
        "--save-interval", "2", "--log-interval", "1", "--device", "cpu",
    ])

    # ckpt 契约：含 optimizer 状态，可加载回原模型且前向形状正确
    ckpt = torch.load(ckpt_dir / "pretrain.ckpt", map_location="cpu", weights_only=False)
    assert {"model", "optimizer", "step", "config"} <= ckpt.keys()
    assert ckpt["step"] == 3
    model = LlamaModel(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    assert model(torch.zeros(1, SEQ_LEN, dtype=torch.long)).shape == (1, SEQ_LEN, 32000)
    assert (ckpt_dir / "pretrain_last.ckpt").exists()

    # 日志契约：train 与 eval 条目字段齐全
    records = [json.loads(line) for line in open(log, encoding="utf-8")]
    train_rec = next(r for r in records if r["type"] == "train")
    eval_rec = next(r for r in records if r["type"] == "eval")
    assert {"step", "train_loss", "lr", "grad_norm", "tokens_seen", "elapsed_s"} <= train_rec.keys()
    assert {"step", "val_loss", "val_ppl"} <= eval_rec.keys()
    assert eval_rec["val_ppl"] > 1
