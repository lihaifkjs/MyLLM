"""模块5 SFT 冒烟测试：合成小 jsonl，CPU 跑数步，校验 mask / ckpt / 日志契约。"""
import json
from dataclasses import asdict

import pytest
import torch

from src.model import LlamaModel, ModelConfig
from src.sft.dataset import PROMPT_TEMPLATE, collate_batch, encode_sample
from src.sft.train import main as train_main
from src.tokenizer.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def tok():
    return Tokenizer("tokenizer/tokenizer.model")


def test_encode_sample_mask(tok):
    ids, labels = encode_sample(tok, "失眠了怎么办？", "可以试试睡前泡脚。", max_len=1024)
    prompt_ids = [tok.bos_id] + tok.encode(PROMPT_TEMPLATE.format(instruction="失眠了怎么办？"))
    assert ids[: len(prompt_ids)] == prompt_ids
    assert all(l == -100 for l in labels[: len(prompt_ids)])  # prompt 不监督
    assert labels[len(prompt_ids):] == ids[len(prompt_ids):]  # 回答部分监督
    assert ids[-1] == labels[-1] == tok.eos_id                # 以 <eos> 收尾且被监督


def test_encode_sample_truncation(tok):
    # output 超长：保留完整 prompt、只截 output，保证 <eos> 收尾
    ids, labels = encode_sample(tok, "讲个故事", "很长" * 500, max_len=64)
    prompt_len = 1 + len(tok.encode(PROMPT_TEMPLATE.format(instruction="讲个故事")))
    assert len(ids) == len(labels) == 64
    assert labels[:prompt_len] == [-100] * prompt_len
    assert ids[-1] == labels[-1] == tok.eos_id
    # prompt 本身就超长：整条丢弃
    assert encode_sample(tok, "很长" * 500, "好", max_len=64) is None


def test_collate_padding(tok):
    samples = [encode_sample(tok, "天为什么是蓝的？", "因为瑞利散射。", 128),
               encode_sample(tok, "1+1=?", "2", 128)]
    input_ids, labels = collate_batch(samples, tok.pad_id)
    # batch 内右 padding，向上对齐 64 的倍数
    assert input_ids.shape == labels.shape
    assert input_ids.shape[0] == 2
    assert input_ids.shape[1] % 64 == 0 and input_ids.shape[1] >= len(samples[0][0])
    short = len(samples[1][0])
    assert (input_ids[1, short:] == tok.pad_id).all()
    assert (labels[1, short:] == -100).all()


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_prepare_data_unicode_line_sep(tmp_path):
    """回归：output 含 U+2028 时，convert/sample_split 不得把记录切成两段。"""
    from src.sft.prepare_data import convert, sample_split

    sep = chr(0x2028)  # Unicode 行分隔符，Firefly 原始数据中真实出现
    answer = "答" + sep + "答"
    raw = tmp_path / "raw.jsonl"
    with open(raw, "w", encoding="utf-8") as f:
        # json.dumps(ensure_ascii=False) 会把 sep 原样写出，模拟真实 raw 文件
        f.write(json.dumps({"kind": "NLP", "input": "问", "target": answer}, ensure_ascii=False) + "\n")
        f.write(json.dumps({"kind": "NLP", "input": "", "target": "空指令丢弃"}) + "\n")
    full = tmp_path / "sft.jsonl"
    assert convert(str(raw), str(full)) == 1  # 空指令被丢弃

    # sft.jsonl 严格一行一条（对任何按行工具成立）
    assert len(full.read_text(encoding="utf-8").splitlines()) == 1

    train, val = tmp_path / "train.jsonl", tmp_path / "val.jsonl"
    n_train, n_val = sample_split(str(full), str(train), str(val),
                                  num_samples=0, val_ratio=0.5, seed=42)
    assert (n_train, n_val) == (0, 1)
    row = json.loads(val.read_text(encoding="utf-8").strip())
    assert row == {"instruction": "问", "output": answer}  # 记录完整未被切断


def test_train_smoke(tmp_path):
    train_path = tmp_path / "sft_train.jsonl"
    val_path = tmp_path / "sft_val.jsonl"
    rows = [{"instruction": f"问题{i}", "output": f"回答{i}"} for i in range(40)]
    _write_jsonl(train_path, rows)
    _write_jsonl(val_path, rows[:5])

    # 随机初始化一个 ckpt 作为 SFT 起点（结构契约同 pretrain.ckpt）
    init_ckpt = tmp_path / "pretrain.ckpt"
    cfg = ModelConfig()
    torch.save({"model": LlamaModel(cfg).state_dict(), "optimizer": {}, "step": 0,
                "config": asdict(cfg)}, init_ckpt)

    ckpt_dir = tmp_path / "ckpt"
    log = tmp_path / "log.jsonl"
    train_main([
        "--train", str(train_path), "--val", str(val_path),
        "--init-ckpt", str(init_ckpt), "--ckpt-dir", str(ckpt_dir),
        "--log", str(log), "--micro-batch", "4", "--grad-accum", "1",
        "--warmup-steps", "1", "--eval-interval", "5", "--eval-batches", "2",
        "--save-interval", "5", "--log-interval", "1", "--device", "cpu",
    ])

    # ckpt 契约：可加载回原模型且前向形状正确
    ckpt = torch.load(ckpt_dir / "sft.ckpt", map_location="cpu", weights_only=False)
    assert {"model", "optimizer", "step", "config"} <= ckpt.keys()
    assert ckpt["step"] == 10  # 40 样本 ÷ micro_batch 4 = 10 步
    model = LlamaModel(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    assert model(torch.zeros(1, 8, dtype=torch.long)).shape == (1, 8, 32000)
    assert (ckpt_dir / "sft_last.ckpt").exists()

    # 日志契约：train/eval 条目字段齐全，含监督 token 计数
    records = [json.loads(line) for line in open(log, encoding="utf-8")]
    train_rec = next(r for r in records if r["type"] == "train")
    eval_rec = next(r for r in records if r["type"] == "eval")
    assert {"step", "train_loss", "lr", "grad_norm", "tokens_seen",
            "sup_tokens_seen", "elapsed_s"} <= train_rec.keys()
    assert train_rec["sup_tokens_seen"] > 0
    assert {"step", "val_loss", "val_ppl"} <= eval_rec.keys()
