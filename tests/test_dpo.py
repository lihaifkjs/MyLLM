"""模块6 DPO 冒烟测试：合成偏好对，CPU 跑数步，校验 mask / loss 数学 / ckpt / 日志契约。"""
import json
import math
from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from src.dpo.dataset import DPODataset, collate_pairs, encode_pair
from src.dpo.train import dpo_loss, response_logprob
from src.dpo.train import main as train_main
from src.model import LlamaModel, ModelConfig
from src.sft.dataset import PROMPT_TEMPLATE
from src.tokenizer.tokenizer import Tokenizer


@pytest.fixture(scope="module")
def tok():
    return Tokenizer("tokenizer/tokenizer.model")


def test_encode_pair_mask(tok):
    (c_ids, c_lab), (r_ids, r_lab) = encode_pair(tok, "失眠了怎么办？", "睡前泡脚。", "喝咖啡。", 1024)
    prompt_ids = [tok.bos_id] + tok.encode(PROMPT_TEMPLATE.format(instruction="失眠了怎么办？"))
    for ids, lab in ((c_ids, c_lab), (r_ids, r_lab)):
        assert ids[: len(prompt_ids)] == prompt_ids            # 同一 prompt
        assert all(l == -100 for l in lab[: len(prompt_ids)])  # prompt 不参与对数概率和
        assert lab[len(prompt_ids):] == ids[len(prompt_ids):]  # answer + <eos> 监督
        assert ids[-1] == lab[-1] == tok.eos_id
    assert c_ids != r_ids


def test_encode_pair_truncation(tok):
    # chosen/rejected 分别截断到同一预算，都保证 <eos> 收尾
    (c_ids, c_lab), (r_ids, r_lab) = encode_pair(tok, "讲个故事", "很长" * 500, "短", 64)
    assert len(c_ids) == len(r_ids) == 64 or len(r_ids) < 64
    assert len(c_ids) == 64 and c_ids[-1] == c_lab[-1] == tok.eos_id
    assert r_ids[-1] == r_lab[-1] == tok.eos_id
    # prompt 本身就超长：整对丢弃
    assert encode_pair(tok, "很长" * 500, "好", "坏", 64) is None


def test_collate_pairs(tok):
    pairs = [encode_pair(tok, "天为什么是蓝的？", "因为瑞利散射。", "不知道。", 128),
             encode_pair(tok, "1+1=?", "2", "3", 128)]
    input_ids, labels = collate_pairs(pairs, tok.pad_id)
    # 2N 条序列：前 N chosen，后 N rejected；padding 向上对齐 64
    assert input_ids.shape == labels.shape == (4, input_ids.shape[1])
    assert input_ids.shape[1] % 64 == 0
    assert input_ids[0].tolist()[: len(pairs[0][0][0])] == pairs[0][0][0]  # 第 0 条是 chosen
    assert input_ids[2].tolist()[: len(pairs[0][1][0])] == pairs[0][1][0]  # 第 N 条是 rejected


def test_response_logprob():
    # 手工构造：vocab=4，序列长 4；labels 只监督最后两位（含 eos 位）
    torch.manual_seed(0)
    logits = torch.randn(1, 4, 4)
    labels = torch.tensor([[-100, -100, 2, 3]])
    got = response_logprob(logits, labels)
    logp = F.log_softmax(logits.float(), dim=-1)
    # next-token 对齐：第 t 位监督 tgt=labels[t+1]，用 logits[t] 打分
    expect = logp[0, 1, 2] + logp[0, 2, 3]
    assert torch.allclose(got[0], expect)


def test_dpo_loss_math():
    beta = 0.1
    ref = torch.zeros(2)  # ref 对 chosen/rejected 的 logprob 和都为 0
    # policy 与 ref 相同 → reward=0，loss=log 2，acc=0（不严格偏好 chosen）
    loss, margin, acc = dpo_loss(torch.zeros(2), torch.zeros(2), ref, ref, beta)
    assert torch.allclose(loss, torch.tensor(math.log(2)))
    assert margin.item() == 0.0 and acc.item() == 0.0
    # policy 更偏好 chosen（+1 vs -1）→ reward>0，loss < log 2，acc=1
    loss, margin, acc = dpo_loss(torch.ones(2), -torch.ones(2), ref, ref, beta)
    assert margin.item() == pytest.approx(2 * beta)
    assert loss.item() < math.log(2) and acc.item() == 1.0
    # policy 反向偏好 → acc=0，loss > log 2
    loss, _, acc = dpo_loss(-torch.ones(2), torch.ones(2), ref, ref, beta)
    assert loss.item() > math.log(2) and acc.item() == 0.0


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def test_train_smoke(tmp_path):
    train_path = tmp_path / "dpo_train.jsonl"
    val_path = tmp_path / "dpo_val.jsonl"
    rows = [{"instruction": f"问题{i}", "chosen": f"好回答{i}", "rejected": f"坏回答{i}"}
            for i in range(40)]
    _write_jsonl(train_path, rows)
    _write_jsonl(val_path, rows[:5])

    # 随机初始化一个 ckpt 作为 policy 与 ref 的共同起点（结构契约同 sft.ckpt）
    init_ckpt = tmp_path / "sft.ckpt"
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

    # ckpt 契约：只存 policy，可加载回原模型且前向形状正确
    ckpt = torch.load(ckpt_dir / "dpo.ckpt", map_location="cpu", weights_only=False)
    assert {"model", "optimizer", "step", "config"} <= ckpt.keys()
    assert ckpt["step"] == 10  # 40 对 ÷ micro_batch 4 对 = 10 步
    model = LlamaModel(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    assert model(torch.zeros(1, 8, dtype=torch.long)).shape == (1, 8, 32000)
    assert (ckpt_dir / "dpo_last.ckpt").exists()

    # 日志契约：train/eval 条目字段齐全，reward_acc 在 [0,1]
    records = [json.loads(line) for line in open(log, encoding="utf-8")]
    train_rec = next(r for r in records if r["type"] == "train")
    eval_rec = next(r for r in records if r["type"] == "eval")
    assert {"step", "train_loss", "reward_margin", "reward_acc", "lr", "grad_norm",
            "tokens_seen", "elapsed_s"} <= train_rec.keys()
    assert 0.0 <= train_rec["reward_acc"] <= 1.0
    assert {"step", "val_loss", "val_reward_acc"} <= eval_rec.keys()
    assert 0.0 <= eval_rec["val_reward_acc"] <= 1.0
