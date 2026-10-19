"""统一困惑度入口：对任意 ckpt 在同一评测语料上算 ppl（docs/08）。

评测语料（--source 可强制，默认按 ckpt 文件名推断）：
- pretrain：data/clean/tokens.bin 末尾 --max-tokens 个 token（尾部大概率被 1 epoch 训练见过，
  教学向可接受；跨 ckpt 可对比是首要目标）
- sft：data/sft_val.jsonl 的回答字段逐条编码（每条末尾补 <eos>）后拼接，
  与 SFT 监督区间一致，更贴近下游任务

算法：非重叠窗口 (seq_len+1) 错位一位构成 next-token 对，batch 前向，
ppl = exp(总 NLL / 预测 token 数)。结果追加到 logs/ppl_report.jsonl，多次运行自动累积。
"""
from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

import numpy as np
import torch

from src.model import LlamaModel, ModelConfig
from src.tokenizer.tokenizer import Tokenizer


def load_eval_ids(source: str, tok: Tokenizer, max_tokens: int,
                  tokens_bin: str = "data/clean/tokens.bin",
                  sft_val: str = "data/sft_val.jsonl") -> list[int]:
    """按 source 取评测 token 序列（截到 max_tokens）。"""
    if source == "pretrain":
        meta = json.loads(Path(tokens_bin).with_suffix(".meta.json").read_text(encoding="utf-8"))
        tokens = np.memmap(tokens_bin, dtype=np.uint16, mode="r", shape=(meta["num_tokens"],))
        return [int(t) for t in tokens[-max_tokens:]]
    if source == "sft":
        ids: list[int] = []
        with open(sft_val, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                ids.extend(tok.encode(r["output"]) + [tok.eos_id])
                if len(ids) >= max_tokens:
                    break
        return ids[:max_tokens]
    raise ValueError(f"未知 source：{source}（可选 pretrain / sft）")


@torch.no_grad()
def compute_nll(model: LlamaModel, ids: list[int], device: torch.device,
                seq_len: int, batch_size: int = 8) -> tuple[float, int]:
    """teacher forcing：返回 (总 NLL, 预测 token 数)。尾部不足一个窗口的部分丢弃。"""
    n_win = (len(ids) - 1) // seq_len
    if n_win == 0:
        raise ValueError(f"评测 token 数 {len(ids)} 不足一个窗口（需 > {seq_len + 1}）")
    total_nll, total_tok = 0.0, 0
    for i in range(0, n_win, batch_size):
        buf = np.stack([ids[j * seq_len: j * seq_len + seq_len + 1]
                        for j in range(i, min(i + batch_size, n_win))]).astype(np.int64)
        x = torch.from_numpy(buf[:, :-1]).to(device)
        y = torch.from_numpy(buf[:, 1:]).to(device)
        logits = model(x)
        total_nll += torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction="sum").item()
        total_tok += y.numel()
    return total_nll, total_tok


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="统一困惑度评测（模块7）")
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", choices=["auto", "pretrain", "sft"], default="auto")
    ap.add_argument("--max-tokens", type=int, default=200_000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--log", default="logs/ppl_report.jsonl")
    ap.add_argument("--device", default=None, help="默认自动选择 cuda > mps > cpu")
    args = ap.parse_args(argv)

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    source = args.source
    if source == "auto":
        source = "pretrain" if "pretrain" in Path(args.ckpt).name else "sft"

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = LlamaModel(ModelConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    tok = Tokenizer()

    ids = load_eval_ids(source, tok, args.max_tokens)
    nll, n_tok = compute_nll(model, ids, device, model.cfg.max_seq_len, args.batch_size)
    ppl = float(np.exp(nll / n_tok))

    record = {"ckpt": args.ckpt, "source": source, "ppl": round(ppl, 4),
              "n_tokens": n_tok,
              "date": datetime.datetime.now().isoformat(timespec="seconds")}
    Path(args.log).parent.mkdir(parents=True, exist_ok=True)
    with open(args.log, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"{args.ckpt}（{source}，{n_tok} tokens）ppl = {ppl:.4f}，已追加到 {args.log}")


if __name__ == "__main__":
    main()
