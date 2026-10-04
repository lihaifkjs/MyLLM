"""tokens.bin 的 memmap 读取与训练/验证窗口采样。

训练集随机起点切窗口；验证集取末尾连续 val_ratio 的 token，不重叠顺序切窗，
保证 ppl 可复现。窗口长 seq_len+1，x 与 y 错位一位构成 next-token 预测对。
"""
import json
from pathlib import Path

import numpy as np
import torch


class TokenData:
    def __init__(self, bin_path: str, val_ratio: float = 0.01):
        bin_path = Path(bin_path)
        meta = json.loads(bin_path.with_suffix(".meta.json").read_text(encoding="utf-8"))
        tokens = np.memmap(bin_path, dtype=np.uint16, mode="r", shape=(meta["num_tokens"],))
        n_val = int(len(tokens) * val_ratio)
        self.train = tokens[: len(tokens) - n_val]
        self.val = tokens[len(tokens) - n_val:]

    def random_batch(self, batch_size: int, seq_len: int, rng: np.random.Generator):
        hi = len(self.train) - (seq_len + 1)
        if hi <= 0:
            raise ValueError(f"训练 token 数 {len(self.train)} 不足一个窗口（需 > {seq_len + 1}）")
        starts = rng.integers(0, hi, size=batch_size)
        buf = np.stack([self.train[s: s + seq_len + 1] for s in starts]).astype(np.int64)
        return torch.from_numpy(buf[:, :-1]), torch.from_numpy(buf[:, 1:])

    def val_batches(self, batch_size: int, seq_len: int, max_batches: int):
        n_win = (len(self.val) - 1) // seq_len
        starts = [i * seq_len for i in range(n_win)]
        for i in range(0, min(len(starts), batch_size * max_batches), batch_size):
            sel = starts[i: i + batch_size]
            buf = np.stack([self.val[s: s + seq_len + 1] for s in sel]).astype(np.int64)
            yield torch.from_numpy(buf[:, :-1]), torch.from_numpy(buf[:, 1:])
