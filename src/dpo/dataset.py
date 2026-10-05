"""DPO 偏好对编码与 batch 组装。

模板、监督区间与截断规则同 docs/06-SFT（复用 src/sft/dataset.encode_sample）：
chosen / rejected 共用同一 max_len 预算各自编码，监督区间仅 answer + <eos>。
一个 batch 含 2N 条序列：前 N 条为 chosen，后 N 条为 rejected（顺序是 train.py 切分的依据）。
"""
from __future__ import annotations

import json

from torch.utils.data import Dataset

from src.sft.dataset import collate_batch, encode_sample
from src.tokenizer.tokenizer import Tokenizer


def encode_pair(tok: Tokenizer, instruction: str, chosen: str, rejected: str, max_len: int):
    """编码一个偏好对，返回 ((c_ids, c_labels), (r_ids, r_labels))；prompt 超长返回 None。"""
    c = encode_sample(tok, instruction, chosen, max_len)
    r = encode_sample(tok, instruction, rejected, max_len)
    if c is None or r is None:
        return None
    return c, r


class DPODataset(Dataset):
    """加载 dpo jsonl 并预编码为偏好对列表（3 万对短文本，内存可放下）。"""

    def __init__(self, jsonl_path: str, tok: Tokenizer, max_len: int = 1024):
        self.pairs: list[tuple[tuple, tuple]] = []
        n_drop = 0
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                p = encode_pair(tok, r["instruction"], r["chosen"], r["rejected"], max_len)
                if p is None:
                    n_drop += 1
                    continue
                self.pairs.append(p)
        if n_drop:
            print(f"{jsonl_path}: prompt 超长丢弃 {n_drop} 对")

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, i):
        return self.pairs[i]


def collate_pairs(pairs: list, pad_id: int, pad_multiple: int = 64):
    """展开为 2N 条序列（前 N chosen，后 N rejected）后统一右 pad；padding 语义同 src/sft collate_batch。"""
    flat = [c for c, _ in pairs] + [r for _, r in pairs]
    return collate_batch(flat, pad_id, pad_multiple)
