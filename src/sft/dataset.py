"""SFT 样本编码与 batch 组装。

样本模板（docs/06-SFT.md）：
    <bos>用户：{instruction}\\n助手：{output}<eos>
监督区间：仅 {output} + <eos>；prompt（含 <bos>）与 padding 一律 label=-100。
超长样本保留完整 prompt、只截断 output；prompt 本身超长则整条丢弃。
"""
from __future__ import annotations

import json

import torch
from torch.utils.data import Dataset

from src.tokenizer.tokenizer import Tokenizer

PROMPT_TEMPLATE = "用户：{instruction}\n助手："


def encode_sample(tok: Tokenizer, instruction: str, output: str, max_len: int):
    """拼模板并编码，返回 (input_ids, labels)；prompt 超长返回 None。"""
    prompt_ids = [tok.bos_id] + tok.encode(PROMPT_TEMPLATE.format(instruction=instruction))
    budget = max_len - len(prompt_ids) - 1  # 末尾固定留给 <eos>
    if budget < 0:  # prompt 已占满 max_len，连 <eos> 都放不下
        return None
    answer_ids = tok.encode(output)[:budget] + [tok.eos_id]
    input_ids = prompt_ids + answer_ids
    labels = [-100] * len(prompt_ids) + answer_ids
    return input_ids, labels


class SFTDataset(Dataset):
    """加载 sft jsonl 并预编码为 (input_ids, labels) 列表（10 万条短文本，内存可放下）。"""

    def __init__(self, jsonl_path: str, tok: Tokenizer, max_len: int = 1024):
        self.samples: list[tuple[list[int], list[int]]] = []
        n_drop = 0
        with open(jsonl_path, encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                s = encode_sample(tok, r["instruction"], r["output"], max_len)
                if s is None:
                    n_drop += 1
                    continue
                self.samples.append(s)
        if n_drop:
            print(f"{jsonl_path}: prompt 超长丢弃 {n_drop} 条")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int):
        return self.samples[i]


def collate_batch(samples: list, pad_id: int, pad_multiple: int = 64):
    """右 padding 到 batch 内最长（向上对齐 pad_multiple）；pad 位 input=pad_id、label=-100。
    因果注意力下 pad 只出现在序列尾部且不计算 loss，不会污染监督 token。
    对齐到固定倍数是为了限制 batch 形状种类：MPS 缓存分配器对变长形状会碎片化囤积显存。
    """
    raw_max = max(len(ids) for ids, _ in samples)
    max_len = (raw_max + pad_multiple - 1) // pad_multiple * pad_multiple
    input_ids = torch.full((len(samples), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(samples), max_len), -100, dtype=torch.long)
    for i, (ids, lab) in enumerate(samples):
        input_ids[i, : len(ids)] = torch.tensor(ids)
        labels[i, : len(lab)] = torch.tensor(lab)
    return input_ids, labels
