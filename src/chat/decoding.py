"""解码策略：贪心 / top-p 采样 + 复读缓解（repetition_penalty、no_repeat_ngram）。

供 src/chat/chat.py 与 src/eval/case_compare.py 共用（docs/08）。
复读缓解的统计区间包含 prompt + 已生成部分，防止模型复读输入文本。
"""
from __future__ import annotations

import torch


def _apply_repetition_penalty(logits: torch.Tensor, seen: list[int], penalty: float) -> None:
    """HF 风格：已出现 token 的 logits 除以 penalty（>1 降低其再被选中的概率）。"""
    if penalty <= 1.0 or not seen:
        return
    logits[seen] /= penalty


def _banned_by_ngram(seen: list[int], n: int, candidate: int) -> bool:
    """candidate 是否会与 seen 末尾组成一个已出现过的 n-gram。"""
    if n <= 0 or len(seen) < n - 1:
        return False
    tail = seen[-(n - 1):] + [candidate]
    return any(seen[i:i + n] == tail for i in range(len(seen) - n + 1))


@torch.no_grad()
def generate(model, tok, prompt_ids: list[int], device: torch.device,
             max_new_tokens: int = 200, temperature: float = 0.0,
             top_p: float = 1.0, repetition_penalty: float = 1.0,
             no_repeat_ngram: int = 0) -> list[int]:
    """自回归生成，返回新生成的 token id 列表（不含 prompt，遇 <eos> 停止）。

    temperature=0 表示贪心；>0 表示采样（先按 temperature 缩放、
    再 top-p 截断、最后 multinomial 采样）。
    """
    ids = list(prompt_ids)
    seen = list(prompt_ids)  # 复读缓解统计区间：prompt + 已生成
    model.eval()
    new_tokens: list[int] = []
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.cfg.max_seq_len:]], dtype=torch.long, device=device)
        logits = model(x)[0, -1].float()
        _apply_repetition_penalty(logits, seen, repetition_penalty)
        if no_repeat_ngram > 0:
            for cand in range(logits.shape[0]):
                if _banned_by_ngram(seen, no_repeat_ngram, cand):
                    logits[cand] = float("-inf")
        if temperature <= 0:
            next_id = int(logits.argmax())
        else:
            probs = torch.softmax(logits / temperature, dim=-1)
            if top_p < 1.0:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cum = torch.cumsum(sorted_probs, dim=-1)
                # 保留累计概率首次达到 top_p 为止的前缀（cum - p < top_p 等价于前缀截断）
                keep = (cum - sorted_probs) < top_p
                keep[0] = True  # 至少保留概率最大的一项
                probs = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
                probs = probs / probs.sum()
                next_id = int(sorted_idx[int(torch.multinomial(probs, 1))])
            else:
                next_id = int(torch.multinomial(probs, 1))
        if next_id == tok.eos_id:
            break
        ids.append(next_id)
        seen.append(next_id)
        new_tokens.append(next_id)
    return new_tokens
