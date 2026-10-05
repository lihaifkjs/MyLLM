"""加载 sft.ckpt，对固定问题贪心生成，供人工抽查（正式评测属模块7）。"""
from __future__ import annotations

import argparse

import torch

from src.model import LlamaModel, ModelConfig
from src.sft.dataset import PROMPT_TEMPLATE
from src.tokenizer.tokenizer import Tokenizer

DEFAULT_QUESTIONS = [
    "失眠了怎么办？",
    "中国的首都是哪里？",
    "请解释什么是光合作用。",
    "写一句关于春天的诗。",
    "感冒了应该注意什么？",
]


@torch.no_grad()
def generate(model: LlamaModel, tok: Tokenizer, question: str,
             device: torch.device, max_new_tokens: int = 128) -> str:
    """贪心解码：逐 token 取 argmax，遇到 <eos> 停止。"""
    ids = tok.encode(PROMPT_TEMPLATE.format(instruction=question), add_bos=True)
    prompt_len = len(ids)
    model.eval()
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.cfg.max_seq_len:]], dtype=torch.long, device=device)
        next_id = int(model(x)[0, -1].argmax())
        if next_id == tok.eos_id:
            break
        ids.append(next_id)
    return tok.decode(ids[prompt_len:])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="SFT 抽查生成（模块5）")
    ap.add_argument("--ckpt", default="checkpoints/sft.ckpt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.model")
    ap.add_argument("--question", action="append", help="自定义问题，可多次传入；默认用内置 5 题")
    ap.add_argument("--max-new-tokens", type=int, default=128)
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

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    model = LlamaModel(ModelConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    tok = Tokenizer(args.tokenizer)

    for q in args.question or DEFAULT_QUESTIONS:
        print(f"用户：{q}\n助手：{generate(model, tok, q, device, args.max_new_tokens)}\n")


if __name__ == "__main__":
    main()
