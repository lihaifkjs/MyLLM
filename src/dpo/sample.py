"""加载 sft.ckpt 与 dpo.ckpt，对同一批问题分别贪心生成并排对照，供人工抽查（正式评测属模块7）。"""
from __future__ import annotations

import argparse

import torch

from src.model import LlamaModel, ModelConfig
from src.pretrain.train import pick_device
from src.sft.sample import DEFAULT_QUESTIONS, generate
from src.tokenizer.tokenizer import Tokenizer


def load_model(ckpt_path: str, device: torch.device) -> LlamaModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = LlamaModel(ModelConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    return model


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="DPO 对照抽查（模块6）")
    ap.add_argument("--sft-ckpt", default="checkpoints/sft.ckpt")
    ap.add_argument("--dpo-ckpt", default="checkpoints/dpo.ckpt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.model")
    ap.add_argument("--question", action="append", help="自定义问题，可多次传入；默认用内置 5 题")
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--device", default=None, help="默认自动选择 cuda > mps > cpu")
    args = ap.parse_args(argv)

    device = pick_device(args.device)
    tok = Tokenizer(args.tokenizer)
    sft_model = load_model(args.sft_ckpt, device)
    dpo_model = load_model(args.dpo_ckpt, device)

    for q in args.question or DEFAULT_QUESTIONS:
        print(f"用户：{q}")
        print(f"[SFT] {generate(sft_model, tok, q, device, args.max_new_tokens)}")
        print(f"[DPO] {generate(dpo_model, tok, q, device, args.max_new_tokens)}\n")


if __name__ == "__main__":
    main()
