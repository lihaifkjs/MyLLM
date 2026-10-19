"""固定问题集 SFT vs DPO 生成对照表（模块7，docs/08）。

同一批问题分别用 sft.ckpt / dpo.ckpt 生成，两边用同一套解码参数（默认带复读缓解），
输出 Markdown 对照表到 logs/case_compare.md，作为 DPO 效果的人工抽查载体。
复读判定（启发式）：存在长度为 4 的字符子串重复出现 ≥3 次。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.chat.decoding import generate
from src.eval.questions import DEFAULT_QUESTIONS
from src.model import LlamaModel, ModelConfig
from src.sft.dataset import PROMPT_TEMPLATE
from src.tokenizer.tokenizer import Tokenizer


def is_repetitive(text: str, n: int = 4, times: int = 3) -> bool:
    """启发式复读检测：任意 n-gram 字符子串出现 ≥ times 次即判复读。"""
    grams: dict[str, int] = {}
    for i in range(len(text) - n + 1):
        grams[text[i:i + n]] = grams.get(text[i:i + n], 0) + 1
        if grams[text[i:i + n]] >= times:
            return True
    return False


def load_model(ckpt_path: str, device: torch.device) -> LlamaModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = LlamaModel(ModelConfig(**ckpt["config"])).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="SFT vs DPO case 对照表（模块7）")
    ap.add_argument("--sft-ckpt", default="checkpoints/sft.ckpt")
    ap.add_argument("--dpo-ckpt", default="checkpoints/dpo.ckpt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.model")
    ap.add_argument("--questions-file", default=None, help="自定义问题集（一行一题），默认内置 12 题")
    ap.add_argument("--out", default="logs/case_compare.md")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42, help="采样解码的随机种子（两边同种子，保证可比）")
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

    if args.questions_file:
        questions = [ln.strip() for ln in Path(args.questions_file).read_text(
            encoding="utf-8").splitlines() if ln.strip()]
    else:
        questions = DEFAULT_QUESTIONS

    tok = Tokenizer(args.tokenizer)
    torch.manual_seed(args.seed)  # 两边用同一随机种子，差异只来自权重
    outputs = {}
    for name, ckpt_path in (("SFT", args.sft_ckpt), ("DPO", args.dpo_ckpt)):
        model = load_model(ckpt_path, device)
        outputs[name] = []
        for q in questions:
            prompt_ids = [tok.bos_id] + tok.encode(PROMPT_TEMPLATE.format(instruction=q))
            new_ids = generate(model, tok, prompt_ids, device,
                               max_new_tokens=args.max_new_tokens,
                               temperature=args.temperature, top_p=args.top_p,
                               repetition_penalty=args.repetition_penalty,
                               no_repeat_ngram=args.no_repeat_ngram)
            outputs[name].append(tok.decode(new_ids).replace("\n", " "))

    sft_rep = sum(is_repetitive(t) for t in outputs["SFT"])
    dpo_rep = sum(is_repetitive(t) for t in outputs["DPO"])
    lines = [
        "# SFT vs DPO 生成对照（case_compare）",
        f"- 解码参数：temperature={args.temperature}, top_p={args.top_p}, "
        f"repetition_penalty={args.repetition_penalty}, no_repeat_ngram={args.no_repeat_ngram}, "
        f"max_new_tokens={args.max_new_tokens}, seed={args.seed}",
        f"- 复读率（4-gram 重复≥3 次）：SFT {sft_rep}/{len(questions)}，DPO {dpo_rep}/{len(questions)}",
        "",
        "| # | 问题 | SFT 输出 | DPO 输出 |",
        "|---|---|---|---|",
    ]
    for i, q in enumerate(questions):
        lines.append(f"| {i + 1} | {q} | {outputs['SFT'][i]} | {outputs['DPO'][i]} |")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"已生成 {args.out}（SFT 复读 {sft_rep} 题，DPO 复读 {dpo_rep} 题）")


if __name__ == "__main__":
    main()
