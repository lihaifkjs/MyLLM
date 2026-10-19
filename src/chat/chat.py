"""命令行交互式多轮对话（模块7，docs/08）。

模板继承 docs/06：<bos>用户：{q}\n助手：{a}<eos>，历史轮直接拼接。
超 max_seq_len 时从最早一轮开始丢弃（docs/08 决策 4）；单轮问题本身超长则硬截尾部。
输入 /exit 退出，/reset 清空历史。
"""
from __future__ import annotations

import argparse

import torch

from src.chat.decoding import generate
from src.model import LlamaModel, ModelConfig
from src.tokenizer.tokenizer import Tokenizer


def build_prompt_ids(tok: Tokenizer, history: list[tuple[str, str]],
                     question: str, max_len: int) -> list[int]:
    """把多轮历史 + 当前问题渲染成 prompt token；超长丢弃最早轮。"""
    hist = list(history)

    def render() -> list[int]:
        text = "".join(f"用户：{q}\n助手：{a}<eos>" for q, a in hist)
        return [tok.bos_id] + tok.encode(text + f"用户：{question}\n助手：")

    ids = render()
    while len(ids) > max_len and hist:
        hist.pop(0)  # 丢弃最早一轮，模板其余部分不受影响
        ids = render()
    if len(ids) > max_len:  # 只有当前问题就已超长：硬截窗口尾部
        ids = ids[-max_len:]
    return ids


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="命令行多轮对话（模块7）")
    ap.add_argument("--ckpt", default="checkpoints/dpo.ckpt")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.model")
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.9)
    ap.add_argument("--repetition-penalty", type=float, default=1.1)
    ap.add_argument("--no-repeat-ngram", type=int, default=3)
    ap.add_argument("--greedy", action="store_true", help="贪心解码（默认采样+复读缓解）")
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
    temperature = 0.0 if args.greedy else args.temperature
    print(f"已加载 {args.ckpt}（复读缓解：temp={args.temperature}, top_p={args.top_p}, "
          f"rep_penalty={args.repetition_penalty}, no_repeat_ngram={args.no_repeat_ngram}）\n"
          "输入 /exit 退出，/reset 清空历史")

    history: list[tuple[str, str]] = []
    while True:
        try:
            question = input("\n用户：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见！")
            break
        if question == "/exit":
            break
        if question == "/reset":
            history.clear()
            print("（历史已清空）")
            continue
        if not question:
            continue
        prompt_ids = build_prompt_ids(tok, history, question, model.cfg.max_seq_len)
        new_ids = generate(model, tok, prompt_ids, device,
                           max_new_tokens=args.max_new_tokens, temperature=temperature,
                           top_p=args.top_p, repetition_penalty=args.repetition_penalty,
                           no_repeat_ngram=args.no_repeat_ngram)
        answer = tok.decode(new_ids)
        print(f"助手：{answer}")
        history.append((question, answer))


if __name__ == "__main__":
    main()
