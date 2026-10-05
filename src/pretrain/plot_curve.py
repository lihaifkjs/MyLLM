"""读取 pretrain_log.jsonl，绘制 train/val loss 与 val ppl 曲线 → pretrain_curve.png。"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")  # 无显示环境下也能出图
import matplotlib.pyplot as plt


def main() -> None:
    ap = argparse.ArgumentParser(description="绘制预训练 loss/ppl 曲线")
    ap.add_argument("--log", default="logs/pretrain_log.jsonl")
    ap.add_argument("--out", default="logs/pretrain_curve.png")
    args = ap.parse_args()

    train_steps, train_loss, eval_steps, val_loss, val_ppl = [], [], [], [], []
    with open(args.log, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            if r["type"] == "train":
                train_steps.append(r["step"])
                train_loss.append(r["train_loss"])
            elif r["type"] == "eval":
                eval_steps.append(r["step"])
                val_loss.append(r["val_loss"])
                if "val_ppl" in r:  # DPO 等日志无 ppl，仅在有该字段时画 ppl 面板
                    val_ppl.append(r["val_ppl"])

    if val_ppl:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        ax2.plot(eval_steps, val_ppl, "o-", color="tab:red")
        ax2.set(xlabel="step", ylabel="ppl", title="Val Perplexity")
    else:
        fig, ax1 = plt.subplots(figsize=(6, 4))
    ax1.plot(train_steps, train_loss, label="train")
    ax1.plot(eval_steps, val_loss, "o-", label="val")
    ax1.set(xlabel="step", ylabel="loss", title="Loss")
    ax1.legend()
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"曲线已保存：{args.out}")


if __name__ == "__main__":
    main()
