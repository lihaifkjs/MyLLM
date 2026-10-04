"""预训练主循环：next-token 预测，含 eval / ckpt / 断点续训 / 可观测性。

产物契约（docs/05-预训练.md）：
- checkpoints/pretrain.ckpt（最终）与 pretrain_last.ckpt（定期，用于续训）
- logs/pretrain_log.jsonl（train/eval 指标，plot_curve.py 的数据源）
"""
from __future__ import annotations

import argparse
import json
import math
import time
from contextlib import nullcontext
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from src.model import LlamaModel, ModelConfig
from src.pretrain.dataset import TokenData

MAX_CONSECUTIVE_NAN = 5  # 连续 NaN/Inf 达到此次数判定为梯度爆炸，中断训练
SPIKE_FACTOR = 2.0       # loss 超过滑动均值该倍数时告警（疑似脏数据段）
EMA_DECAY = 0.05         # 训练 loss 滑动平均的更新权重


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="预训练（模块4）")
    ap.add_argument("--data", default="data/clean/tokens.bin")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--log", default="logs/pretrain_log.jsonl")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=6e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup-steps", type=int, default=None, help="默认 max(100, 2% total)")
    ap.add_argument("--total-steps", type=int, default=None, help="默认按 1 epoch 推出")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--val-ratio", type=float, default=0.01)
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--save-interval", type=int, default=500)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--device", default=None, help="默认自动选择 cuda > mps > cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="从 pretrain_last.ckpt 续训")
    return ap.parse_args(argv)


def lr_at(step: int, total: int, warmup: int, peak: float, floor: float) -> float:
    """线性 warmup 后 cosine 衰减到 floor。"""
    if step < warmup:
        return peak * (step + 1) / warmup
    t = (step - warmup) / max(1, total - warmup)
    return floor + 0.5 * (peak - floor) * (1 + math.cos(math.pi * t))


def pick_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_ctx(device: torch.device):
    """CUDA 用 bf16 混合精度；MPS/CPU 保持 fp32（M2 上 bf16 算子支持不全）。"""
    if device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()


def build_optimizer(model: torch.nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    """weight decay 只作用于 ≥2D 的权重矩阵，norm/embedding 不衰减。"""
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    groups = [{"params": decay, "weight_decay": weight_decay},
              {"params": no_decay, "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95))


@torch.no_grad()
def evaluate(model, data: TokenData, device, batch_size: int, seq_len: int, num_batches: int):
    model.eval()
    losses = []
    for x, y in data.val_batches(batch_size, seq_len, num_batches):
        x, y = x.to(device), y.to(device)
        with autocast_ctx(device):
            loss = F.cross_entropy(model(x).reshape(-1, model.cfg.vocab_size), y.reshape(-1))
        losses.append(loss.item())
    model.train()
    mean = sum(losses) / max(1, len(losses))
    return mean, math.exp(min(mean, 20.0))  # 截断防止初期 ppl 溢出


def save_ckpt(path: Path, model, optimizer, step: int, cfg: ModelConfig) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "step": step, "config": asdict(cfg)}, path)


def try_make_tb_writer(log_path: Path):
    """可选 tensorboard：装了则同步打点，未安装自动跳过。"""
    try:
        from torch.utils.tensorboard import SummaryWriter
        return SummaryWriter(log_dir=str(log_path.parent / "tb"))
    except ImportError:
        return None


def main(argv=None) -> None:
    args = parse_args(argv)

    # ---- 启动自检：前置条件缺失时给出具体错因 ----
    if not Path(args.data).exists():
        raise FileNotFoundError(f"{args.data} 不存在，请先运行：python -m src.pretrain.encode_corpus")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    data = TokenData(args.data, args.val_ratio)

    tokens_per_step = args.micro_batch * args.grad_accum * args.seq_len
    total_steps = args.total_steps or len(data.train) // tokens_per_step
    if total_steps <= 0:
        raise ValueError(f"训练 token 数 {len(data.train)} 不足 1 step（{tokens_per_step}）")
    if args.total_steps and args.total_steps * tokens_per_step > len(data.train):
        print(f"警告：{args.total_steps} 步超过 1 epoch，数据将被重复采样")
    warmup = args.warmup_steps or max(100, total_steps // 50)

    cfg = ModelConfig()
    model = LlamaModel(cfg).to(device)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    start_step = 0
    if args.resume:
        last = ckpt_dir / "pretrain_last.ckpt"
        if not last.exists():
            raise FileNotFoundError(f"--resume 指定但 {last} 不存在")
        ckpt = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"从 step {start_step} 续训")

    print(f"设备={device} | 精度={'bf16' if device.type == 'cuda' else 'fp32'} | "
          f"tokens/step={tokens_per_step:,} | 总步数={total_steps} | warmup={warmup}")

    tb = try_make_tb_writer(log_path)
    log_f = open(log_path, "a", encoding="utf-8")

    def record(entry: dict) -> None:
        log_f.write(json.dumps(entry) + "\n")
        log_f.flush()
        if tb:
            for k, v in entry.items():
                if isinstance(v, (int, float)) and k != "step":
                    tb.add_scalar(f"{entry['type']}/{k}", v, entry["step"])

    ema_loss = None
    nan_streak = 0
    t0 = time.time()
    t_prev = t0
    model.train()
    try:
        with tqdm(range(start_step, total_steps), desc="pretrain", unit="step") as bar:
            for step in bar:
                lr = lr_at(step, total_steps, warmup, args.lr, args.min_lr)
                for g in optimizer.param_groups:
                    g["lr"] = lr

                # ---- 梯度累积：每个 optimizer step 前向/反向 grad_accum 个 micro batch ----
                optimizer.zero_grad(set_to_none=True)
                step_loss = 0.0
                for _ in range(args.grad_accum):
                    x, y = data.random_batch(args.micro_batch, args.seq_len, rng)
                    x, y = x.to(device), y.to(device)
                    with autocast_ctx(device):
                        logits = model(x)
                        loss = F.cross_entropy(
                            logits.reshape(-1, cfg.vocab_size), y.reshape(-1)) / args.grad_accum
                    loss.backward()
                    step_loss += loss.item()

                # ---- NaN/Inf 守卫：跳过本步，连续超限判定梯度爆炸 ----
                if not math.isfinite(step_loss):
                    nan_streak += 1
                    tqdm.write(f"警告：step {step} loss={step_loss}，已跳过（连续 {nan_streak} 次）")
                    if nan_streak >= MAX_CONSECUTIVE_NAN:
                        raise RuntimeError(
                            f"连续 {MAX_CONSECUTIVE_NAN} 次 NaN/Inf，疑似梯度爆炸；"
                            f"建议降低 --lr 或用 --resume 从最近 ckpt 续训")
                    continue
                nan_streak = 0

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()

                # ---- loss spike 告警（在 EMA 更新前比较）----
                if ema_loss is not None and step_loss > SPIKE_FACTOR * ema_loss:
                    tqdm.write(f"警告：step {step} loss={step_loss:.3f} 超过滑动均值 "
                               f"{SPIKE_FACTOR} 倍，疑似脏数据段")
                ema_loss = step_loss if ema_loss is None else (1 - EMA_DECAY) * ema_loss + EMA_DECAY * step_loss

                now = time.time()
                tokens_s = tokens_per_step / max(now - t_prev, 1e-9)
                t_prev = now
                bar.set_postfix(loss=f"{ema_loss:.3f}", lr=f"{lr:.2e}", tok_s=f"{tokens_s:.2e}")

                if (step + 1) % args.log_interval == 0 or step == start_step:
                    record({"type": "train", "step": step + 1, "train_loss": round(ema_loss, 4),
                            "lr": lr, "grad_norm": float(grad_norm),
                            "tokens_seen": (step + 1) * tokens_per_step,
                            "elapsed_s": round(time.time() - t0, 1)})

                if (step + 1) % args.eval_interval == 0 or step + 1 == total_steps:
                    val_loss, val_ppl = evaluate(model, data, device, args.micro_batch,
                                                 args.seq_len, args.eval_batches)
                    record({"type": "eval", "step": step + 1, "val_loss": round(val_loss, 4),
                            "val_ppl": round(val_ppl, 2)})
                    tqdm.write(f"step {step + 1}: val_loss={val_loss:.4f} ppl={val_ppl:.2f}")

                if (step + 1) % args.save_interval == 0:
                    save_ckpt(ckpt_dir / "pretrain_last.ckpt", model, optimizer, step + 1, cfg)

    except KeyboardInterrupt:
        save_ckpt(ckpt_dir / "pretrain_last.ckpt", model, optimizer, step + 1, cfg)
        print(f"\n已中断，进度保存至 {ckpt_dir}/pretrain_last.ckpt（step {step + 1}），可用 --resume 续训")
        return
    except torch.cuda.OutOfMemoryError:
        print("CUDA 显存不足（OOM）：请调小 --micro-batch 或调大 --grad-accum 保持 tokens/step 不变")
        raise
    finally:
        log_f.close()
        if tb:
            tb.close()

    save_ckpt(ckpt_dir / "pretrain.ckpt", model, optimizer, total_steps, cfg)
    save_ckpt(ckpt_dir / "pretrain_last.ckpt", model, optimizer, total_steps, cfg)
    print(f"训练完成：{ckpt_dir}/pretrain.ckpt")


if __name__ == "__main__":
    main()
