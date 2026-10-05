"""SFT 主循环：从 pretrain.ckpt 初始化（只载模型权重），loss 只算回答部分。

产物契约（docs/06-SFT.md）：
- checkpoints/sft.ckpt（最终）与 sft_last.ckpt（定期，用于续训）
- logs/sft_log.jsonl（train/eval 指标；train_loss 为监督 token 平均，另记监督 token 数）
训练骨架复用 src/pretrain/train.py 的公共函数（lr 调度/设备/精度/优化器/日志/ckpt）。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.model import LlamaModel, ModelConfig
from src.pretrain.train import (EMA_DECAY, MAX_CONSECUTIVE_NAN, SPIKE_FACTOR,
                                autocast_ctx, build_optimizer, lr_at, pick_device,
                                save_ckpt, try_make_tb_writer)
from src.sft.dataset import SFTDataset, collate_batch
from src.tokenizer.tokenizer import Tokenizer


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="SFT（模块5）")
    ap.add_argument("--train", default="data/sft_train.jsonl")
    ap.add_argument("--val", default="data/sft_val.jsonl")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.model")
    ap.add_argument("--init-ckpt", default="checkpoints/pretrain.ckpt", help="只加载模型权重，优化器重建")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--log", default="logs/sft_log.jsonl")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=16)
    ap.add_argument("--grad-accum", type=int, default=8, help="默认 16×8=128 样本/step")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=2e-5)
    ap.add_argument("--min-lr", type=float, default=1e-6)
    ap.add_argument("--warmup-steps", type=int, default=None, help="默认 3%% total_steps")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--save-interval", type=int, default=200)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--device", default=None, help="默认自动选择 cuda > mps > cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="从 sft_last.ckpt 续训")
    return ap.parse_args(argv)


def supervised_loss(logits: torch.Tensor, labels: torch.Tensor, vocab_size: int):
    """labels 右移一位做 next-token 对齐，ignore_index=-100 只统计回答部分。
    返回 (loss_sum, n_sup)：由调用方跨 micro batch 汇总成"监督 token 平均"。
    """
    tgt = labels[:, 1:]
    n_sup = int((tgt != -100).sum())
    loss_sum = F.cross_entropy(logits[:, :-1].reshape(-1, vocab_size), tgt.reshape(-1),
                               ignore_index=-100, reduction="sum")
    return loss_sum, n_sup


@torch.no_grad()
def evaluate(model, val_ds: SFTDataset, device, batch_size: int, max_batches: int, pad_id: int):
    model.eval()
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate_batch(b, pad_id))
    loss_sum, n_sup = 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast_ctx(device):
            s, n = supervised_loss(model(x), y, model.cfg.vocab_size)
        loss_sum += s.item()
        n_sup += n
    model.train()
    mean = loss_sum / max(1, n_sup)
    return mean, math.exp(min(mean, 20.0))  # 截断防止初期 ppl 溢出


def main(argv=None) -> None:
    args = parse_args(argv)

    # ---- 启动自检 ----
    for p in (args.train, args.val):
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} 不存在，请先运行：python -m src.sft.prepare_data")
    if not args.resume and not Path(args.init_ckpt).exists():
        raise FileNotFoundError(f"初始化权重 {args.init_ckpt} 不存在（模块4 产物）")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    tok = Tokenizer(args.tokenizer)
    train_ds = SFTDataset(args.train, tok, args.max_len)
    val_ds = SFTDataset(args.val, tok, args.max_len)

    samples_per_step = args.micro_batch * args.grad_accum
    steps_per_epoch = len(train_ds) // samples_per_step
    if steps_per_epoch <= 0:
        raise ValueError(f"训练样本数 {len(train_ds)} 不足 1 step（{samples_per_step} 样本）")
    total_steps = steps_per_epoch * args.epochs
    warmup = args.warmup_steps or max(10, total_steps * 3 // 100)

    # ---- 初始化：从 pretrain.ckpt 只载模型权重，优化器重建 ----
    init = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**init["config"])
    model = LlamaModel(cfg).to(device)
    model.load_state_dict(init["model"])
    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    start_step = 0
    if args.resume:
        last = ckpt_dir / "sft_last.ckpt"
        if not last.exists():
            raise FileNotFoundError(f"--resume 指定但 {last} 不存在")
        ckpt = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"从 step {start_step} 续训")

    print(f"设备={device} | 精度={'bf16' if device.type == 'cuda' else 'fp32'} | "
          f"样本={len(train_ds)} | 样本/step={samples_per_step} | 总步数={total_steps} | warmup={warmup}")

    loader = DataLoader(train_ds, batch_size=args.micro_batch, shuffle=True,
                        collate_fn=lambda b: collate_batch(b, tok.pad_id))
    it = iter(loader)

    def next_batch():
        nonlocal it
        try:
            return next(it)
        except StopIteration:  # 多 epoch 时重新洗牌进入下一轮
            it = iter(loader)
            return next(it)

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
    tokens_seen = 0
    sup_tokens_seen = 0
    model.train()
    step = start_step
    try:
        with tqdm(range(start_step, total_steps), desc="sft", unit="step") as bar:
            for step in bar:
                lr = lr_at(step, total_steps, warmup, args.lr, args.min_lr)
                for g in optimizer.param_groups:
                    g["lr"] = lr

                # ---- 梯度累积：loss 按监督 token 数加权汇总 ----
                optimizer.zero_grad(set_to_none=True)
                step_loss_sum, step_sup, step_tokens = 0.0, 0, 0
                for _ in range(args.grad_accum):
                    x, y = next_batch()
                    x, y = x.to(device), y.to(device)
                    with autocast_ctx(device):
                        loss_sum, n_sup = supervised_loss(model(x), y, cfg.vocab_size)
                        loss = loss_sum / n_sup / args.grad_accum
                    loss.backward()
                    step_loss_sum += loss_sum.item()
                    step_sup += n_sup
                    step_tokens += int((x != tok.pad_id).sum())

                if not math.isfinite(step_loss_sum):
                    nan_streak += 1
                    tqdm.write(f"警告：step {step} loss={step_loss_sum}，已跳过（连续 {nan_streak} 次）")
                    if nan_streak >= MAX_CONSECUTIVE_NAN:
                        raise RuntimeError(
                            f"连续 {MAX_CONSECUTIVE_NAN} 次 NaN/Inf，疑似梯度爆炸；"
                            f"建议降低 --lr 或用 --resume 从最近 ckpt 续训")
                    continue
                nan_streak = 0

                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                if device.type == "mps":
                    # MPS 缓存分配器对变长 batch 会碎片化囤积显存，每步主动释放未用块
                    torch.mps.empty_cache()

                step_loss = step_loss_sum / max(1, step_sup)
                if ema_loss is not None and step_loss > SPIKE_FACTOR * ema_loss:
                    tqdm.write(f"警告：step {step} loss={step_loss:.3f} 超过滑动均值 "
                               f"{SPIKE_FACTOR} 倍，疑似脏数据段")
                ema_loss = step_loss if ema_loss is None else (1 - EMA_DECAY) * ema_loss + EMA_DECAY * step_loss

                tokens_seen += step_tokens
                sup_tokens_seen += step_sup
                now = time.time()
                tokens_s = step_tokens / max(now - t_prev, 1e-9)
                t_prev = now
                bar.set_postfix(loss=f"{ema_loss:.3f}", lr=f"{lr:.2e}", tok_s=f"{tokens_s:.2e}")

                if (step + 1) % args.log_interval == 0 or step == start_step:
                    record({"type": "train", "step": step + 1, "train_loss": round(ema_loss, 4),
                            "lr": lr, "grad_norm": float(grad_norm),
                            "tokens_seen": tokens_seen, "sup_tokens_seen": sup_tokens_seen,
                            "elapsed_s": round(time.time() - t0, 1)})

                if (step + 1) % args.eval_interval == 0 or step + 1 == total_steps:
                    val_loss, val_ppl = evaluate(model, val_ds, device, args.micro_batch,
                                                 args.eval_batches, tok.pad_id)
                    record({"type": "eval", "step": step + 1, "val_loss": round(val_loss, 4),
                            "val_ppl": round(val_ppl, 2)})
                    tqdm.write(f"step {step + 1}: val_loss={val_loss:.4f} ppl={val_ppl:.2f}")

                if (step + 1) % args.save_interval == 0:
                    save_ckpt(ckpt_dir / "sft_last.ckpt", model, optimizer, step + 1, cfg)

    except KeyboardInterrupt:
        save_ckpt(ckpt_dir / "sft_last.ckpt", model, optimizer, step + 1, cfg)
        print(f"\n已中断，进度保存至 {ckpt_dir}/sft_last.ckpt（step {step + 1}），可用 --resume 续训")
        return
    except torch.cuda.OutOfMemoryError:
        print("CUDA 显存不足（OOM）：请调小 --micro-batch 或调大 --grad-accum 保持样本/step 不变")
        raise
    finally:
        log_f.close()
        if tb:
            tb.close()

    save_ckpt(ckpt_dir / "sft.ckpt", model, optimizer, total_steps, cfg)
    save_ckpt(ckpt_dir / "sft_last.ckpt", model, optimizer, total_steps, cfg)
    print(f"训练完成：{ckpt_dir}/sft.ckpt")


if __name__ == "__main__":
    main()
