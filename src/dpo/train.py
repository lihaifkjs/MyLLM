"""DPO 主循环：policy 从 sft.ckpt 初始化，ref = 冻结的 sft.ckpt 副本。

产物契约（docs/07-DPO.md）：
- checkpoints/dpo.ckpt（最终）与 dpo_last.ckpt（定期，用于续训）；只存 policy，ref 可由 sft.ckpt 重建
- logs/dpo_log.jsonl（train 记 train_loss/reward_margin/reward_acc；eval 记 val_loss/val_reward_acc）
训练骨架复用 src/sft/train.py，公共函数（lr 调度/设备/精度/优化器/ckpt）来自 src/pretrain/train.py。
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
from src.dpo.dataset import DPODataset, collate_pairs
from src.tokenizer.tokenizer import Tokenizer


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="DPO（模块6）")
    ap.add_argument("--train", default="data/dpo_train.jsonl")
    ap.add_argument("--val", default="data/dpo_val.jsonl")
    ap.add_argument("--tokenizer", default="tokenizer/tokenizer.model")
    ap.add_argument("--init-ckpt", default="checkpoints/sft.ckpt",
                    help="policy 初始化与 ref 冻结副本的来源（只载模型权重，优化器重建）")
    ap.add_argument("--ckpt-dir", default="checkpoints")
    ap.add_argument("--log", default="logs/dpo_log.jsonl")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--micro-batch", type=int, default=8, help="每 micro batch 的偏好对数（序列数 ×2）")
    ap.add_argument("--grad-accum", type=int, default=8, help="默认 8×8=64 对/step（=128 条序列/step）")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--beta", type=float, default=0.1, help="控制 policy 偏离 ref 的幅度，大则保守")
    ap.add_argument("--lr", type=float, default=5e-7)
    ap.add_argument("--min-lr", type=float, default=5e-8)
    ap.add_argument("--warmup-steps", type=int, default=None, help="默认 3%% total_steps")
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--eval-interval", type=int, default=100)
    ap.add_argument("--eval-batches", type=int, default=20)
    ap.add_argument("--save-interval", type=int, default=200)
    ap.add_argument("--log-interval", type=int, default=10)
    ap.add_argument("--device", default=None, help="默认自动选择 cuda > mps > cpu")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--resume", action="store_true", help="从 dpo_last.ckpt 续训（ref 始终从 init-ckpt 重建）")
    return ap.parse_args(argv)


def response_logprob(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """每条序列的 s(π, y) = 监督区间（labels != -100，即 answer + <eos>）的 log π 求和。

    labels 右移一位做 next-token 对齐；先在 fp32 下算 log_softmax，避免 bf16 下数值误差。
    """
    tgt = labels[:, 1:]
    mask = tgt != -100
    logp = F.log_softmax(logits[:, :-1].float(), dim=-1)
    token_logp = torch.gather(logp, 2, tgt.clamp(min=0).unsqueeze(-1)).squeeze(-1)
    return (token_logp * mask).sum(dim=1)


def dpo_loss(pi_chosen: torch.Tensor, pi_rejected: torch.Tensor,
             ref_chosen: torch.Tensor, ref_rejected: torch.Tensor, beta: float):
    """L = -log σ(reward)，reward = β·[(s(π,c)-s(ref,c)) - (s(π,r)-s(ref,r))]。

    返回 (loss, reward_margin, reward_acc)：margin/acc 是核心可观测指标，
    acc 表示 policy 相对 ref 更偏好 chosen 的样本比例，训练前应约 0.5。
    """
    rewards = beta * ((pi_chosen - ref_chosen) - (pi_rejected - ref_rejected))
    loss = -F.logsigmoid(rewards).mean()
    acc = (rewards > 0).float().mean()
    return loss, rewards.mean(), acc


def pair_metrics(model, ref_model, x, y, beta: float):
    """一个 batch（前 N chosen、后 N rejected）上过 policy 与 ref，返回 (loss, margin, acc)。"""
    n = x.shape[0] // 2
    pi = response_logprob(model(x), y)
    with torch.no_grad():
        ref = response_logprob(ref_model(x), y)
    return dpo_loss(pi[:n], pi[n:], ref[:n], ref[n:], beta)


@torch.no_grad()
def evaluate(model, ref_model, val_ds: DPODataset, device, batch_size: int,
             max_batches: int, pad_id: int, beta: float):
    model.eval()
    loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                        collate_fn=lambda b: collate_pairs(b, pad_id))
    loss_sum, acc_sum, n_pairs = 0.0, 0.0, 0
    for i, (x, y) in enumerate(loader):
        if i >= max_batches:
            break
        x, y = x.to(device), y.to(device)
        with autocast_ctx(device):
            loss, _, acc = pair_metrics(model, ref_model, x, y, beta)
        n = x.shape[0] // 2
        loss_sum += loss.item() * n
        acc_sum += acc.item() * n
        n_pairs += n
    model.train()
    return loss_sum / max(1, n_pairs), acc_sum / max(1, n_pairs)


def main(argv=None) -> None:
    args = parse_args(argv)

    # ---- 启动自检 ----
    for p in (args.train, args.val):
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} 不存在，请先运行：python -m src.dpo.prepare_data")
    if not args.resume and not Path(args.init_ckpt).exists():
        raise FileNotFoundError(f"初始化权重 {args.init_ckpt} 不存在（模块5 产物）")
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    device = pick_device(args.device)
    torch.manual_seed(args.seed)
    tok = Tokenizer(args.tokenizer)
    train_ds = DPODataset(args.train, tok, args.max_len)
    val_ds = DPODataset(args.val, tok, args.max_len)

    pairs_per_step = args.micro_batch * args.grad_accum
    steps_per_epoch = len(train_ds) // pairs_per_step
    if steps_per_epoch <= 0:
        raise ValueError(f"训练对数 {len(train_ds)} 不足 1 step（{pairs_per_step} 对）")
    total_steps = steps_per_epoch * args.epochs
    warmup = args.warmup_steps or max(10, total_steps * 3 // 100)

    # ---- 初始化：policy 与 ref 都来自 sft.ckpt；ref 冻结不训练 ----
    init = torch.load(args.init_ckpt, map_location="cpu", weights_only=False)
    cfg = ModelConfig(**init["config"])
    model = LlamaModel(cfg).to(device)
    model.load_state_dict(init["model"])
    ref_model = LlamaModel(cfg).to(device)
    ref_model.load_state_dict(init["model"])
    ref_model.eval()
    ref_model.requires_grad_(False)
    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    start_step = 0
    if args.resume:
        last = ckpt_dir / "dpo_last.ckpt"
        if not last.exists():
            raise FileNotFoundError(f"--resume 指定但 {last} 不存在")
        ckpt = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_step = ckpt["step"]
        print(f"从 step {start_step} 续训")

    print(f"设备={device} | 精度={'bf16' if device.type == 'cuda' else 'fp32'} | "
          f"偏好对={len(train_ds)} | 对/step={pairs_per_step} | 总步数={total_steps} | "
          f"warmup={warmup} | beta={args.beta}")

    loader = DataLoader(train_ds, batch_size=args.micro_batch, shuffle=True,
                        collate_fn=lambda b: collate_pairs(b, tok.pad_id))
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
    model.train()
    step = start_step
    try:
        with tqdm(range(start_step, total_steps), desc="dpo", unit="step") as bar:
            for step in bar:
                lr = lr_at(step, total_steps, warmup, args.lr, args.min_lr)
                for g in optimizer.param_groups:
                    g["lr"] = lr

                # ---- 梯度累积：指标按偏好对数加权汇总 ----
                optimizer.zero_grad(set_to_none=True)
                step_loss_sum, step_margin_sum, step_acc_sum = 0.0, 0.0, 0.0
                step_pairs, step_tokens = 0, 0
                for _ in range(args.grad_accum):
                    x, y = next_batch()
                    x, y = x.to(device), y.to(device)
                    with autocast_ctx(device):
                        loss, margin, acc = pair_metrics(model, ref_model, x, y, args.beta)
                        loss = loss / args.grad_accum
                    loss.backward()
                    n = x.shape[0] // 2
                    step_loss_sum += loss.item() * args.grad_accum * n
                    step_margin_sum += margin.item() * n
                    step_acc_sum += acc.item() * n
                    step_pairs += n
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

                step_loss = step_loss_sum / max(1, step_pairs)
                if ema_loss is not None and step_loss > SPIKE_FACTOR * ema_loss:
                    tqdm.write(f"警告：step {step} loss={step_loss:.3f} 超过滑动均值 "
                               f"{SPIKE_FACTOR} 倍，疑似脏数据段")
                ema_loss = step_loss if ema_loss is None else (1 - EMA_DECAY) * ema_loss + EMA_DECAY * step_loss

                tokens_seen += step_tokens
                now = time.time()
                tokens_s = step_tokens / max(now - t_prev, 1e-9)
                t_prev = now
                step_acc = step_acc_sum / max(1, step_pairs)
                bar.set_postfix(loss=f"{ema_loss:.3f}", acc=f"{step_acc:.2f}", lr=f"{lr:.2e}")

                if (step + 1) % args.log_interval == 0 or step == start_step:
                    record({"type": "train", "step": step + 1, "train_loss": round(ema_loss, 4),
                            "reward_margin": round(step_margin_sum / max(1, step_pairs), 4),
                            "reward_acc": round(step_acc, 4),
                            "lr": lr, "grad_norm": float(grad_norm),
                            "tokens_seen": tokens_seen,
                            "elapsed_s": round(time.time() - t0, 1)})

                if (step + 1) % args.eval_interval == 0 or step + 1 == total_steps:
                    val_loss, val_acc = evaluate(model, ref_model, val_ds, device,
                                                 args.micro_batch, args.eval_batches,
                                                 tok.pad_id, args.beta)
                    record({"type": "eval", "step": step + 1, "val_loss": round(val_loss, 4),
                            "val_reward_acc": round(val_acc, 4)})
                    tqdm.write(f"step {step + 1}: val_loss={val_loss:.4f} val_reward_acc={val_acc:.4f}")

                if (step + 1) % args.save_interval == 0:
                    save_ckpt(ckpt_dir / "dpo_last.ckpt", model, optimizer, step + 1, cfg)

    except KeyboardInterrupt:
        save_ckpt(ckpt_dir / "dpo_last.ckpt", model, optimizer, step + 1, cfg)
        print(f"\n已中断，进度保存至 {ckpt_dir}/dpo_last.ckpt（step {step + 1}），可用 --resume 续训")
        return
    except torch.cuda.OutOfMemoryError:
        print("CUDA 显存不足（OOM）：请调小 --micro-batch 或调大 --grad-accum 保持对/step 不变")
        raise
    finally:
        log_f.close()
        if tb:
            tb.close()

    save_ckpt(ckpt_dir / "dpo.ckpt", model, optimizer, total_steps, cfg)
    save_ckpt(ckpt_dir / "dpo_last.ckpt", model, optimizer, total_steps, cfg)
    print(f"训练完成：{ckpt_dir}/dpo.ckpt")


if __name__ == "__main__":
    main()
