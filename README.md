# MyLLM：从零自制中文大模型（教学向）

亲手走通 **语料 → Tokenizer → 预训练 → SFT → DPO → 评测 → 推理** 全流程，验证各环节原理，不追求 SOTA。

- 模型：Llama 结构（RMSNorm + SwiGLU + RoPE + MHA），**~109.5M 参数**（d768 / L12 / ctx 1024）
- 词表：sentencepiece BPE，32k（byte_fallback，含 4 个特殊 token）
- 语料：中文，Skypile-150B 采样，规则清洗 + 精确/MinHash 去重
- 对齐：SFT（Firefly 中文指令）→ DPO（COIG-P 中文偏好对），不做 RLHF
- 环境：Mac M2（MPS，开发与小实验）/ Windows + RTX 4070（CUDA，正式训练），代码跨平台

## 模块进度

| # | 模块 | 输出契约 | 状态 | Tag |
|---|---|---|---|---|
| 1 | 语料处理 | `data/skypile_corpus.txt`（一行一段） | ✅ | v0.1-corpus |
| 2 | Tokenizer | `tokenizer/tokenizer.model` + encode/decode 接口 | ✅ | v0.2-tokenizer |
| 3 | 模型定义 | `src/model/`：`forward(input_ids) → logits` | ✅ | v0.3-model |
| 4 | 预训练 | `checkpoints/pretrain.ckpt` + loss 曲线 | ✅（M2 val ppl 1170 / 4070 语料尾部 ppl 59.78） | v0.4-pretrain |
| 5 | SFT | `checkpoints/sft.ckpt`（loss 只算回答+eos） | ✅（M2 val ppl 687 / 4070 val ppl 56.18） | v0.5-sft-windows |
| 6 | DPO | `checkpoints/dpo.ckpt`（ref=冻结 sft，β=0.1） | ✅（4070 正式训练，val_reward_acc 0.5→0.80） | v0.6-dpo-implement |
| 7 | 评测与推理 | 困惑度报告、case 对比、命令行对话 | ✅（C-Eval 26.37%、复读 0/12） | v0.7-eval-and-chat |

## 训练效果（Windows + RTX 4070 Ti 12GB，CUDA bf16）

| 阶段 | 规模 | 耗时 | 关键指标 |
|---|---|---|---|
| 预训练 | 2,107 步 × 262,144 token/步 ≈ **5.5 亿 token**（1 epoch） | — | 语料尾部 ppl **59.78** |
| SFT | 10 万条，773 步 × 128 条/步（micro8 × accum16，≈2.1 s/step） | 22~27 分钟 | val ppl **56.18**（loss 只算回答+eos 生效） |
| DPO | 29,700 对，464 步 × 64 对/步（micro4 对 × accum16，≈5.1~5.7 s/step） | 41 分钟 | **val_reward_acc 0.5 → 0.80**，val_loss 0.693 → 0.445 |

客观评测（`checkpoints/sft.ckpt` vs `checkpoints/dpo.ckpt`，均为本机实测）：

| 评测 | sft | dpo | 参照 |
|---|---|---|---|
| C-Eval val 800 题 | 25.50% | 26.37% | 随机线 25%，100M 基本贴随机线属预期 |
| ppl（sft_val 回答语料） | 79.62 | 80.06 | dpo 略高符合"牺牲似然换偏好" |
| 12 题机械复读率 | 0/12 | 0/12 | 采样 + 复读缓解解码，复读已被控制 |

> **显存坑点**：12GB 下 SFT 需 micro8 × accum16、DPO 需 micro4 对 × accum16（双模型前向，micro 对数要减半），默认值会打满显存触发共享内存回退。
> **能力边界**：50 个真实 demo 分档为 ~8 条基本可用 / ~30 条语句通顺但答错跑题 / ~12 条无效输出，事实与数学类基本不可用——属 100M 规模的固有短板而非训练故障，详见 **[实际效果.md](实际效果.md)**。


## 快速开始

```bash
pip install -r requirements.txt
python3 -m pytest tests/        # 全部冒烟测试（39 项）
```

正式训练（Windows + 4070）的完整自包含手册见 **[使用说明.md](使用说明.md)**（环境→冒烟→数据→训练→监控→抽查→故障处理）；训练完的真实生成效果（50 个 demo 及质量分档）见 **[实际效果.md](实际效果.md)**。

各阶段入口（均支持 `--help`）：

```bash
# SFT（模块5）：Firefly → data/sft.jsonl → 训练 → 抽查
python -m src.sft.prepare_data
python -m src.sft.train
python -m src.sft.sample

# DPO（模块6）：COIG-P → data/dpo.jsonl → 训练 → sft/dpo 对照抽查
python -m src.dpo.prepare_data      # 国内可设 HF_ENDPOINT=https://hf-mirror.com
python -m src.dpo.train
python -m src.dpo.sample

# 评测与对话（模块7）
python -m src.eval.ppl --ckpt checkpoints/dpo.ckpt      # 统一困惑度入口（按 ckpt 名自动选语料）
python -m src.eval.ceval --ckpt checkpoints/dpo.ckpt    # C-Eval val 子集 800 题 logprob 打分
python -m src.eval.case_compare                         # sft vs dpo 固定 12 题对照表
python -m src.chat.chat                                 # 带复读缓解解码的多轮对话

# 预训练（模块4）
python -m src.pretrain.encode_corpus
python -m src.pretrain.train

# loss 曲线（各阶段通用）
python -m src.pretrain.plot_curve --log logs/dpo_log.jsonl --out logs/dpo_curve.png
```

语料处理（模块1）与 Tokenizer 训练（模块2）的命令见对应专题文档。

## 文档地图（docs/）

- `01-整体方案.md`：总体设计与模块契约（只认文件不认人）
- `02-语料处理.md` / `03-tokenizer.md` / `04-模型定义.md`
- `05-预训练.md`：memmap 数据管线、AdamW+cosine、断点续训、可观测性
- `06-SFT.md`：模板 `<bos>用户：...\n助手：...<eos>`，loss 只算回答+eos
- `07-DPO.md`：DPO 原理（ref 模型 + β）、COIG-P 数据、reward_acc 指标
- `08-评测与推理.md`：ppl / C-Eval / 案例对照 / 多轮对话与复读缓解解码
- 根目录 `实际效果.md`：端到端真实效果（三阶段指标 + 曲线 + 50 个生成 demo 及分档）

## 目录结构

```
src/
  corpus/     # 模块1：语料下载/清洗/去重
  tokenizer/  # 模块2：BPE 训练与封装
  model/      # 模块3：Llama 结构（config.py + model.py）
  pretrain/   # 模块4：encode_corpus / dataset / train / plot_curve
  sft/        # 模块5：prepare_data / dataset / train / sample
  dpo/        # 模块6：prepare_data / dataset / train / sample
  eval/       # 模块7：ppl / ceval / case_compare / questions
  chat/       # 模块7：decoding（复读缓解）/ chat（多轮 REPL）
tests/        # 各模块冒烟测试（39 项）
docs/         # 专题设计文档（文档驱动开发）
checkpoints/  # 权重产物（.ckpt，结构 {model, optimizer, step, config}）
data/         # 语料与数据集产物（.txt / .jsonl）
logs/         # 训练日志（.jsonl）与曲线（.png）
```

## 关键设计决策

- **文档驱动**：任何功能先对齐 `docs/` 专题文档再实现；各模块只通过文件契约对接
- **跨平台训练**：CUDA 走 bf16 autocast，MPS/CPU 走 fp32；支持 `--resume` 断点续训与 Ctrl-C 安全中断
- **验证方法**：每模块配冒烟测试；预训练/SFT/DPO 均做过"过拟合检查"（小样本多 epoch 压到 loss≈0 / reward_acc=1）验证全链路数学正确
