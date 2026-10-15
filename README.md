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
| 4 | 预训练 | `checkpoints/pretrain.ckpt` + loss 曲线 | ✅（M2，val ppl 1170） | v0.4-pretrain |
| 5 | SFT | `checkpoints/sft.ckpt`（loss 只算回答+eos） | ✅（M2，val ppl 687） | — |
| 6 | DPO | `checkpoints/dpo.ckpt`（ref=冻结 sft，β=0.1） | ✅（Mac 验证完，待 4070 正式训练） | v0.6-dpo-implement |
| 7 | 评测与推理 | 困惑度报告、case 对比、命令行对话 | ⬜ | — |

## 快速开始

```bash
pip install -r requirements.txt
python3 -m pytest tests/        # 全部冒烟测试（30 项）
```

正式训练（Windows + 4070）的完整自包含手册见 **[使用说明.md](使用说明.md)**（环境→冒烟→数据→训练→监控→抽查→故障处理）。

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

## 目录结构

```
src/
  corpus/     # 模块1：语料下载/清洗/去重
  tokenizer/  # 模块2：BPE 训练与封装
  model/      # 模块3：Llama 结构（config.py + model.py）
  pretrain/   # 模块4：encode_corpus / dataset / train / plot_curve
  sft/        # 模块5：prepare_data / dataset / train / sample
  dpo/        # 模块6：prepare_data / dataset / train / sample
tests/        # 各模块冒烟测试（30 项）
docs/         # 专题设计文档（文档驱动开发）
checkpoints/  # 权重产物（.ckpt，结构 {model, optimizer, step, config}）
data/         # 语料与数据集产物（.txt / .jsonl）
logs/         # 训练日志（.jsonl）与曲线（.png）
```

## 关键设计决策

- **文档驱动**：任何功能先对齐 `docs/` 专题文档再实现；各模块只通过文件契约对接
- **跨平台训练**：CUDA 走 bf16 autocast，MPS/CPU 走 fp32；支持 `--resume` 断点续训与 Ctrl-C 安全中断
- **验证方法**：每模块配冒烟测试；预训练/SFT/DPO 均做过"过拟合检查"（小样本多 epoch 压到 loss≈0 / reward_acc=1）验证全链路数学正确
