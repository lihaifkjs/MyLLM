# 06 SFT（模块5）
> 输入契约（来自 docs/01）：`checkpoints/pretrain.ckpt` + `data/sft.jsonl`（Q/A 对）；
> 输出契约：`checkpoints/sft.ckpt`。
> 训练框架约定继承 docs/05（设备/精度/日志/ckpt 结构/可观测性），本文只写 SFT 特有决策。
> **状态（2026-10-05）**：模块完成。正式训练（10 万条，M2/MPS）已跑完并验收，`checkpoints/sft.ckpt` 为正式产物。

# 目标
用中文指令数据把 pretrain 基座微调成"按模板回答问题"的模型；**loss 只算回答部分（问题 mask 为 -100）**。教学向，不追求 SOTA。

# 数据
- **源**：Firefly（`YeungNLP/firefly-train-1.1M`，实测 165 万条中文指令，字段 `kind/input/target`）
- **落盘**：`prepare_data.py` 下载并转为统一 schema（只认文件不认人）：
  ```json
  {"instruction": "...", "output": "..."}
  ```
  其中 `instruction ← input`（Firefly 的 input 已包含任务描述），`output ← target`
- **采样**：默认随机采 10 万条（`--num-samples` 可调，seed=42）；末尾 1% 作验证集
- **路径**：`data/sft.jsonl`（全量转换结果）、`data/sft_train.jsonl` / `data/sft_val.jsonl`（采样切分后，训练实际读取这两个文件）

# 样本格式与 loss mask（核心决策）
- 无角色 token（词表固定 32k，不扩充），用纯文本角色风格模板：
  ```
  <bos>用户：{instruction}\n助手：{output}<eos>
  ```
- **监督区间**：仅 `{output}` 的 token + 结尾 `<eos>`；`<bos>`、"用户：{instruction}\n助手："、padding 全部 label=-100
- **长度**：超过 ctx（1024）截断（先截断 output 侧，保留完整 prompt）；batch 内右 padding、向上对齐 64 的倍数——因因果注意力 + pad 位置 label=-100，右 padding 不污染监督 token；对齐 64 是为限制 batch 形状种类（MPS 显存碎片化对策，见验证记录 bug②）
- eos_id / pad_id 从 Tokenizer 接口属性读取，不硬编码

# 训练配置（默认值）
| 项 | 值 | 说明 |
|---|---|---|
| 初始化 | `checkpoints/pretrain.ckpt`（只载 model 权重，不载 optimizer） | SFT 重新建优化器 |
| lr | 峰值 2e-5，cosine 衰减到 1e-6，warmup 3% | 远小于预训练峰值 6e-4 |
| epoch | 1 | 10 万条 × 1 |
| batch | micro_batch 16 × grad_accum 8 = 128 样本/step | MPS 冒烟可调小 |
| 其余 | 优化器/裁剪/精度/设备/seed 全部同 docs/05 | 优先 import 复用 src/pretrain 的公共函数，不做大重构 |

# 日志与产物
- `logs/sft_log.jsonl`：字段约定同 docs/05（train loss 为**监督 token 平均**，另记监督 token 数；eval 记 val_loss/val_ppl）
- `checkpoints/sft_last.ckpt` / `checkpoints/sft.ckpt`：结构 `{model, optimizer, step, config}` 同 docs/05，支持 `--resume`
- 曲线：复用 `src/pretrain/plot_curve.py` 读 sft jsonl 出 `logs/sft_curve.png`

# 文件结构
- `src/sft/prepare_data.py`：Firefly → `data/sft.jsonl` → 采样切分 train/val
- `src/sft/dataset.py`：jsonl → (input_ids, labels)，含模板拼接、mask、padding collate
- `src/sft/train.py`：训练主循环（骨架同 pretrain，差异：loss 带 mask、按样本计数而非 token 计数）
- `src/sft/sample.py`：加载 sft.ckpt，对固定几个问题贪心生成，人工抽查用（正式评测属模块7）
- `tests/test_sft.py`：冒烟测试

# 完成标准
- 冒烟测试：合成小 jsonl 跑数步，校验 ①labels 的 prompt/pad 位全为 -100 ②ckpt 能加载回模型 ③日志字段齐全
- M2 小子集（数千条）跑通，train loss 正常下降
- 抽查：`sample.py` 对 5 个固定问题生成，输出与基座（pretrain.ckpt）相比明显偏向"回答问题"风格

# 验证记录
- 冒烟测试（2026-10-04，M2/CPU）：`python3 -m pytest tests/test_sft.py` 4 项全过（mask 边界/截断保 eos/padding/训练契约）；全量 `tests/` 23 passed
- 已修 bug①：初版截断会把末尾 `<eos>` 切掉，改为截断前先给 `<eos>` 预留位置
- 已修 bug②：**MPS 显存 OOM**。预训练序列定长 1024 无此问题；SFT 变长 batch 导致 MPS 缓存分配器按形状碎片化囤积（micro16 第一步即超 36GB 上限）。修复：collate padding 向上对齐 64 的倍数（限制形状种类）+ MPS 下每 step `torch.mps.empty_cache()`
- 已修 bug③：**U+2028 切断 JSON 记录**。Firefly 个别回答含 Unicode 行分隔符 U+2028/U+2029，`json.dumps(ensure_ascii=False)` 原样写出、`splitlines()` 又把它当换行，导致记录被切成两段（10 万采样第 7755 行 JSONDecodeError，2000 条小流量未踩中纯属运气）。修复：convert 显式转义 U+2028/2029 保证严格一行一条 + sample_split 改按 `\n` 严格切行 + SFTDataset 跳过空白行；附回归测试 `test_prepare_data_unicode_line_sep`
- 数据（2026-10-04）：Firefly 实际 1,649,399 条（比标称 1.1M 多），转换无报错；采样 2000 条切分 train=1980/val=20。2026-10-05 修复 bug③ 后重新生成 10 万条版（train=99000/val=1000），全量行 JSON 校验通过
- 小流量端到端（M2/MPS，2000 条 × 3 epoch，micro4×accum32=128 样本/step，45 步，14 分钟）：train loss 7.86→7.36，val ppl 1636→1432 单调下降，监督 token 占比 63%，grad_norm ~1，无 NaN/OOM。`sample.py` 抽查：已学会"说完就停"，内容质量差（符合小流量预期）
- **过拟合检查（强证据）**：40 条样本 × 30 epoch（300 步，lr 1e-3），train loss 7.86→**0.040**；`sample.py` 对训练集内问题精确复现标准答案并遇 `<eos>` 正确停止——模板拼接/loss mask/优化器/生成全链路数学正确。val ppl 爆炸与 loss spike 告警均为过拟合预期现象（同 docs/05 记录的误报规律）
- **正式训练（2026-10-05，M2/MPS 完成）**：10 万条 × 1 epoch，773 步 × 128 样本/step（micro4×accum32），3.5 小时。train loss 7.72→6.40，val ppl 1019→**687**（step 100→773 全程单调下降），grad_norm ~1.5~2.4，无 NaN/OOM。ckpt 契约复核通过（step=773，前向 (1,16,32000) 无 NaN）。曲线见 `logs/sft_curve.png`
- 生成抽查（正式 ckpt，贪心）：输出呈"回答"口吻且能停止，但**内容连贯性差**（如"失眠了怎么办？→答案：我喜欢你。"）。根因在基座：预训练仅用 1/10 语料 1 epoch（val ppl 1170），100M 模型的语言/知识能力上限如此；SFT 只负责教会格式，符合教学向预期。若想要更像样的回答，方向是增强预训练（更多语料/epoch），不是加大 SFT
- **当前产物状态**：`checkpoints/sft.ckpt`、`data/sft_train/val.jsonl`（10 万条版）、`logs/sft_log.jsonl`、`logs/sft_curve.png` 均为正式产物；小流量产物备份为 `logs/sft_log_smallflow.jsonl` / `sft_curve_smallflow.png`
- 无新增依赖（下载复用 httpx）；CUDA bf16 路径尚未实机验证（正式训练实际在 M2/MPS 完成，Windows 手册留作复现/重训用）
- 边界说明：`sample.py` 是单轮贪心抽查脚本；多轮交互对话、temperature 采样属模块 7（评测与推理）
