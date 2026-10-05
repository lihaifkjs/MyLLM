# 06 SFT（模块5）
> 输入契约（来自 docs/01）：`checkpoints/pretrain.ckpt` + `data/sft.jsonl`（Q/A 对）；
> 输出契约：`checkpoints/sft.ckpt`。
> 训练框架约定继承 docs/05（设备/精度/日志/ckpt 结构/可观测性），本文只写 SFT 特有决策。
> **状态（2026-10-05）**：模块完成。正式训练（10 万条 × 1 epoch）已在 M2/MPS 与 Windows/4070 Ti（CUDA bf16 首次实机验证）双平台跑完；`checkpoints/sft.ckpt` 当前为 **Windows 版**（val ppl 56，基于 train_loss 3.83 新预训练基座），M2 版日志已备份 `logs/sft_log_m2_official.jsonl`。

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
- **Windows 正式训练（2026-10-05，Windows/4070 Ti 12GB，CUDA bf16 首次实机验证）**：10 万条 × 1 epoch，773 步 × 128 样本/step（**micro8 × accum16**，非文档默认 micro16），**22 分钟**（2.1 s/step）。train_loss 5.25→3.77，val_ppl **88→56** 全程单调下降（8 个 eval 点），无 NaN/OOM，监督 token 占比 64.3%，ckpt 契约正常。基座对比：M2 基座（val ppl 1170）SFT 后 val ppl 687；本次新基座（train_loss 3.83）SFT 后 val ppl 56——**基座质量决定 SFT 上限**，验证 docs/01 流程假设
- 生成抽查（Windows 正式 ckpt，贪心）：4/5 正常停止且 topical，1/5 无限重复（"中国的首都是哪里"→"中国，中国，…"），内容整体肤浅——100M 模型 + 纯贪心的固有天花板；复读问题缓解归后续模块（采样/解码策略）
- **坑点与教训（Windows 实录）**：
  - ① **micro16 默认在本机不可用**：12GB 显存打满（11.2/12.2GB）触发共享内存回退，步速从 2 s/step 劣化到 60~70 s/step（tok_s 仅 ~800），表象是"训练极慢"而非 OOM 报错；**步速比基线慢一个数量级 = 显存压力早期信号**，应先查显存（pynvml）再决定是否继续硬等
  - ② **nvidia-smi 在部分终端报 "Failed to initialize NVML"**：Windows 会话隔离所致，与 CUDA 计算无关（同终端 torch CUDA 正常）；改用 pynvml 查询（命令已记入 使用说明.md §6）
  - ③ **训练中途发生 nvlddmkm Event 153**（GPU 驱动级崩溃，进程消失且无 traceback）；当时恰在执行 taskkill 强杀 CUDA 进程，可能为诱因；重跑后未复发。本机 10-04 预训练阶段亦有崩溃排查记录（logs/_crash.ps1、_whea.ps1、_set_plimit.py）
  - ④ **日志会跨轮次追加**：Windows 训练直接追加在 M2 日志后，绘图前须按"最后一个 step==1"拆分（logs/_live_plot.py 同款逻辑）；M2 正式记录已备份 `logs/sft_log_m2_official.jsonl`
  - ⑤ **checkpoint 每 200 步才保存**：早于 step 200 崩溃 = 无 ckpt 只能重跑（本次两次崩溃均损失全部进度）
  - ⑥ **本机 conda 未入 PATH**：终端命令须用全路径 `C:\Users\fkjslee\miniconda3\envs\myllm\python.exe`（系统默认 python 无 torch）
  - ⑦ **HF 直连不可用**：huggingface.co ConnectTimeout，数据下载需走 hf-mirror.com（同 COIG-P 下载通道）
- **当前产物状态**：`checkpoints/sft.ckpt`（Windows 版，11:15 生成）+ `data/sft_train/val.jsonl`（10 万条版）+ `logs/sft_log.jsonl`（含 M2+Windows 两段，拆分逻辑见坑点④）为当前正式产物；M2 版训练日志备份 `logs/sft_log_m2_official.jsonl`；小流量产物备份 `logs/sft_log_smallflow.jsonl` / `sft_curve_smallflow.png`
- 无新增依赖（下载复用 httpx）；~~CUDA bf16 路径尚未实机验证~~ → **已于 2026-10-05 在 Windows/4070 Ti 完成实机验证**（实测结论见验证记录"坑点与教训"）
- 边界说明：`sample.py` 是单轮贪心抽查脚本；多轮交互对话、temperature 采样属后续模块（评测与推理，文档待建）——"复读/内容肤浅"问题的缓解依赖该模块的解码策略

# 模型当前状态与使用效果（2026-10-05 实测）
- **当前正式产物**：`checkpoints/sft.ckpt`（Windows 版，基于 train_loss 3.83 新预训练基座，val ppl 56）
- **会做什么**：识别 `<bos>用户：…\n助手：…<eos>` 单轮格式并作答；多数情况能在合理位置停止；回答 topical（能扣题）
- **不会做什么**：内容肤浅无细节、不能写可用代码、开放式问题易复读或复读式罗列、偶尔直接复述问题
- **实测记录**（sample.py 贪心生成，问答节选）：
  | 问题 | 输出（节选） | 判定 |
  |---|---|---|
  | 中国的首都是哪里？ | "中国，中国，中国，…"（无限重复至上限） | ❌ 复读 |
  | 如何用Python实现快速排序？ | "1. 排序：使用Python或Python编写代码。2. …"（同句循环） | ❌ 复读 |
  | 请解释什么是光合作用 | "光合作用是光合作用中的一种，它包括光、电、热、化学、生物等。" | ⚠️ 扣题但空洞 |
  | 写一句关于春天的诗 | "春日暖阳照春。" | ⚠️ 尚可 |
  | 失眠了怎么办？/ 感冒了应该注意什么？ | 直接复述问题 | ❌ 未作答 |
- **与 M2 版对比**：M2 版（val ppl 687）"会停但内容连贯性差"；Windows 版（val ppl 56）停止行为更好、更扣题，复读仍偶发——改善来自基座而非 SFT 本身，符合"基座质量决定 SFT 上限"的结论
- **使用建议**：适合演示格式学习效果与短问答 demo；不要期待代码生成/长推理解答；复读问题待后续模块用解码策略（repetition penalty / temperature 采样）缓解后再评估

# 待办
- [ ] `使用说明.md` 增补 Windows SFT 修正：micro8 × accum16（micro16 在本机不可用）、python 全路径说明、实测基线 2.1 s/step ≈ 22 分钟、"步速异常先查显存"自检步骤
- [ ] "中国的首都是哪里"复读样本：留待后续模块用解码策略（repetition penalty / 采样）验证缓解效果
- [ ] git 提交：建议 `docs(sft): 增补 Windows/4070 正式训练验证记录与坑点`（涉及 `docs/06-SFT.md`、`使用说明.md`）
