# 07 DPO（模块6）
> 输入契约（来自 docs/01）：`checkpoints/sft.ckpt` + `data/dpo.jsonl`（chosen/rejected 偏好对）；
> 输出契约：`checkpoints/dpo.ckpt`。
> 训练框架约定继承 docs/05（设备/精度/日志/ckpt 结构/可观测性），样本模板继承 docs/06，本文只写 DPO 特有决策。
> **状态（2026-10-05）**：模块完成。代码实现、Mac 验证（冒烟 6 项 + 小流量端到端 + 过拟合检查）与 **Windows/4070 Ti 正式训练**（29,700 对 × 1 epoch，val_reward_acc 0.5→0.80）全部完成，`checkpoints/dpo.ckpt` 为当前正式产物。

# 目标
直接偏好优化：相对一个**冻结的参考模型**（ref = sft.ckpt 副本，no_grad 不更新），让策略模型对 chosen 回答的相对概率升高、对 rejected 的相对概率降低。教学向，不追求 SOTA。

# 原理（loss）
对每个回答 y，定义其在模型 π 下的"回答对数概率和"（求和区间 = SFT 的监督区间，即 answer token + 结尾 `<eos>`）：

    s(π, y) = Σ log π(y_t | x, y_<t)

DPO loss（β 为控制偏离 ref 幅度的系数，大则保守、小则激进）：

    L = -E log σ( β · [ (s(π, chosen) - s(ref, chosen)) - (s(π, rejected) - s(ref, rejected)) ] )

# 数据
- **源**：`m-a-p/COIG-P`（HuggingFace，中文偏好数据集，1,006,946 对，9 个 parquet 分片共 ~800MB；chosen/rejected 由奖励模型打分构造，见 arXiv:2504.05535）
- **下载**：默认仅下载第 1 个分片（~89MB，~11.2 万对，足够采样），`--num-shards` 可追加；base URL 走 `HF_ENDPOINT` 环境变量（默认 `https://huggingface.co`，国内可设 `https://hf-mirror.com`）
- **转换**：原始 schema 为 `conversations`（多轮对话列表）+ `chosen/rejected`（最后一轮回答）。**仅保留单轮样本**（conversations 恰含 1 条 human 消息），转为统一 schema（只认文件不认人）：
  ```json
  {"instruction": "...", "chosen": "...", "rejected": "..."}
  ```
  空字段丢弃；多轮样本丢弃（我们的模板是单轮，教学项目不引入多轮拼接）
- **采样**：默认随机采 3 万对（`--num-samples` 可调，seed=42）；末尾 1% 作验证集
- **路径**：`data/dpo.jsonl`（全量转换结果）、`data/dpo_train.jsonl` / `data/dpo_val.jsonl`（训练实际读取）
- **新增依赖**：`pyarrow`（读 parquet），需加入 `requirements.txt`——本模块唯一新增依赖
- **实测（2026-10-05）**：第 1 分片 111,883 对中单轮样本 111,823 对（仅 60 对多轮/空字段被丢弃），单分片足够采样 3 万对

# 样本格式与 mask
- 模板同 SFT：`<bos>用户：{instruction}\n助手：{answer}<eos>`；chosen / rejected 各自独立编码，装入同一 batch
- 对数概率和 `s(π, y)` 只统计 answer + `<eos>` 区间（prompt 与 padding 不计），等价于复用 SFT 的 -100 mask 思路
- 截断规则同 SFT：保留完整 prompt、先截 answer；同一对 chosen/rejected 共用同一 token 预算，分别截断
- padding：一对中两条序列长度不同，collate 统一右 pad 到 batch 内最长、向上对齐 64 的倍数（MPS 碎片化对策同 docs/06）

# 训练配置（默认值）
| 项 | 值 | 说明 |
|---|---|---|
| 初始化 | policy ← `sft.ckpt`（重建优化器）；ref ← `sft.ckpt`（冻结，eval + no_grad） | ckpt 只存 policy，ref 可由 sft.ckpt 重建 |
| β | 0.1 | 惯例默认值 |
| lr | 峰值 5e-7，cosine 衰减到 5e-8，warmup 3% | DPO 对 lr 敏感，比 SFT（2e-5）低约 40 倍 |
| epoch | 1 | 3 万对 |
| batch | micro_batch 8 对 × grad_accum 8 = 64 对/step（=128 条序列/step，与 SFT 相当） | 每对含 chosen+rejected 两条序列 |
| 其余 | 优化器/裁剪/精度/设备/seed 同 docs/05 | 复用 src/pretrain 公共函数 |

# 日志与产物
- `logs/dpo_log.jsonl`：train 记 `train_loss`（dpo_loss 的滑动平均，沿用 pretrain/sft 字段名以便复用 plot_curve.py）、`reward_margin`（= β·[(s(π,c)-s(ref,c)) - (s(π,r)-s(ref,r))]，即 loss 里 σ 的输入）、`reward_acc`（chosen reward > rejected reward 的比例，核心指标，应从 ~0.5 起步上升）、lr、grad_norm；eval 记 `val_loss` / `val_reward_acc`（无 val_ppl，plot_curve.py 已兼容该字段缺失）
- `checkpoints/dpo_last.ckpt` / `checkpoints/dpo.ckpt`：结构 `{model, optimizer, step, config}` 同 docs/05，支持 `--resume`
- 曲线：复用 `src/pretrain/plot_curve.py` 出 `logs/dpo_curve.png`

# 文件结构
- `src/dpo/prepare_data.py`：COIG-P parquet → `data/dpo.jsonl` → 采样切分 train/val
- `src/dpo/dataset.py`：jsonl → 偏好对编码（chosen/rejected 双份 (input_ids, labels)）+ collate
- `src/dpo/train.py`：训练主循环（骨架同 src/sft/train.py，差异：双模型前向、DPO loss、batch 装偏好对）
- `src/dpo/sample.py`：对固定几个问题分别用 sft.ckpt / dpo.ckpt 贪心生成，人工抽查对照
- `tests/test_dpo.py`：冒烟测试

# 完成标准
- 冒烟测试：合成偏好对跑数步，校验 ①prompt/pad 不参与对数概率和 ②训练后 chosen 相对 logprob 上升（reward_acc 提升）③ckpt 能加载回模型 ④日志字段齐全
- M2 小子集（数千对）跑通，reward_acc 从 ~0.5 明显上升
- 过拟合检查（强证据）：少量样本 × 多 epoch，reward_acc→1.0、dpo_loss→0
- 抽查：`sample.py` 对比同一问题下 sft 与 dpo 的输出差异

# 验证记录
- 冒烟测试（2026-10-05，M2/CPU）：`python3 -m pytest tests/test_dpo.py` 6 项全过（mask/截断保 eos/collate 前 N chosen 后 N rejected/response_logprob 数学/dpo_loss 数学/训练契约）；全量 `tests/` 30 passed
- 数据（2026-10-05）：COIG-P 第 1 分片经 hf-mirror 下载（75MB），转换得单轮 111,823 对（丢弃 60），采样切分 train=29,700 / val=300
- 小流量起步检查（2026-10-05，MPS，500 对，因与本机 SFT 正式训练争抢 GPU 仅完成 step 1）：loss=0.6931=log 2、reward_margin=0，符合"初始 policy=ref"的理论预期；`reward_acc=0.125` 而非精确 0，系 MPS 前向微小非确定性导致 policy/ref 输出有尾数差异，非 bug
- 小流量端到端（2026-10-05，M2/MPS，500 对 × 31 步，16 对/step，12 分钟）：train_loss 0.693→0.655，train reward_acc ~0.5→0.81，val_reward_acc 0.50→0.62，reward_margin 0→0.61，无 NaN/OOM，ckpt 契约正常——偏好学习方向正确（31 步小 lr 下 val 涨幅温和属预期）
- **过拟合检查（强证据，2026-10-05，M2/MPS）**：16 对 × 20 epoch（80 步，lr 峰值 1e-5），train_loss 0.693→**0.016**，reward_acc 第 5 步起稳定 **1.00**，val_reward_acc=1.0000——偏好对编码/双模型前向/DPO loss/优化器全链路数学正确
- **Windows 正式训练（2026-10-05，Windows/4070 Ti 12GB，CUDA bf16）**：29,700 对 × 1 epoch，464 步 × 64 对/step（**micro4 对 × accum16**，非文档默认 micro8 对），**41 分钟**（5.1~5.7 s/step）。train_loss 0.693→~0.45，val_loss 0.6931→**0.4447**，**val_reward_acc 0.5→0.80**（5 个 eval 点 0.75/0.80/0.80/0.8125/0.80），train reward_acc 末段 ~0.91，reward_margin 0→~0.5，无 NaN/OOM，ckpt 契约正常。基座为 Windows 版 sft.ckpt（val ppl 56，优于 Mac 验证时的基座），val 涨幅（+0.30）大于 M2 小流量（+0.12），符合"基座质量决定上限"的已有结论
  - 与 MPS 不同：CUDA 下 step 1 的 reward_acc 精确为 0、loss 精确 = log 2（前向确定性，无线次差异）
  - 曲线：`logs/dpo_curve.png`，train/val 贴合下降无过拟合间隙；日志 `logs/dpo_log.jsonl`
- **生成抽查（Windows 正式 ckpt，sample.py 贪心对照）**：DPO 与 SFT 输出差异可见（如"中国的首都是哪里"SFT 复读"中国，"、DPO 复读"西方的第三是"），但复读/复述问题/内容肤浅依旧——属 100M 基座 + 纯贪心解码的固有短板（同 docs/06 结论），**缓解归模块 7 解码策略**；DPO 本模块的核心指标 val_reward_acc 达标（0.5→0.80）
- **坑点（Windows 实录）**：
  - ① **默认 micro8 对在本机不可用**：policy+ref 双前向下 16 条序列/microbatch 把 12GB 显存打满（12195/12282 MB），触发共享内存回退——首步 6 分钟未完成即判定（步速自检有效，同 docs/06 坑点①规律）；改 micro4 对 × accum16（对/step 不变）后正常。DPO 显存经验：**micro 对数要比 SFT 的 micro 数减半**
  - ② pyarrow 在训练环境中缺失（requirements.txt 已含但环境未装），pip 直连超时需走清华镜像 `-i https://pypi.tuna.tsinghua.edu.cn/simple`
  - ③ COIG-P 数据 Mac 版未同步到 Windows，本机经 hf-mirror 重新下载分片 0 并重新转换采样（结果与 Mac 完全一致：111,823 对、train=29,700/val=300，seed=42 可复现）

# 待办
- 模块 7（评测与推理：复读缓解的解码策略、MMLU/C-Eval 子集、多轮对话）
