# 03 Tokenizer（模块2）
> 基于 sentencepiece 训练 BPE 分词器，供预训练/SFT/DPO 统一使用。

# 契约（对接 docs/01 模块划分）
- **输入**：`data/skypile_corpus.txt`（一行一段，UTF-8）
- **输出**：
  - `tokenizer/tokenizer.model`：BPE 模型文件（词表 + 合并规则 + 特殊 token）
  - `src/tokenizer/`：`train()` / `encode()` / `decode()` 接口

# 关键决策
| 项 | 决策 | 理由 |
|---|---|---|
| 框架 | sentencepiece（BPE） | HF tokenizers 在 GB 级中文语料上合并阶段过慢（实测 ~10min/merge）；sentencepiece 专为大语料设计，中文 32k 训练是业界标配路径 |
| byte_fallback | 开启 | 任意字符可由字节回退表示，不会 OOV，`<unk>` 仅作占位 |
| 词表大小 | 32000（含特殊 token） | 整体方案约定 ~32k |
| 特殊 token | `<pad>`=0, `<unk>`=1, `<bos>`=2, `<eos>`=3 | id 固定，下游硬编码安全 |
| 归一化 | identity（不归一化） | 默认 nmt_nfkc 会把全角标点折叠成半角（`，`→`,`），破坏 roundtrip 保真且影响生成文本风格 |
| 训练数据 | 全量 2.5GB（约 1195 万行） | 语料已清洗，无需再采样 |

# 接口签名
```python
# src/tokenizer/tokenizer.py
class Tokenizer:
    def __init__(self, path: str = "tokenizer/tokenizer.model"): ...
    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
    @property
    def vocab_size(self) -> int: ...
    # 特殊 token id 属性：pad_id / unk_id / bos_id / eos_id

# src/tokenizer/train.py
def train(corpus_path: str = "data/skypile_corpus.txt",
          out_dir: str = "tokenizer",
          vocab_size: int = 32000) -> None: ...
```

# 使用约定
- 预训练拼接语料时由数据侧决定加不加 `<bos>/<eos>`，tokenizer 默认不加（`add_bos/add_eos` 显式控制）
- `<pad>` 仅在 batch 对齐时使用，不参与训练文本

# 冒烟测试
- 小样例 roundtrip：`decode(encode(text)) == text`（含中文、英文、数字、标点、emoji）
- 词表检查与 4 个特殊 token id 校验
- 文件契约：`tokenizer/tokenizer.model` 可被独立加载

# 备注
- 训练机：Mac M2 即可，预计几十分钟
- 依赖：`sentencepiece`（写入 requirements.txt）

# 训练记录（2026-10-03，模块2 已完成）
> 供后续 agent 快速了解过程与坑点。

## 结果
- 产物：`tokenizer/tokenizer.model`（495K）+ `tokenizer.vocab`，已入库
- 全量 2.5GB 语料（1195 万行，9.5 亿字符），耗时 **63 分钟**（Mac M2）
- 验证：词表精确 32000；特殊 token id 0~3；中/英/生僻字/emoji roundtrip 无损；压缩率 **1.72 字符/token**（全语料约合 5.5 亿 token，符合整体方案"几亿 token"目标）

## 过程与坑点（按时间序）
1. **初版用 HF `tokenizers`（Byte-level BPE）**：小样本正常，但全量训练合并阶段 ~10min/merge，32000 次合并约需 240 天，不可用。
2. **尝试修复①：标点切分预分词**（`Split(标点正则, isolated) + ByteLevel`）：思路是避免整段长文成为不重复的巨型"词"，实测无效，合并依然 ~10min/merge。结论：瓶颈在 HF BPE 训练器本身的算法复杂度（每轮合并成本随语料线性增长），与预分词无关。
3. **最终方案：换 sentencepiece**：专为大语料设计，63 分钟完成。接口签名（`train/encode/decode/vocab_size/特殊token id`）保持不变，下游无感。
4. **坑点②：归一化**。sentencepiece 默认 `nmt_nfkc` 会把全角标点折叠成半角（`，`→`,`），破坏 roundtrip 且会让模型生成半角标点的中文，故改用 `identity`（不归一化）。
5. **工程经验**：后台长跑任务要用 pty 包装（`pty.spawn`）才能看到 Rust/C++ 端进度条；Mac 合盖/睡眠会冻结训练，锁屏不会；小样本测试时注意 sentencepiece 要求 `vocab_size ≥ 语料必需字符数 + 特殊token数`。

## 冒烟测试
`python3 -m pytest tests/test_tokenizer.py`（4 项：词表、特殊token id、roundtrip、bos/eos 标记）
