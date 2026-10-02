# 03 Tokenizer（模块2）
> 基于 HuggingFace `tokenizers` 训练 Byte-level BPE 分词器，供预训练/SFT/DPO 统一使用。

# 契约（对接 docs/01 模块划分）
- **输入**：`data/skypile_corpus.txt`（一行一段，UTF-8）
- **输出**：
  - `tokenizer/tokenizer.json`：BPE 模型文件（词表 + 合并规则 + 特殊 token）
  - `src/tokenizer/`：`train()` / `encode()` / `decode()` 接口

# 关键决策
| 项 | 决策 | 理由 |
|---|---|---|
| 算法 | Byte-level BPE | 初始符号为字节，任意字符可表示，不会 OOV；Qwen 等中文模型同方案 |
| 词表大小 | 32000（含特殊 token） | 整体方案约定 ~32k |
| 特殊 token | `<pad>`=0, `<unk>`=1, `<bos>`=2, `<eos>`=3 | id 固定，下游硬编码安全 |
| 归一化 | NFC | 统一 Unicode 组合形式，避免同一字符多种编码 |
| 训练数据 | 全量 2.5GB（约 1195 万行） | 语料已清洗，无需再采样 |

# 接口签名
```python
# src/tokenizer/tokenizer.py
class Tokenizer:
    def __init__(self, path: str = "tokenizer/tokenizer.json"): ...
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
- 词表检查：`vocab_size == 32000`，4 个特殊 token id 正确
- 文件契约：`tokenizer/tokenizer.json` 可被独立加载

# 备注
- 训练机：Mac M2 即可（Rust 实现，内存约需数 GB）
- 依赖：`tokenizers`（写入 requirements.txt）
