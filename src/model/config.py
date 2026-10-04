"""模型超参配置，取值见 docs/04-模型定义.md。"""
from dataclasses import dataclass


@dataclass
class ModelConfig:
    vocab_size: int = 32000
    d_model: int = 768
    n_layers: int = 12
    n_heads: int = 12
    ffn_hidden: int = 2048
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    rms_norm_eps: float = 1e-5
    tie_weights: bool = True  # lm_head 与 embedding 共享权重

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads
