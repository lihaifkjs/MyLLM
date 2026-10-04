"""模块4 预训练：离线 token 编码 + memmap 数据管线 + 训练主循环。契约见 docs/05-预训练.md。"""
from .dataset import TokenData
from .encode_corpus import encode_corpus

__all__ = ["TokenData", "encode_corpus"]
