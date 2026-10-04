"""BPE 分词器封装：加载 tokenizer.model，提供 encode/decode 接口。

契约见 docs/03-tokenizer.md。
"""

import sentencepiece as spm

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class Tokenizer:
    def __init__(self, path: str = "tokenizer/tokenizer.model"):
        self._sp = spm.SentencePieceProcessor(model_file=path)

    @property
    def vocab_size(self) -> int:
        return self._sp.get_piece_size()

    @property
    def pad_id(self) -> int:
        return self._sp.pad_id()

    @property
    def unk_id(self) -> int:
        return self._sp.unk_id()

    @property
    def bos_id(self) -> int:
        return self._sp.bos_id()

    @property
    def eos_id(self) -> int:
        return self._sp.eos_id()

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = list(self._sp.encode(text))
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self._sp.decode(ids)
