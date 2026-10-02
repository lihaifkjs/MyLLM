"""BPE 分词器封装：加载 tokenizer.json，提供 encode/decode 接口。

契约见 docs/03-tokenizer.md。
"""

from tokenizers import Tokenizer as HFTokenizer

SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]


class Tokenizer:
    def __init__(self, path: str = "tokenizer/tokenizer.json"):
        self._tk = HFTokenizer.from_file(path)

    @property
    def vocab_size(self) -> int:
        return self._tk.get_vocab_size()

    @property
    def pad_id(self) -> int:
        return self._tk.token_to_id("<pad>")

    @property
    def unk_id(self) -> int:
        return self._tk.token_to_id("<unk>")

    @property
    def bos_id(self) -> int:
        return self._tk.token_to_id("<bos>")

    @property
    def eos_id(self) -> int:
        return self._tk.token_to_id("<eos>")

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = self._tk.encode(text).ids
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids: list[int]) -> str:
        return self._tk.decode(ids)
