"""Tokenizer 冒烟测试：小样本训练 + roundtrip + 特殊 token 校验。"""

import pytest

from src.tokenizer import SPECIAL_TOKENS, Tokenizer, train

SAMPLE_LINES = [
    "本报太原2月14日讯，山西省公安厅对外发布消息称，扫黑除恶专项行动启动。",
    "截至目前，全省公安机关共打掉涉黑涉恶犯罪组织和团伙54个。",
    "Hello, world! 中英文混排测试 123。",
    "特殊字符测试：emoji 😀、符号 ©®、繁体 臺灣。",
] * 50


@pytest.fixture(scope="module")
def toy_tokenizer(tmp_path_factory):
    corpus = tmp_path_factory.mktemp("tok") / "corpus.txt"
    corpus.write_text("\n".join(SAMPLE_LINES), encoding="utf-8")
    # sentencepiece 要求 vocab_size >= 语料必需字符数 + 特殊 token 数（本样本约 348）
    out_path = train(str(corpus), str(corpus.parent / "out"), vocab_size=500)
    return Tokenizer(str(out_path))


def test_vocab_size(toy_tokenizer):
    # 小样本语料合并对不足，词表可能达不到目标值；真实 2.5GB 语料可收敛到 32000
    assert toy_tokenizer.vocab_size <= 500


def test_special_token_ids(toy_tokenizer):
    ids = [toy_tokenizer.pad_id, toy_tokenizer.unk_id,
           toy_tokenizer.bos_id, toy_tokenizer.eos_id]
    assert ids == list(range(len(SPECIAL_TOKENS)))


def test_roundtrip(toy_tokenizer):
    for text in SAMPLE_LINES[:4]:
        assert toy_tokenizer.decode(toy_tokenizer.encode(text)) == text


def test_encode_special_markers(toy_tokenizer):
    ids = toy_tokenizer.encode("测试", add_bos=True, add_eos=True)
    assert ids[0] == toy_tokenizer.bos_id
    assert ids[-1] == toy_tokenizer.eos_id
