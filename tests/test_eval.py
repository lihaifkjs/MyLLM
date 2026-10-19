"""模块7 评测与推理冒烟测试：合成输入 + tiny 模型，CPU 验证数学与契约（无需网络/GPU）。"""
from types import SimpleNamespace

import pytest
import torch

from src.chat.chat import build_prompt_ids
from src.chat.decoding import generate
from src.eval.case_compare import is_repetitive
from src.eval.ceval import build_prompt, letter_ids, score_questions
from src.eval.ppl import compute_nll
from src.model import LlamaModel, ModelConfig
from src.tokenizer.tokenizer import Tokenizer


class FakeModel:
    """可调 logits 的假模型：fn(当前序列) → 末位 logits 向量，模拟 LlamaModel 前向契约。"""

    def __init__(self, max_seq_len: int, vocab_size: int, fn):
        self.cfg = SimpleNamespace(max_seq_len=max_seq_len)
        self.vocab_size = vocab_size
        self._fn = fn

    def eval(self):
        pass

    @torch.no_grad()
    def __call__(self, x):
        logits = self._fn(x[0].tolist())
        return logits.view(1, 1, -1).expand(1, x.shape[1], -1).clone()


@pytest.fixture(scope="module")
def tok():
    return Tokenizer("tokenizer/tokenizer.model")


# ---------- C-Eval 打分 ----------

def test_letter_ids_single_token(tok):
    lids = letter_ids(tok)
    assert set(lids) == {"A", "B", "C", "D"}
    assert len(set(lids.values())) == 4  # 四个字母对应四个不同 token


def test_score_questions_pred():
    real_tok = Tokenizer("tokenizer/tokenizer.model")
    lids = letter_ids(real_tok)
    vocab = max(lids.values()) + 1
    logits = torch.full((vocab,), -10.0)
    logits[lids["C"]] = 10.0  # 模型永远偏向选 C
    model = FakeModel(1024, vocab, lambda ids: logits)
    questions = [
        {"subject": "math", "question": "1+1=?", "A": "1", "B": "2", "C": "3", "D": "4", "answer": "C"},
        {"subject": "math", "question": "2+2=?", "A": "3", "B": "4", "C": "5", "D": "6", "answer": "B"},
    ]
    results = score_questions(model, real_tok, questions, torch.device("cpu"))
    assert [r["pred"] for r in results] == ["C", "C"]
    assert [r["correct"] for r in results] == [True, False]  # 准确率 0.5


def test_build_prompt_format():
    q = {"subject": "math", "question": "1+1=?", "A": "1", "B": "2", "C": "3", "D": "4"}
    p = build_prompt("math", q)
    assert p.startswith("以下是中国关于math的考试题目")
    assert "1+1=?" in p and "A. 1" in p and p.endswith("答案：")


# ---------- 解码与复读缓解 ----------

def _prefer_tok7_logits(vocab):
    logits = torch.zeros(vocab)
    logits[7] = 10.0
    logits[8] = 8.0
    return logits


def test_generate_greedy_repetition_penalty():
    tok = SimpleNamespace(eos_id=0)
    vocab = 10
    model = FakeModel(64, vocab, lambda ids: _prefer_tok7_logits(vocab))
    plain = generate(model, tok, [1, 2], torch.device("cpu"), max_new_tokens=3)
    assert plain == [7, 7, 7]  # 无惩罚：复读
    penalized = generate(model, tok, [1, 2], torch.device("cpu"), max_new_tokens=2,
                         repetition_penalty=2.0)
    # 第一步选 7；7 被惩罚后 10/2=5 < 8，第二步改选 8
    assert penalized == [7, 8]


def test_generate_no_repeat_ngram():
    tok = SimpleNamespace(eos_id=0)
    vocab = 10

    def fn(ids):
        logits = torch.zeros(vocab)
        logits[1] = 10.0  # 永远偏向 token 1
        logits[2] = 5.0
        return logits

    model = FakeModel(64, vocab, fn)
    plain = generate(model, tok, [3], torch.device("cpu"), max_new_tokens=4)
    assert plain == [1, 1, 1, 1]
    blocked = generate(model, tok, [3], torch.device("cpu"), max_new_tokens=4,
                       no_repeat_ngram=2)
    # (1,1) 被禁后让位给 token 2，随后 (2,1)/(1,2) 未出现过，1 与 2 交替
    assert blocked == [1, 1, 2, 1]


# ---------- 多轮拼接与截断 ----------

def test_build_prompt_ids_truncation(tok):
    history = [(f"第{i}个问题", f"第{i}个回答" * 20) for i in range(5)]
    ids = build_prompt_ids(tok, history, "当前问题", max_len=48)
    assert len(ids) <= 48
    text = tok.decode(ids)
    assert text.endswith("用户：当前问题\n助手：")
    assert "第0个问题" not in text  # 最早轮被丢弃


def test_build_prompt_ids_single_oversize(tok):
    ids = build_prompt_ids(tok, [], "很" * 500, max_len=32)
    assert len(ids) == 32  # 单轮超长：硬截窗口尾部


# ---------- ppl 数学 ----------

def test_compute_nll_math():
    cfg = ModelConfig(vocab_size=64, d_model=32, n_layers=1, n_heads=2,
                      ffn_hidden=64, max_seq_len=32)
    model = LlamaModel(cfg)
    model.eval()
    ids = list(range(1, 26))  # 3 个完整窗口（seq_len=8），尾部 1 个 token 丢弃
    nll, n_tok = compute_nll(model, ids, torch.device("cpu"), seq_len=8, batch_size=2)
    # 手算对照：逐窗口 teacher forcing 的交叉熵之和
    manual = 0.0
    for j in range(3):
        x = torch.tensor([ids[j * 8: j * 8 + 8]])
        y = torch.tensor([ids[j * 8 + 1: j * 8 + 9]])
        logits = model(x)
        manual += torch.nn.functional.cross_entropy(
            logits.reshape(-1, 64), y.reshape(-1), reduction="sum").item()
    assert n_tok == 24
    assert abs(nll - manual) < 1e-4


# ---------- 复读判定启发式 ----------

def test_is_repetitive():
    assert is_repetitive("ababababab")       # "abab" 出现 7 次
    assert not is_repetitive("今天天气不错，适合出去走走。")
    assert not is_repetitive("短")
