"""模块3冒烟测试：参数量、形状契约、因果性、反向传播。见 docs/04-模型定义.md。"""
import torch

from src.model import LlamaModel, ModelConfig

EXPECTED_PARAMS = 109_529_856  # embed 24.6M + 12 层 * 7.08M + 末层 norm


def make_model() -> LlamaModel:
    torch.manual_seed(0)
    return LlamaModel(ModelConfig())


def test_param_count():
    n = sum(p.numel() for p in make_model().parameters())
    assert abs(n - EXPECTED_PARAMS) / EXPECTED_PARAMS < 0.01, f"参数量 {n}"


def test_forward_shape_and_finite():
    model = make_model()
    ids = torch.randint(0, 32000, (2, 16))
    logits = model(ids)
    assert logits.shape == (2, 16, 32000)
    assert torch.isfinite(logits).all()


def test_causality():
    """改动第 t 位输入，t 之前位置的 logits 必须不变。"""
    model = make_model().eval()
    ids = torch.randint(0, 32000, (1, 16))
    t = 8
    ids2 = ids.clone()
    ids2[0, t] = (ids2[0, t] + 1) % 32000
    with torch.no_grad():
        diff = (model(ids)[:, :t] - model(ids2)[:, :t]).abs().max()
    assert diff == 0, f"因果性被破坏，前 {t} 位 logits 最大差异 {diff}"


def test_backward():
    model = make_model()
    logits = model(torch.randint(0, 32000, (2, 16)))
    logits.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, f"{name} 无梯度"


def test_seq_len_limit():
    model = make_model()
    try:
        model(torch.randint(0, 32000, (1, 1025)))
    except ValueError:
        return
    raise AssertionError("超长序列未抛 ValueError")
