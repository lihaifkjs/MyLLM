"""过拟合检查：验证模型具备学习能力（docs/04 验证方案第二层）。
用一小段真实文本反复训练，loss 应降到接近 0；降不动说明 RoPE/attention 等实现有误。
用法：python3 -m src.model.overfit_check
"""
import time

import torch
import torch.nn.functional as F

from src.model import LlamaModel, ModelConfig
from src.tokenizer.tokenizer import Tokenizer

TEXT = (
    "人工智能是计算机科学的一个分支，它企图了解智能的实质，"
    "并生产出一种新的能以人类智能相似的方式做出反应的智能机器。"
    "该领域的研究包括机器人、语言识别、图像识别、自然语言处理和专家系统等。"
)
STEPS = 300
LR = 1e-3


def main() -> None:
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    torch.manual_seed(0)

    ids = torch.tensor(Tokenizer().encode(TEXT, add_bos=True), dtype=torch.long, device=device)
    inputs, targets = ids[:-1], ids[1:]
    print(f"设备={device} 序列长度={len(ids)}")

    model = LlamaModel(ModelConfig()).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=LR)
    model.train()

    t0 = time.time()
    for step in range(1, STEPS + 1):
        logits = model(inputs.unsqueeze(0))
        loss = F.cross_entropy(logits[0], targets)
        optim.zero_grad()
        loss.backward()
        optim.step()
        if step == 1 or step % 50 == 0:
            print(f"step {step:4d}  loss {loss.item():.4f}")

    model.eval()
    with torch.no_grad():
        pred = model(inputs.unsqueeze(0))[0].argmax(-1)
    acc = (pred == targets).float().mean().item()
    print(f"最终 token 命中率 {acc:.2%}，耗时 {time.time() - t0:.1f}s")
    assert loss.item() < 0.05 and acc > 0.95, "过拟合失败：模型学不动，实现可能有误"
    print("过拟合检查通过")


if __name__ == "__main__":
    main()
