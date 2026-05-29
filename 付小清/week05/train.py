"""训练脚本：读取文本，字符级词典，训练 DecoderOnlyLM。"""

from __future__ import annotations

import argparse
import json
import os

import torch
from torch.utils.data import DataLoader, Dataset

from model import DecoderOnlyLM, count_parameters


class CharDataset(Dataset):
    def __init__(self, text: str, block_size: int) -> None:
        self.chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.vocab_size = len(self.chars)
        self.block_size = block_size

        data = torch.tensor([self.stoi[ch] for ch in text], dtype=torch.long)

        n = len(data) - block_size
        if n <= 0:
            raise ValueError(
                f"文本太短：至少需要 block_size+1={block_size + 1} 个字符，当前 {len(data)}。"
            )
        self.x = torch.stack([data[i : i + block_size] for i in range(n)])
        self.y = torch.stack([data[i + 1 : i + block_size + 1] for i in range(n)])

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, i: int):
        return self.x[i], self.y[i]


def main() -> None:
    ap = argparse.ArgumentParser(description="训练单向 Transformer 语言模型（字符级）")
    ap.add_argument("--data", type=str, default="data/sample_zh.txt", help="训练文本路径")
    ap.add_argument("--out_dir", type=str, default="checkpoints", help="保存目录")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--block_size", type=int, default=64, help="上下文长度")
    ap.add_argument("--n_layers", type=int, default=4)
    ap.add_argument("--n_heads", type=int, default=4)
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--d_ff", type=int, default=512)
    ap.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    with open(args.data, encoding="utf-8") as f:
        text = f.read()
    text = text.strip()
    if not text:
        raise SystemExit(f"训练文件为空: {args.data}")

    ds = CharDataset(text, args.block_size)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True, drop_last=False)

    model = DecoderOnlyLM(
        vocab_size=ds.vocab_size,
        block_size=args.block_size,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
    ).to(args.device)

    meta = {
        "stoi": ds.stoi,
        "itos": {str(k): v for k, v in ds.itos.items()},
        "vocab_size": ds.vocab_size,
        "block_size": args.block_size,
        "model_config": {
            "n_layers": args.n_layers,
            "n_heads": args.n_heads,
            "d_model": args.d_model,
            "d_ff": args.d_ff,
        },
    }
    meta_path = os.path.join(args.out_dir, "meta.json")
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, ensure_ascii=False, indent=2)

    print(f"参数量: {count_parameters(model):,}")
    print(f"词典大小: {ds.vocab_size}，样本数: {len(ds)}, device={args.device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        steps = 0
        for xb, yb in loader:
            xb, yb = xb.to(args.device), yb.to(args.device)
            opt.zero_grad(set_to_none=True)
            _, loss = model(xb, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
            total_loss += loss.item()
            steps += 1
        avg = total_loss / max(steps, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch + 1}/{args.epochs}  avg_loss={avg:.4f}")

    ckpt = os.path.join(args.out_dir, "lm.pt")
    torch.save({"model_state": model.state_dict(), "meta_ref": meta_path}, ckpt)
    print(f"已保存: {ckpt} 与元数据 {meta_path}")


if __name__ == "__main__":
    main()
