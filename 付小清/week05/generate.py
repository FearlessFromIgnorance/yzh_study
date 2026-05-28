"""加载权重并继续生成文本（自回归）。"""

from __future__ import annotations

import argparse
import json
import os

import torch

from model import DecoderOnlyLM


def main() -> None:
    ap = argparse.ArgumentParser(description="文本生成：单向 Transformer LM")
    ap.add_argument("--ckpt", type=str, default="checkpoints/lm.pt")
    ap.add_argument("--meta", type=str, default="", help="若为空则从 ckpt 同目录读 meta.json")
    ap.add_argument("--prompt", type=str, default="人工智能", help="起始提示（字符）")
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--temperature", type=float, default=0.9)
    ap.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = ap.parse_args()

    meta_path = args.meta.strip() or os.path.join(os.path.dirname(args.ckpt), "meta.json")
    if not os.path.isfile(meta_path):
        raise FileNotFoundError(f"找不到 meta.json: {meta_path}，请先运行 train.py")

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    stoi = meta["stoi"]
    mc = meta["model_config"]
    vocab_size = int(meta["vocab_size"])
    block_size = int(meta["block_size"])

    model = DecoderOnlyLM(
        vocab_size=vocab_size,
        block_size=block_size,
        n_layers=int(mc["n_layers"]),
        n_heads=int(mc["n_heads"]),
        d_model=int(mc["d_model"]),
        d_ff=int(mc["d_ff"]),
    ).to(args.device)

    ck = torch.load(args.ckpt, map_location=args.device)
    model.load_state_dict(ck["model_state"])
    model.eval()

    unknown = set(args.prompt) - set(stoi.keys())
    if unknown:
        raise ValueError(f"prompt 中含训练时未见字符: {unknown}")

    ids = torch.tensor([[stoi[c] for c in args.prompt]], dtype=torch.long, device=args.device)
    out = model.generate(ids, max_new_tokens=args.max_new_tokens, temperature=args.temperature)
    itos = {int(k): v for k, v in meta["itos"].items()}
    text = "".join(itos[int(t)] for t in out[0].tolist())

    print(text)


if __name__ == "__main__":
    main()
