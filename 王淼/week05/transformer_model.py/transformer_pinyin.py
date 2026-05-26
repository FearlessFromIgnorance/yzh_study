"""
基于字符级 Transformer 语言模型的拼音输入法。
用法:
    python pinyin_ime.py --model_path best_transformer.pt --topk 8 --beam 10
"""

import argparse
import json
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# ─────────────────────── 拼音 → 候选汉字映射表 ───────────────────────

def _load_pinyin_map(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到拼音映射表文件: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)

PINYIN_MAP = {}

# ─────────────────────── 🔥 这里换成 Transformer 模型 ───────────────────────

class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, embed_dim, 2).float() * (-math.log(10000.0) / embed_dim))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerLM(nn.Module):
    def __init__(
        self, vocab_size, embed_dim, hidden_dim, num_layers, 
        num_heads=8, dropout=0.1, max_seq_len=512
    ):
        super().__init__()
        self.embed_dim = embed_dim
        
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.pos_encoder = PositionalEncoding(embed_dim, max_len=max_seq_len, dropout=dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Linear(embed_dim, vocab_size)

    def generate_causal_mask(self, seq_len, device):
        mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def forward(self, x):
        B, T = x.shape
        device = x.device
        
        x = self.embedding(x) * math.sqrt(self.embed_dim)
        x = self.pos_encoder(x)
        
        causal_mask = self.generate_causal_mask(T, device)
        out = self.transformer_encoder(x, mask=causal_mask)
        
        logits = self.fc(out)
        return logits

# ─────────────────────── 拼音分词 ───────────────────────

_SYLLABLES = []

def segment(pinyin_str):
    syllables = []
    for token in pinyin_str.strip().lower().split():
        i = 0
        while i < len(token):
            matched = next((s for s in _SYLLABLES if token[i:].startswith(s)), None)
            if matched:
                syllables.append(matched)
                i += len(matched)
            else:
                i += 1
    return syllables

# ─────────────────────── 束搜索 ───────────────────────

def beam_search(syllables, prefix, model, char2idx, idx2char, beam_size, device):
    beams = [(0.0, "")]

    for syllable in syllables:
        candidates = [c for c in PINYIN_MAP.get(syllable, []) if c in char2idx]
        if not candidates:
            continue

        new_beams = []
        for score, partial in beams:
            context = prefix + partial
            if context:
                ids = [char2idx[c] for c in context if c in char2idx]
                x = torch.tensor([ids], dtype=torch.long, device=device)
                with torch.no_grad():
                    logits = model(x)
                log_probs = F.log_softmax(logits[0, -1, :], dim=-1)
            else:
                log_probs = None

            for char in candidates:
                if log_probs is not None:
                    lp = log_probs[char2idx[char]].item()
                else:
                    lp = 0.0
                new_beams.append((score + lp, partial + char))

        new_beams.sort(reverse=True)
        beams = new_beams[:beam_size]

    return beams

# ─────────────────────── 交互主循环 ───────────────────────

def run(model, char2idx, idx2char, topk, beam_size, device):
    print("=" * 52)
    print("  Transformer 拼音输入法")
    print("  输入拼音回车 → 选候选编号追加到已输入文字")
    print("  r = 重置  q = 退出")
    print("=" * 52)

    confirmed = ""

    while True:
        print(f"\n已输入: 「{confirmed}」" if confirmed else "\n已输入: （空）")
        raw = input("拼音> ").strip()

        if not raw:
            continue
        if raw == "q":
            print("退出。")
            break
        if raw == "r":
            confirmed = ""
            continue

        syllables = segment(raw)
        if not syllables:
            print("无法识别任何音节，请检查拼音拼写。")
            continue

        print(f"音节: {' '.join(syllables)}")
        results = beam_search(syllables, confirmed, model, char2idx, idx2char, beam_size, device)

        if not results:
            print("无候选结果。")
            continue

        print("候选:")
        for i, (score, text) in enumerate(results[:topk]):
            print(f"  [{i}] {text}  ({score:.2f})")

        choice = input("选择编号 (回车跳过): ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 0 <= idx < len(results):
                confirmed += results[idx][1]
            else:
                print("编号超出范围。")

# ─────────────────────── 🔥 加载模型部分也改成 Transformer ───────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default="best_transformer.pt")
    parser.add_argument("--pinyin_map", default="pinyin_map.json")
    parser.add_argument("--topk",       type=int, default=5)
    parser.add_argument("--beam",       type=int, default=10)
    args = parser.parse_args()

    global PINYIN_MAP, _SYLLABLES
    PINYIN_MAP = _load_pinyin_map(args.pinyin_map)
    _SYLLABLES = sorted(PINYIN_MAP.keys(), key=len, reverse=True)
    print(f"拼音表: {args.pinyin_map}  ({len(PINYIN_MAP)} 个音节)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt     = torch.load(args.model_path, map_location=device)
    char2idx = ckpt["char2idx"]
    idx2char = ckpt["idx2char"]
    cfg      = ckpt["args"]

    # 加载 Transformer 模型
    model = TransformerLM(
        vocab_size=len(char2idx),
        embed_dim=cfg["embed_dim"],
        hidden_dim=cfg["hidden_dim"],
        num_layers=cfg["num_layers"],
        num_heads=cfg.get("num_heads", 8),
        dropout=0.0,
        max_seq_len=cfg["seq_len"]
    ).to(device)
    
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    print(f"模型: {args.model_path}  (Transformer, 词表 {len(char2idx)} 字)")
    run(model, char2idx, idx2char, args.topk, args.beam, device)

if __name__ == "__main__":
    main()
