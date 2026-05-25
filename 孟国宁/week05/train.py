"""
    训练基于transformer的单向语言模型，并完成文本生成。
"""

import torch
import torch.nn as nn
import torch.optim as optim
from model import GPT
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def build_vocab(text):
    chars = sorted(list(set(text)))
    vocab_size = len(chars)
    char_to_idx = {ch: i for i, ch in enumerate(chars)}
    idx_to_char = {i: ch for i, ch in enumerate(chars)}
    return chars, vocab_size, char_to_idx, idx_to_char

def encode(text, char_to_idx):
    return [char_to_idx[ch] for ch in text]

def decode(idx_list, idx_to_char):
    return ''.join([idx_to_char[idx] for idx in idx_list])

def get_batch(data, batch_size, block_size):
    B, T = batch_size, block_size
    ix = torch.randint(len(data) - T, (B,))
    x = torch.stack([torch.tensor(data[i:i+T], dtype=torch.long) for i in ix])
    y = torch.stack([torch.tensor(data[i+1:i+T+1], dtype=torch.long) for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

def train():
    text = load_data('input.txt')
    chars, vocab_size, char_to_idx, idx_to_char = build_vocab(text)
    data = encode(text, char_to_idx)

    batch_size = 64
    block_size = 256
    d_model = 512
    num_heads = 8
    hidden_dim = 2048
    num_layers = 6
    dropout = 0.1
    learning_rate = 3e-4
    max_iters = 5000
    eval_interval = 500
    eval_iters = 200

    model = GPT(vocab_size, d_model, num_heads, hidden_dim, num_layers, dropout).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    def estimate_loss():
        model.eval()
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(data, batch_size, block_size)
            logits = model(X)
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            Y = Y.view(B*T)
            loss = criterion(logits, Y)
            losses[k] = loss.item()
        model.train()
        return losses.mean()

    for iter in range(max_iters):
        if iter % eval_interval == 0:
            loss = estimate_loss()
            print(f"Step {iter}: loss {loss:.4f}")

        xb, yb = get_batch(data, batch_size, block_size)

        logits = model(xb)
        B, T, C = logits.shape
        logits = logits.view(B*T, C)
        yb = yb.view(B*T)
        loss = criterion(logits, yb)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), 'gpt_model.pth')
    print("Model saved to gpt_model.pth")

    model.eval()
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    generated = decode(model.generate(context, max_new_tokens=500)[0].tolist(), idx_to_char)
    print("Generated text:")
    print(generated)

if __name__ == "__main__":
    train()
