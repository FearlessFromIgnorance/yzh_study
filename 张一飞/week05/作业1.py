import torch
import torch.nn as nn

# 准备数据
text = "床前明月光，疑是地上霜。举头望明月，低头思故乡。"
chars = sorted(list(set(text)))  # 获取去重后的字符表
vocab_size = len(chars)
char2id = {c: i for i, c in enumerate(chars)}
id2char = {i: c for i, c in enumerate(chars)}

# 错位预测数据集：X 是当前文本，Y 是向后偏移一位的下一个字
raw_data = [char2id[c] for c in text]
X = torch.tensor([raw_data[:-1]], dtype=torch.long)
Y = torch.tensor([raw_data[1:]], dtype=torch.long)


# 定义单向 Transformer 语言模型
class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=32, nhead=2, num_layers=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(100, d_model)

        # 使用 Transformer 编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=64, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        seq_len = x.size(1)
        # 生成上三角因果掩码，防止当前位置看到未来的字
        mask = nn.Transformer.generate_square_subsequent_mask(seq_len).to(x.device)

        # 词嵌入 + 位置嵌入
        pos = torch.arange(seq_len, device=x.device).unsqueeze(0)
        out = self.token_emb(x) + self.pos_emb(pos)

        # 传入 mask，使其变为单向注意力
        out = self.transformer(out, mask=mask, is_causal=True)
        return self.fc(out)


model = MiniGPT(vocab_size)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# 模型训练
print("模型训练中（让 Transformer 背诗）...")
for epoch in range(150):
    optimizer.zero_grad()
    output = model(X)  # shape: (1, seq_len, vocab_size)
    loss = criterion(output.view(-1, vocab_size), Y.view(-1))
    loss.backward()
    optimizer.step()

# 自回归文本生成（Autoregressive Generation）

print("\n--- 文本生成测试 ---")
model.eval()
start_char = "床"
generated = start_char
input_seq = torch.tensor([[char2id[start_char]]], dtype=torch.long)

with torch.no_grad():
    for _ in range(22):  # 往后生成 22 个字
        output = model(input_seq)

        # 每次只取最后一个时间步（最新预测的那一个字）的概率分布
        next_token_logits = output[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).item()

        # 解码并记录
        next_char = id2char[next_token_id]
        generated += next_char

        # 将新生成的字拼接到输入序列的末尾，滚动作为下一次的输入
        input_seq = torch.cat([input_seq, torch.tensor([[next_token_id]])], dim=1)

print(f"给定开头: '{start_char}'")
print(f"生成结果: {generated}")
