import torch
import torch.nn as nn
import torch.nn.functional as F
import random

# 超参数
VOCAB_SIZE    = 26    # a~z 26个字母
EMBED_DIM     = 128
NUM_HEADS     = 4
NUM_LAYERS    = 2     # Transformer 层数
HIDDEN_DIM    = 256
SEQ_LEN       = 16    # 句子长度
BATCH_SIZE    = 16
EPOCHS        = 20
LR            = 1e-3
DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"

# 1. 造简单数据
vocab = [chr(ord('a')+i) for i in range(26)]
w2i = {c:i for i,c in enumerate(vocab)}
i2w = {i:c for i,c in enumerate(vocab)}

def generate_data(n_samples=2000, seq_len=SEQ_LEN):
    data = []
    for _ in range(n_samples):
        seq = [random.choice(vocab) for _ in range(seq_len)]
        ids = [w2i[w] for w in seq]
        data.append(ids)
    return torch.tensor(data, dtype=torch.long)

# 2. 位置编码
class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=SEQ_LEN):
        super().__init__()
        self.pos_emb = nn.Embedding(max_len, embed_dim)
    def forward(self, x):
        B, T, C = x.shape
        pos = torch.arange(T, device=DEVICE).unsqueeze(0)
        return x + self.pos_emb(pos)

# 3. 多头掩码自注意力
class CausalMultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.qkv = nn.Linear(embed_dim, embed_dim*3)
        self.out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.view(B, T, self.num_heads, self.head_dim).transpose(1,2), qkv)

        # 注意力分数
        attn = (q @ k.transpose(-2,-1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # 掩码：挡住未来位置
        mask = torch.tril(torch.ones(T, T, device=DEVICE))
        attn = attn.masked_fill(mask == 0, -1e9)

        attn = F.softmax(attn, dim=-1)
        out = attn @ v
        out = out.transpose(1,2).contiguous().view(B,T,C)
        return self.out(out)

# 4. Transformer 层
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, hidden_dim):
        super().__init__()
        self.attn = CausalMultiHeadAttention(embed_dim, num_heads)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x

# 5. 单向语言模型
class GPTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(VOCAB_SIZE, EMBED_DIM)
        self.pos_enc = PositionalEncoding(EMBED_DIM)
        self.layers = nn.Sequential(*[
            TransformerLayer(EMBED_DIM, NUM_HEADS, HIDDEN_DIM) for _ in range(NUM_LAYERS)
        ])
        self.norm = nn.LayerNorm(EMBED_DIM)
        self.head = nn.Linear(EMBED_DIM, VOCAB_SIZE)

    def forward(self, x):
        x = self.emb(x)
        x = self.pos_enc(x)
        x = self.layers(x)
        x = self.norm(x)
        logits = self.head(x)
        return logits

# 6. 训练
def train():
    model = GPTModel().to(DEVICE)
    data = generate_data().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    criterion = nn.CrossEntropyLoss()

    print("开始训练...")
    for epoch in range(EPOCHS):
        model.train()
        idx = torch.randperm(data.shape[0])
        batches = data[idx].split(BATCH_SIZE)
        total_loss = 0

        for batch in batches:
            x = batch[:, :-1]  # 输入
            y = batch[:, 1:]   # 目标：下一个字
            logits = model(x)
            loss = criterion(logits.reshape(-1, VOCAB_SIZE), y.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"Epoch {epoch+1:2d} | loss = {total_loss/len(batches):.4f}")

    return model

# 7. 文本生成
@torch.no_grad()
def generate(model, prompt, max_gen=10):
    model.eval()
    ids = [w2i[c] for c in prompt]
    ids = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(DEVICE)

    for _ in range(max_gen):
        logits = model(ids)
        next_logits = logits[:, -1, :]
        next_id = torch.argmax(next_logits, dim=-1, keepdim=True)  # 贪心采样
        ids = torch.cat([ids, next_id], dim=1)

    gen_ids = ids[0].cpu().numpy()
    return ''.join([i2w[i] for i in gen_ids])

# 测试
if __name__ == "__main__":
    model = train()

    # 测试生成
    prompt = "abc"
    gen_text = generate(model, prompt)
    print("\n【生成结果】")
    print(f"输入：{prompt}")
    print(f"输出：{gen_text}")