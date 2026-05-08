import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
N_SAMPLES   = 4000
# 固定 5 个字！
MAXLEN      = 5
EMBED_DIM   = 64
HIDDEN_DIM  = 64
LR          = 1e-3
BATCH_SIZE  = 64
EPOCHS      = 20
TRAIN_RATIO = 0.8
NUM_CLS     = 5       # 5分类

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 生成数据：“你”字在第几位，就是第几类 ────────────────
def make_sentence(label_cls):
    """
    label_cls: 0~4 代表“你”在第 1~5 位
    """
    # 5个位置的句子模板
    chars = ['我', '他', '她', '它', '们']
    sentence = [''] * 5
    
    # 在指定位置放入“你”
    sentence[label_cls] = '你'
    
    # 其他位置随机填充
    for i in range(5):
        if sentence[i] == '':
            sentence[i] = random.choice(chars)
            
    return ''.join(sentence)


def build_dataset(n=N_SAMPLES):
    data = []
    for _ in range(n // NUM_CLS):
        for label in range(NUM_CLS):
            sent = make_sentence(label)
            data.append((sent, label))
    random.shuffle(data)
    return data


# ─── 2. 词表构建与编码 ──────────────────────────────────────
def build_vocab(data):
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    ids  = [vocab.get(ch, 1) for ch in sent]
    ids  = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# ─── 3. Dataset / DataLoader ────────────────────────────────
class TextDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),
        )


# ─── 4. 模型定义 ───────────────────────────
class KeywordRNN(nn.Module):
    def __init__(self, vocab_size, use_rnn=True):
        super().__init__()
        self.use_rnn = use_rnn
        self.embedding = nn.Embedding(vocab_size, EMBED_DIM, padding_idx=0)
        
        if self.use_rnn:
            self.rnn = nn.RNN(EMBED_DIM, HIDDEN_DIM, batch_first=True)
            self.fc = nn.Linear(HIDDEN_DIM, NUM_CLS)
        else:
            self.fc = nn.Linear(EMBED_DIM, NUM_CLS)

    def forward(self, x):
        e = self.embedding(x)  
        
        if self.use_rnn:
            e, _ = self.rnn(e)
        
        # Max Pooling 会丢失【位置信息】！
        # 不加RNN，模型根本不知道“你”在哪里
        pooled = e.max(dim=1)[0]
        out = self.fc(pooled)
        return out


# ─── 5. 评估函数 ──────────────────────────────────────────
def evaluate(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X, y in loader:
            y_pred = model(X)
            pred = torch.argmax(y_pred, dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return correct / total if total > 0 else 0


# ─── 6. 训练函数 ──────────────────────────────────────────
def train(use_rnn=True):
    print(f"\n======= use_rnn = {use_rnn} =======")
    data  = build_dataset(N_SAMPLES)
    vocab = build_vocab(data)

    split      = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data   = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(TextDataset(val_data,   vocab), batch_size=BATCH_SIZE)

    model     = KeywordRNN(len(vocab), use_rnn=use_rnn)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0
        for X, y in train_loader:
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS} loss={avg_loss:.4f} val_acc={val_acc:.4f}")


if __name__ == '__main__':
    # 不加 RNN → 20% 左右
    train(use_rnn=False)
    # 加上 RNN → 100%
    train(use_rnn=True)