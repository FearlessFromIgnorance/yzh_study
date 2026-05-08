"""
任务：判断“你”字在五字文本中的位置（1-5分类）
输入：长度为5的中文字符串（必须包含一个“你”字）
输出：位置类别（0-based索引：0,1,2,3,4）
模型：Embedding → RNN/LSTM → MaxPooling → Linear → CrossEntropyLoss
"""

import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# ─── 超参数 ────────────────────────────────────────────────
SEED = 42
N_SAMPLES = 10000  # 总样本数
MAXLEN = 5  # 固定长度5
EMBED_DIM = 32
HIDDEN_DIM = 64
NUM_CLASSES = 5  # 5个类别：位置0,1,2,3,4
LR = 1e-3
BATCH_SIZE = 64
EPOCHS = 30
TRAIN_RATIO = 0.8

random.seed(SEED)
torch.manual_seed(SEED)

# ─── 1. 数据生成 ────────────────────────────────────────────
# 常用中文字符池（用于填充其他位置）
CHAR_POOL = list('天地玄黄宇宙洪荒日月盈昃辰宿列张寒来暑往秋收冬藏')
# 用于替代"你"的其他字（干扰项，用于生成无效样本时可选）
OTHER_CHARS = list('你我他她它这那这些那些')


def generate_sample_with_ni_position():
    """生成一个包含“你”字的5字文本，并返回标签（0-4）"""
    position = random.randint(0, 4)  # 随机决定“你”的位置
    chars = []
    for i in range(MAXLEN):
        if i == position:
            chars.append('你')
        else:
            chars.append(random.choice(CHAR_POOL))
    text = ''.join(chars)
    return text, position


def generate_invalid_sample():
    """生成不包含“你”字的样本（用于测试集或增强）"""
    chars = [random.choice(OTHER_CHARS) for _ in range(MAXLEN)]
    # 确保不包含'你'
    while '你' in chars:
        idx = chars.index('你')
        chars[idx] = random.choice(CHAR_POOL)
    return ''.join(chars), -1  # 标签-1表示无效


def build_dataset(n=N_SAMPLES, include_invalid=False):
    """构建数据集

    Args:
        n: 样本总数
        include_invalid: 是否包含无效样本（不含“你”字）
    """
    data = []

    # 生成有效样本（包含“你”字）
    valid_count = n if not include_invalid else int(n * 0.9)
    for _ in range(valid_count):
        text, label = generate_sample_with_ni_position()
        data.append((text, label))

    # 可选：生成无效样本
    if include_invalid:
        invalid_count = n - valid_count
        for _ in range(invalid_count):
            text, label = generate_invalid_sample()
            data.append((text, label))

    random.shuffle(data)
    return data


# ─── 2. 词表构建 ────────────────────────────────────────────
def build_vocab(data):
    """构建字符到ID的映射"""
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    """将文本编码为固定长度的ID序列"""
    ids = [vocab.get(ch, 1) for ch in sent]  # 未知字符用<UNK>(1)表示
    ids = ids[:maxlen]  # 截断
    ids += [0] * (maxlen - len(ids))  # 填充
    return ids


# ─── 3. Dataset ─────────────────────────────────────────────
class PositionDataset(Dataset):
    def __init__(self, data, vocab):
        self.X = [encode(s, vocab) for s, _ in data]
        self.y = [lb for _, lb in data]

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return (
            torch.tensor(self.X[i], dtype=torch.long),
            torch.tensor(self.y[i], dtype=torch.long),  # 多分类用long
        )


# ─── 4. RNN 模型 ───────────────────────────────────────────
class PositionRNN(nn.Module):
    """基础RNN模型（使用最后时刻的输出）"""

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, nonlinearity='tanh')
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        rnn_out, hidden = self.rnn(embedded)  # rnn_out: (batch, seq_len, hidden_dim)

        # 取最后一个时间步的输出
        last_output = rnn_out[:, -1, :]  # (batch, hidden_dim)
        last_output = self.dropout(last_output)
        logits = self.fc(last_output)  # (batch, num_classes)
        return logits


# ─── 5. LSTM 模型（更好捕捉序列依赖性） ──────────────────────
class PositionLSTM(nn.Module):
    """LSTM模型（使用最后时刻的输出）"""

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=2, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # 取最后一层最后一个时间步的输出
        # hidden: (num_layers, batch, hidden_dim)
        last_hidden = hidden[-1]  # (batch, hidden_dim)
        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden)  # (batch, num_classes)
        return logits


# ─── 6. 带MaxPooling的LSTM ────────────────────────
class PositionLSTMWithPooling(nn.Module):
    """LSTM + MaxPooling（聚合所有时刻的信息）"""

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM,
                 num_layers=2, num_classes=NUM_CLASSES, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, num_classes)  # 双向需要*2

    def forward(self, x):
        # x: (batch, seq_len)
        embedded = self.embedding(x)  # (batch, seq_len, embed_dim)
        lstm_out, _ = self.lstm(embedded)  # (batch, seq_len, hidden_dim*2)

        # Max pooling over time dimension
        pooled = lstm_out.max(dim=1)[0]  # (batch, hidden_dim*2)
        pooled = self.dropout(pooled)
        logits = self.fc(pooled)  # (batch, num_classes)
        return logits


# ─── 7. 训练与评估 ──────────────────────────────────────────
def evaluate(model, loader, criterion):
    """评估函数（返回准确率和损失）"""
    model.eval()
    correct = 0
    total = 0
    total_loss = 0.0

    with torch.no_grad():
        for X, y in loader:
            logits = model(X)
            loss = criterion(logits, y)
            total_loss += loss.item()

            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += len(y)

    accuracy = correct / total
    avg_loss = total_loss / len(loader)
    return accuracy, avg_loss


def train_model(model, train_loader, val_loader, model_name, epochs=EPOCHS):
    """训练单个模型并记录历史"""
    print(f"\n{'=' * 60}")
    print(f"训练 {model_name}")
    print(f"{'=' * 60}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        # 训练阶段
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for X, y in train_loader:
            optimizer.zero_grad()
            logits = model(X)
            loss = criterion(logits, y)
            loss.backward()

            # 梯度裁剪，防止梯度爆炸
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            train_loss += loss.item()
            pred = logits.argmax(dim=1)
            train_correct += (pred == y).sum().item()
            train_total += len(y)

        train_acc = train_correct / train_total
        train_loss_avg = train_loss / len(train_loader)

        # 验证阶段
        val_acc, val_loss = evaluate(model, val_loader, criterion)

        # 调整学习率
        scheduler.step()

        # 保存历史
        history['train_loss'].append(train_loss_avg)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f'best_{model_name.lower()}.pth')

        # 打印进度
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:3d}/{epochs} | "
                  f"Train Loss: {train_loss_avg:.4f} | Train Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

    print(f"\n{model_name} 最佳验证准确率: {best_val_acc:.4f}")
    return history, best_val_acc


def plot_results(histories):
    """绘制训练曲线对比"""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # 损失曲线
    for name, history in histories.items():
        axes[0].plot(history['val_loss'], label=f'{name} (val)', linestyle='--')
        axes[0].plot(history['train_loss'], label=f'{name} (train)', alpha=0.5)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 准确率曲线
    for name, history in histories.items():
        axes[1].plot(history['val_acc'], label=f'{name} (val)', linestyle='--', marker='o')
        axes[1].plot(history['train_acc'], label=f'{name} (train)', alpha=0.5)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Curves')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=150)
    plt.show()


def inference_example(model, vocab, test_texts):
    """推理示例"""
    model.eval()
    print("\n" + "=" * 60)
    print("推理示例")
    print("=" * 60)

    with torch.no_grad():
        for text in test_texts:
            # 编码并确保长度为5
            if len(text) != 5:
                print(f"警告: '{text}' 长度不为5，将进行截断或填充")
            ids = torch.tensor([encode(text, vocab)], dtype=torch.long)
            logits = model(ids)
            prob = torch.softmax(logits, dim=1)
            pred_class = logits.argmax(dim=1).item()

            # 找到"你"的实际位置
            actual_pos = text.find('你') if '你' in text else -1

            print(f"\n文本: '{text}'")
            print(f"  预测类别: {pred_class} (第{pred_class + 1}位)")
            print(f"  实际位置: {actual_pos} (第{actual_pos + 1}位)" if actual_pos != -1 else "  实际: 不含'你'")
            print(f"  各类别概率: {[f'{p:.3f}' for p in prob[0].tolist()]}")


# ─── 8. 主函数 ──────────────────────────────────────────────
def main():
    print("生成数据集...")
    # 只使用有效样本（都包含"你"字）
    data = build_dataset(N_SAMPLES, include_invalid=False)

    # 分割数据集
    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    # 构建词表
    vocab = build_vocab(data)
    print(f"样本总数: {len(data)}")
    print(f"训练集: {len(train_data)}, 验证集: {len(val_data)}")
    print(f"词表大小: {len(vocab)}")

    # 创建数据加载器
    train_dataset = PositionDataset(train_data, vocab)
    val_dataset = PositionDataset(val_data, vocab)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    # 显示一个batch示例
    sample_batch = next(iter(train_loader))
    print(f"\nBatch示例 - X形状: {sample_batch[0].shape}, y形状: {sample_batch[1].shape}")

    # 词表大小
    vocab_size = len(vocab)

    # 初始化模型
    models = {
        'RNN': PositionRNN(vocab_size, EMBED_DIM, HIDDEN_DIM, NUM_CLASSES),
        'LSTM': PositionLSTM(vocab_size, EMBED_DIM, HIDDEN_DIM, num_layers=2, num_classes=NUM_CLASSES),
        'LSTM_Pool': PositionLSTMWithPooling(vocab_size, EMBED_DIM, HIDDEN_DIM,
                                             num_layers=2, num_classes=NUM_CLASSES)
    }

    # 打印模型参数量
    print("\n模型参数量:")
    for name, model in models.items():
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"  {name:12s}: 总计 {total_params:,} | 可训练 {trainable_params:,}")

    # 训练所有模型
    histories = {}
    best_accuracies = {}

    for name, model in models.items():
        history, best_acc = train_model(model, train_loader, val_loader, name)
        histories[name] = history
        best_accuracies[name] = best_acc

    # 打印最终结果对比
    print("\n" + "=" * 60)
    print("最终结果对比")
    print("=" * 60)
    for name, acc in sorted(best_accuracies.items(), key=lambda x: x[1], reverse=True):
        print(f"{name:12s}: 最佳验证准确率 = {acc:.4f}")

    # 可视化训练过程
    plot_results(histories)

    # 选择最佳模型进行推理演示
    best_model_name = max(best_accuracies, key=best_accuracies.get)
    best_model = models[best_model_name]
    best_model.load_state_dict(torch.load(f'best_{best_model_name.lower()}.pth'))

    print(f"\n使用最佳模型: {best_model_name}")

    # 测试用例
    test_texts = [
        "你我他她它",  # 位置0
        "天你地玄黄",  # 位置1
        "天地你宇宙",  # 位置2
        "天地玄你黄",  # 位置3
        "天地玄黄你",  # 位置4
        "你好天天地",  # 位置0的变体
        "天天你好地",  # 位置2
        "天天天地你",  # 位置4
    ]

    inference_example(best_model, vocab, test_texts)


if __name__ == '__main__':
    main()
