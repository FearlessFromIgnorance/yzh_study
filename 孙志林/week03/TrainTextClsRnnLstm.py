"""
对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类
"""
import random

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# 超参数
SEED = 42  # 随机种子
N_SAMPLES = 4000  # 样本数
MAXLEN = 32  # 最大序列长度
EMBED_DIM = 64  # 嵌入维度
HIDDEN_DIM = 64  # 隐藏层维度
LR = 1e-3  # 学习率
BATCH_SIZE = 64  # 批次大小
EPOCHS = 20  # 训练轮数
TRAIN_RATIO = 0.8  # 训练集比例

# 设置随机种子
random.seed(SEED)
torch.manual_seed(SEED)


# 1.数据生成
# 生成样本数据，每个样本是一个5字的文本，"你"在第几位就属于第几类
def generate_samples(n_samples):
    # 常用汉字集合（简化版，实际应用中可以使用更全面的字符集）
    common_chars = "的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又么利比或但质气第向道命此变条只没结解问意建月公无系军很情者最立代想已通并提直题党程展五果料象员革位入常文总次品式活设及管特件长求老头基资边流路级少图山统接知较将组见计别她手角期根论运农指几九区强放决西被干做必战先回则任取据处队南给色光门即保治北造百规热领七海口东导器压志世金增争济阶油思术极交受联什认六共权收证改清己美再采转更单风切打白教速花带安场身车例真务具万每目至达走积示议声报斗完类八离华名确才科张信马节话米整空元况今集温传土许步群广石记需段研界拉林律叫且究观越织装影算低持音众书布复容儿须际商非验连断深难近矿千周委素技备半办青省列习响约支般史感劳便团往酸历市克何除消构府称太准精值号率族维划选标写存候毛亲快效斯院查江型眼王按格养易置派层片始却专状育厂京识适属圆包火住调满县局照参红细引听该铁价严龙飞"

    data = []

    for _ in range(n_samples):
        # 随机决定"你"字的位置(0-4)
        pos = random.randint(0, 4)
        # 生成5个随机汉字
        chars = [random.choice(common_chars) for _ in range(5)]
        # 在指定位置插入"你"字
        chars[pos] = "你"
        # 组合成字符串并添加到data中
        text = "".join(chars)
        data.append((text, pos))

    return data


# 2.词表构建与编码
def build_vocab(data):
    """
    构建词表，包含<PAD>和<UNK>两个特殊字符
    :param data: 输入数据
    :return: 词表
    """
    vocab = {'<PAD>': 0, '<UNK>': 1}
    for sent, _ in data:
        for ch in sent:
            if ch not in vocab:
                vocab[ch] = len(vocab)
    return vocab


def encode(sent, vocab, maxlen=MAXLEN):
    """
    对文本进行编码，将每个字符转换为对应的索引
    :param sent: 输入文本
    :param vocab: 词表
    :param maxlen: 最大序列长度，超过部分截断，不足部分用<PAD>填充
    :return: 编码后的文本
    """
    ids = [vocab.get(ch, 1) for ch in sent]
    ids = ids[:maxlen]
    ids += [0] * (maxlen - len(ids))
    return ids


# 3.定义DataLoader，数据加载器
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


# 4.模型定义
class TextClassifier(nn.Module):
    """
    中文关键词分类器
    架构：Embedding → RNN → LSTM → MaxPool → BN → Dropout → Linear → (CrossEntropyLoss)
    """

    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, 5)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        x = self.embedding(x)  # (B, seq_len, embed_dim)
        x, _ = self.rnn(x)  # (B, seq_len, hidden_dim)
        x, _ = self.lstm(x)  # (B, seq_len, hidden_dim)
        x = x.max(dim=1)[0]  # (B, hidden_dim)  对序列做 max pooling
        x = self.bn(x)  # (B, hidden_dim)
        x = self.dropout(x)  # (B, hidden_dim)
        x = self.fc(x)  # (B, 5)
        return x


# 5.训练和评估
def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            logits = model(X)  # 获取模型输出
            pred = logits.argmax(dim=1)  # 取概率最大的类别作为预测结果
            correct += (pred == y).sum().item()  # 计算正确预测的数量
            total += len(y)
    return correct / total


def train():
    print("生成数据集...")
    data = generate_samples(N_SAMPLES)
    print("样本:", data)
    vocab = build_vocab(data)
    print("词表:", vocab)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")

    split = int(len(data) * TRAIN_RATIO)
    train_data = data[:split]
    val_data = data[split:]

    train_loader = DataLoader(TextDataset(train_data, vocab), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(TextDataset(val_data, vocab), batch_size=BATCH_SIZE)

    model = TextClassifier(vocab_size=len(vocab))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  模型参数量：{total_params:,}\n")

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for X, y in train_loader:
            pred = model(X)
            loss = criterion(pred, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        val_acc = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")

    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")

    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = generate_samples(5)
    test_sents.append(("你好啊", 0))
    test_sents.append(("请你喝茶", 1))

    with torch.no_grad():
        for text, true_label in test_sents:
            ids = torch.tensor([encode(text, vocab)], dtype=torch.long)
            logits = model(ids)
            pred_label = logits.argmax(dim=1).item()  # 获取预测类别
            print(f"  输入文本: {text}")
            print(f"  编码后的文本: {encode(text, vocab)}")
            print(f"  真实标签: {true_label}")
            print(f"  预测标签: {pred_label}\n")


if __name__ == '__main__':
    train()
