import torch
import torch.nn as nn
import numpy as np
import random
from torch.utils.data import Dataset, DataLoader

from 训练.GradientDescent import batch_size
#对一个任意包含“你”字的五个字的文本，“你”在第几位，就属于第几类样本
#例如：“你是一个好学生” 属于第1类样本
#例如：“你是一个好学生” 属于第2类样本
#例如：“你是一个好学生” 属于第3类样本
#例如：“你是一个好学生” 属于第4类样本
#例如：“你是一个好学生” 属于第5类样本

#超参数
SEED = 42   #随机种子
N_SAMPLES = 4000 #总样本数量，生成4000个样本
MAXLEN = 5 #文本截取最大序列长度，每个样本的序列长度不超过32
EMDED_DIM= 64 #词向量维度，每个字转为64维向量
HIDDEN_DIM=64 #隐藏层维度，RNN隐藏层的维度
LR = 1e-3 #学习率
BATCH_SIZE = 32 #批次大小，每次训练时使用的样本数量，32个样本'
EPOCHS = 20 #训练轮数，100轮
TRAIN_RATIO = 0.8 #训练集比例，80% ,20%测试

# 常用汉字（用来随机生成句子）
CHARS = [c for c in "的一是了我不人在他有这个上们来到时大地为子中你说生国年着就那和要她出也得都看"]

#生成带你的句子
def make_sent():
    #随机生成一个位置
    pos = random.randint(0,4)
    sent  = random.choices(CHARS,k=5)
    sent[pos] = "你"
    # return sent,pos
    return "".join(sent),pos
#组装数据集
def build_dataset(n=N_SAMPLES):
    dataset = []
    for _ in range(n):
        sent,pos = make_sent()
        dataset.append((sent,pos))
    return dataset

#词表和编码
def build_vocab(data):
    #构建词表
    vocab = {"<PAD>":0, "<UNK>":1}
    for sent,pos in data:
        for c in sent:
            if c not in vocab:
                vocab[c] = len(vocab)
    return vocab

#编码数据集
def encode(sent,vocab,max_len=MAXLEN):
    ids = [vocab.get(ch,1) for ch in sent]
    ids = ids[:max_len]
    ids += [0]*(max_len-len(ids))
    return ids

#包装数据集
class TextDataset(Dataset):
    def __init__(self,data,vocab):
        self.x = [encode(s,vocab) for s,_ in data]
        self.y = [lb for _,lb in data]

    def __len__(self):

        return len(self.x)

    def __getitem__(self,idx):

        return (torch.tensor(self.x[idx],dtype=torch.long),torch.tensor(self.y[idx],dtype=torch.long))

#模型定义
class TextRNN(nn.Module):
    def __init__(self,vocab_size,embedding_size,hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size,embedding_size,padding_idx=0)
        self.rnn = nn.RNN(embedding_size,hidden_size,batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size,5)

    def forward(self,x):
        x = self.embedding(x)
        x,_ = self.rnn(x)
        x = x.max(dim=1)[0]
        x = self.dropout(x)
        x = self.fc(x)

        return x

def evaluate(model,loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for X,Y in loader:
            out = model(X)
            #argmax(dim=1) = 看哪个分数最大 → 就是预测的类别
            pred = out.argmax(dim=1)
            correct += (pred==Y).sum().item()
            total += Y.size(0)
    return correct/total

def train ():
    #造数据+构建词表
    print("生成数据集...")
    data = build_dataset()
    vocab = build_vocab(data)
    print(f"  样本数：{len(data)}，词表大小：{len(vocab)}")
    #划分数据集和测试集
    split = int(len(data)*TRAIN_RATIO)
    #训练集80%，测试集20%
    train_data = data[:split]
    val_data = data[split:]
    #构建Dataset+DataLodar
    #训练
    train_ds = TextDataset(train_data,vocab)
    val_ds = TextDataset(val_data,vocab)

    #训练集 shuffle=True → 每轮打乱顺序
    train_loader = DataLoader(train_ds,BATCH_SIZE,shuffle=True)
    val_loader = DataLoader(val_ds,BATCH_SIZE)

    #创建模型、损失函数、优化器
    model = TextRNN(len(vocab),EMDED_DIM,HIDDEN_DIM)
    #二分类问题，用交叉熵损失函数
    criterion = nn.CrossEntropyLoss()
    #优化器
    optimizer = torch.optim.Adam(model.parameters(),lr=LR)
    #模型参数总数
    total_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数总数：{total_params}")
    for epoch in range(1,EPOCHS+1):
        model.train()
        total_loss = 0.0
        for X,Y in train_loader:
            #前向传播
            out = model(X)
            #计算损失
            loss = criterion(out,Y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_acc  = evaluate(model, val_loader)
        print(f"Epoch {epoch:2d}/{EPOCHS}  loss={avg_loss:.4f}  val_acc={val_acc:.4f}")
    print(f"\n最终验证准确率：{evaluate(model, val_loader):.4f}")
    print("\n--- 推理示例 ---")
    model.eval()
    test_sents = [
        '你好',
        '你今天怎么样',
        '我爱你啊',
        '你你你你',
    ]
    with torch.no_grad():
        for sent in test_sents:
            ids   = torch.tensor([encode(sent, vocab)], dtype=torch.long)
            out = model(ids)
            #argmax(dim=1) = 看哪个分数最大 → 就是预测的类别
            pred = out.argmax(dim=1).item()

            print(f"  [{out}] 预测位置：{pred}测试句子：{sent}")

if __name__ == '__main__':
    train()
