'''
文件名为sports_car_game_military_news.csv，文件内容包括新闻ID、分类名称、新闻字符串、新闻关键词四个字段，以逗号分隔。
100 民生 故事 news_story
101 文化 文化 news_culture
102 娱乐 娱乐 news_entertainment
103 体育 体育 news_sports
104 财经 财经 news_finance
106 房产 房产 news_house
107 汽车 汽车 news_car
108 教育 教育 news_edu 
109 科技 科技 news_tech
110 军事 军事 news_military
112 旅游 旅游 news_travel
113 国际 国际 news_world
114 证券 股票 stock
115 农业 三农 news_agriculture
116 电竞 游戏 news_game
'''

import csv
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from build_vocab import load_vocabulary
from tqdm import tqdm

# ─── 超参数 ────────────────────────────────────────────────
SEED        = 42
MAXLEN      = 96
EMBED_DIM   = 256
HIDDEN_DIM  = 256
LR          = 1e-3
BATCH_SIZE  = 512
EPOCHS      = 20
TRAIN_RATIO = 0.8
random.seed(SEED)
torch.manual_seed(SEED)

category_mapping = {
    "news_story": 0,
    "news_culture": 1,
    "news_entertainment": 2,
    "news_sports": 3,
    "news_finance": 4,
    "news_house": 5,
    "news_car": 6,
    "news_edu": 7,
    "news_tech": 8,
    "news_military": 9,
    "news_travel": 10,
    "news_world": 11,
    "stock": 12,
    "news_agriculture": 13,
    "news_game": 14,
    "unknown": 15
}
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的设备: {device}")

#定义dataset，从train_dataset.csv文件中载入数据，按照新闻id返回分类和新闻字符串
class NewsDataset(Dataset):
    def __init__(self,pt_dataset_path):
        data_dict = torch.load(pt_dataset_path)
        self.data = data_dict['data']      # shape: [N, MAXLEN]
        self.labels = data_dict['labels']   # shape: [N]
        
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


#构建一个lstm模型来进行新闻分类
class LSTMClassifier(torch.nn.Module):
    def __init__(self, vocab_size, embed_dim=EMBED_DIM, hidden_dim=HIDDEN_DIM, dropout=0.4):
        super(LSTMClassifier, self).__init__()
        self.embedding = torch.nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = torch.nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True, num_layers=1)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc = torch.nn.Linear(hidden_dim * 2, len(category_mapping))  # *2 for bidirectional
        self.loss = torch.nn.CrossEntropyLoss()

    def forward(self, x,y=None):
        output, _ = self.lstm(self.embedding(x))
        pooled_out = torch.max(output, dim=1)[0]  # Max pooling
        droped_out = self.dropout(pooled_out)
        out = self.fc(droped_out)
        if y is not None:
            loss = self.loss(out, y)
            return loss
        return torch.argmax(out, dim=1)


def evaluate(model, dataloader):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch)
            correct += (outputs == y_batch).sum().item()
            total += y_batch.size(0)
    return correct / total

def train(model, train_loader, val_loader, epochs=EPOCHS):
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        batch_iterator = tqdm(train_loader, 
                             desc=f"Epoch {epoch+1}/{epochs} Training", 
                             leave=True, 
                             ncols=100) # ncols 控制进度条长度
        for X_batch, y_batch in batch_iterator:
            optimizer.zero_grad()
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            loss = model(X_batch, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            batch_iterator.set_postfix({
                'Batch Loss': f'{loss.item():.4f}'
            })
        val_acc = evaluate(model, val_loader)
        avg_loss = total_loss / len(train_loader)
        tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.4f}")
        tqdm.write(f"Validation Accuracy: {val_acc:.4f}")

def main():
    vocab = load_vocabulary("vocabulary.txt")
    dataset = NewsDataset('processed_news_dataset.pt')
    train_size = int(TRAIN_RATIO * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
        )
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    vocab_size = len(vocab)
    print(f"Vocabulary size: {vocab_size}")
    print(f"Number of training samples: {len(train_dataset)}, Number of validation samples: {len(val_dataset)}")
    model = LSTMClassifier(vocab_size)
    
    model = model.to(device)
    
    train(model, train_loader, val_loader)
    #保存模型
    torch.save(model.state_dict(), "lstm_news_classifier.pth")
    
if __name__ == "__main__":
    main()
