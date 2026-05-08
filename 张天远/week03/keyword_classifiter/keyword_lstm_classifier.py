import csv
import random
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from collections import Counter
from tqdm import tqdm
import matplotlib.pyplot as plt
from generate_data import generate_all_data
KEYWORDS = ["你好", "再见", "谢谢", "抱歉"]
KEYWORD_IDS = {"你好": 0, "再见": 1, "谢谢": 2, "抱歉": 3}
NO_KEYWORD_LABEL = 4

SEED = 42
MAX_LEN = 64
EMBED_DIM = 64
HIDDEN_DIM = 256
LR = 1e-3
BATCH_SIZE = 128
EPOCHS = 100
SEARCH_LEN = 5
random.seed(SEED)
torch.manual_seed(SEED)

tran_file ='./train.csv'
test_file = './test.csv'
val_file = './val.csv'
vocab_file = './cnews.vocab.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"当前使用的设备: {device}")

def load_vocab(path):
    vocab = {"<PAD>": 0}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            char = line.strip()
            if char and char != "<PAD>":
                vocab[char] = len(vocab)
    vocab["<UNK>"] = len(vocab)
    return vocab


def text_to_ids(text, vocab, max_len):
    unk_id = vocab["<UNK>"]
    ids = [vocab.get(char, unk_id) for char in text[:max_len]]
    if len(ids) < max_len:
        ids += [vocab["<PAD>"]] * (max_len - len(ids))
    return ids


def check_keyword_in_text(text, search_len=5):
    head = text[:search_len]
    tail = text[-search_len:] if len(text) >= search_len else text
    combined = head + tail
    for kw in KEYWORDS:
        if kw in combined:
            return KEYWORD_IDS[kw]
    return NO_KEYWORD_LABEL


class TextDataset(Dataset):
    def __init__(self, csv_path, vocab, max_len):
        self.data = []
        self.max_len = max_len
        self.vocab = vocab
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                text = row["text"]
                label = int(row["label"])
                ids = text_to_ids(text, vocab, max_len)
                self.data.append((ids, label))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        ids, label = self.data[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(
            label, dtype=torch.long
        )


class TextClassifier(nn.Module):
    def __init__(self, vocab_size, num_classes,type,embed_dim = EMBED_DIM, hidden_dim = HIDDEN_DIM, dropout = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        if type == "lstm":
            self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(embed_dim,hidden_dim,batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.drop = nn.Dropout(dropout)
        self.loss = nn.CrossEntropyLoss()

    def forward(self, x, y = None):
        embed = self.embedding(x)
        # h,_ = self.lstm(embed)
        output, h_n = self.rnn(embed)
        #使用maxpooling，无视位置信息，rnn和lstm都可以有效检测出关键词
        # maxpooling_out = torch.max(output,dim =1)[0] 
        # dropout_out = self.drop(maxpooling_out)
        # last_out = h[:, -1, :]  #取最后一次状态作为输出
        # 输出整个序列的隐藏状态
        
        
        # 不使用pooling 取最后一个时间步的隐状态作为句子的特征，lstm明显胜出，但在更长文本中效果也不好
        # 对于 RNN/LSTM 单层，h_n[-1] 就是最后的输出
        last_hidden = output[:,-1,:]
        dropout_out = self.drop(last_hidden)
        
        
        pred = self.fc(dropout_out)
        if y is not None:
            loss = self.loss(pred,y)
            return loss
        else:
            return torch.argmax(pred,dim=1)
            

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

def train_mode(vocab):
    print(f"--- 开始实验：序列长度 = {MAX_LEN} ---")
    print(f"Vocab size: {len(vocab)}")
    num_classes = len(KEYWORDS) + 1
    model_rnn = TextClassifier(len(vocab),num_classes,"rnn")
    model_lstm = TextClassifier(len(vocab),num_classes,"lstm")
    model_rnn.to(device)
    model_lstm.to(device)
    
    train_dataset = TextDataset(tran_file, vocab, MAX_LEN)
    val_dataset = TextDataset(val_file, vocab, MAX_LEN)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

    
    
    optimizer_rnn = torch.optim.Adam(model_rnn.parameters(), lr=LR)
    optimizer_lstm = torch.optim.Adam(model_lstm.parameters(), lr=LR)
    
   
    rnn_losses = []
    lstm_losses = []
    
    rnn_acc = []
    lstm_acc =[]
    for epoch in range(EPOCHS):
        # --- 训练 RNN ---
        model_rnn.train()
        total_loss_rnn = 0
        for inputs, labels in train_loader:
            optimizer_rnn.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss =model_rnn(inputs, labels)
            loss.backward()
            optimizer_rnn.step()
            total_loss_rnn += loss.item()
        rnn_val_acc = evaluate(model_rnn, val_loader)
        rnn_losses.append(total_loss_rnn / len(train_loader))
        rnn_acc.append(rnn_val_acc)
        
        # --- 训练 LSTM ---
        model_lstm.train()
        total_loss_lstm = 0
        for inputs, labels in train_loader:
            optimizer_lstm.zero_grad()
            inputs = inputs.to(device)
            labels = labels.to(device)
            loss = model_lstm(inputs, labels)
            loss.backward()
            optimizer_lstm.step()
            total_loss_lstm += loss.item()
        lstm_val_acc = evaluate(model_lstm, val_loader)
        lstm_losses.append(total_loss_lstm / len(train_loader))
        lstm_acc.append(lstm_val_acc)
        # if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}: RNN Loss: {rnn_losses[-1]:.4f} | LSTM Loss: {lstm_losses[-1]:.4f}")
        print(f"Epoch {epoch+1}: RNN acc: {rnn_val_acc:.4f} | LSTM acc: {lstm_val_acc:.4f}")
    
    # --- 绘图 ---
    plt.figure(figsize=(10, 5))
    plt.plot(rnn_losses, label='RNN Loss', color='red', alpha=0.7)
    plt.plot(lstm_losses, label='LSTM Loss', color='green', alpha=0.7)
    plt.title(f'Training Loss Comparison (Sequence Length = {MAX_LEN})')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
def main():
    generate_all_data(MAX_LEN)
    vocab = load_vocab(vocab_file)
    train_mode(vocab)   
    
    
if __name__ == "__main__":
    main()
