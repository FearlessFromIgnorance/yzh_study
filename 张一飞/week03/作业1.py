import torch
import torch.nn as nn

# 数据准备
# 词表设定：'x' -> 0, '你' -> 1
sentences = ["你xxxx", "x你xxx", "xx你xx", "xxx你x", "xxxx你"]
# 对应的输入张量 (5个样本，序列长度5)
x_data = torch.tensor([
    [1, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 1, 0, 0],
    [0, 0, 0, 1, 0],
    [0, 0, 0, 0, 1]
])
# 对应的标签（"你"所在的位置: 0, 1, 2, 3, 4）
y_data = torch.tensor([0, 1, 2, 3, 4])

# 模型定义 (Embedding + RNN + Linear)
class SimpleRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.emb = nn.Embedding(num_embeddings=2, embedding_dim=4)
        self.rnn = nn.RNN(input_size=4, hidden_size=8, batch_first=True)
        self.fc = nn.Linear(in_features=8, out_features=5)

    def forward(self, x):
        x = self.emb(x)                  # 映射为词向量
        out, _ = self.rnn(x)             # RNN 处理序列
        return self.fc(out[:, -1, :])    # 取最后一个字输出的状态，进行 5 分类

model = SimpleRNN()
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()

# 训练与测试
print("训练中...")
for epoch in range(100):
    optimizer.zero_grad()
    loss = criterion(model(x_data), y_data)
    loss.backward()
    optimizer.step()

print("\n--- 测试结果 ---")
predicts = torch.argmax(model(x_data), dim=1)
for i, seq in enumerate(sentences):
    print(f"文本: '{seq}' -> 预测类别(位置): {predicts[i].item()}")