"""
    基于pytorch框架写一个训练模型:
        任务：多分类任务，找出多维向量中最大值所在顺序，即最后分类
        本次训练对5维向量进行分类
    训练基本流程：
        1. 定义模型
        2. 定义损失和优化器
        3. 进行训练，循环以下部分
            （1）前向传播
            （2）计算损失
            （3）反向传播
            （4）更新参数
        4. 用训练好的模型进行预测
"""
import torch    
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt

# 构造样本
def build_sample():
    """构造单个样本"""
    x = np.random.rand(5)
    y = np.argmax(x)
    return x, y
# 构造数据集
def build_dataset(num):
    """构造数据集"""
    X = []
    Y = []
    for i in range(num):   # 生成100个样本
        x, y = build_sample()
        X.append(x)
        Y.append(y)
    # print(X)
    # print(Y)
    return torch.FloatTensor(np.array(X)), torch.LongTensor(np.array(Y)) # 先将列表转换为numpy数组，再转化为张量
    # return torch.FloatTensor(X), torch.FloatTensor(Y) # 这种方式创建张量非常慢，先合并为为numpy数组，再进行转化

# 定义模型类
class TorchModel(nn.Module):
    """定义神经网络模型"""
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(5, 5)   # 线性层，(5, 5)第一个参数表示输入特征的维度，第二个参数表示分类个数
        # self.activation = nn.Sigmoid()  # 激活函数

    def forward(self, x):
        x = self.linear(x)
        # x = self.activation(x)
        return x


def train(model, dataloader):
    """模型训练函数"""
    # 初始化参数
    epoch_num = 500 # 训练轮数
    lr = 0.1    # 学习率

    # 2. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()   # 使用交叉熵损失函数
    optimizer = optim.Adam(model.parameters(), lr = lr)

    # 3. 模型训练
    model.train()
    log = []    # 记录每一轮的正确的和损失值，用来后续画图
    for epoch in range(epoch_num):    # 训练轮数
        epoch_loss = 0  # 每轮损失值
        for batch_x, batch_y in dataloader: # 加载一个batch的数据进行处理
            # (1) 前向传播
            y_pred = model(batch_x)
            # (2) 计算损失
            loss = criterion(y_pred, batch_y)
            # (3) 反向传播
            optimizer.zero_grad()   # 梯度清零，因为pytorch中梯度是累加的
            loss.backward()
            # (4) 更新参数
            optimizer.step()

            epoch_loss += loss.item()   # .item() 用于提取数值，不参与梯度计算
            # print(loss) # tensor(0.0013, grad_fn=<NllLossBackward0>) ,带有梯度信息的张量
        avg_loss = epoch_loss / len(dataloader)
        acc = eval(model)   # 本轮模型测试结果
        log.append([acc, avg_loss])
        
        if (epoch + 1) % 10 == 0:
            print(f"第{epoch + 1}轮 | 平均loss：{avg_loss} | 预测正确率为：{(acc * 100) : .2f}%")

    # 保存模型训练后的参数
    torch.save(model.state_dict(), "model.bin")
    print("模型参数已保存到 “model.bin” 文件中")

    # 画图
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()

    return model

def eval(model):
    """模型评估"""
    model.eval()    # 设置为评估模式
    correct = 0     # 预测正确数量
    wrong = 0   # 预测错误数量
    total = 0   # 总数量
    test_sample_num = 100

    # 生成测试集数据
    test_x, test_y = build_dataset(test_sample_num)
    # test_dataset = TensorDataset(test_x, test_y)    # 将测试集的特征与标签进行配对

    with torch.no_grad():
        y_pred = model(test_x)
        _, predicted = torch.max(y_pred, 1)
        correct = (predicted == test_y).sum().item()
        total = test_y.size(0)
    
    return correct / total

def predict(model, test_data):
    """使用训练好的模型进行预测"""
    model.eval()    # 模型设定为评估模式

    # 将测试数据转换为张量
    test_data = torch.FloatTensor(test_data)

    with torch.no_grad():
        y_pred = model(test_data)   # 预测值原始输出
        print("\n模型预测值为：")
        print(y_pred)
        probabilities = torch.softmax(y_pred, dim = 1)   # 转换为概率
        print("\n将预测值通过softmax转换为概率为：")
        print(probabilities)
        max_value, index = torch.max(probabilities, 1)  # 沿维度1（即列）方向，找最大值，返回最大值，最大值所在位置索引
        print(f"\n模型预测结果为：{index + 1}")

def main():
    """主函数"""
    print("=" * 100)
    print("多分类训练任务")
    print("=" * 100)

    # 先构造训练集样本
    print("先构造样本，特征值是五维向量，标签是最大值所在索引")
    train_sample = 1000  # 训练集样本
    batch_size = 20     # 每轮一次性处理数据大小
    print("样本生成中......")
    # 加载数据
    X, Y = build_dataset(train_sample)  # 构造样本
    # print(X)
    # print(Y + 1)
    print("训练集已构造完成！")
    print("-" * 100)
    dataset = TensorDataset(X, Y)   # 将特征值与标签配对
    dataloader = DataLoader(dataset, batch_size = batch_size, shuffle = True)   # 创建数据加载器，批量读取数据
    # for batch_x, batch_y in dataloader:   # 加载一个batch数据
    #     print(batch_x)
    #     print(batch_y + 1)

    # 1. 创建模型对象
    model = TorchModel()

    # 模型训练
    print("模型开始训练......")
    model = train(model, dataloader)
    print("模型训练完成......")
    print("-" * 100)

    # 4. 用训练好的模型进行预测
    test_data = [[0.88889086,0.15229675,0.31082123,0.03504317,0.88920843],  # 预测结果应为5
                [0.94963533,0.5524256,0.95758807,0.95520434,0.84890681],    # 正确结果应为3
                [0.90797868,0.67482528,0.13625847,0.34675372,0.19871392],   # 正确结果应为1
                [0.99349776,0.59416669,0.92579291,0.41567412,0.1358894]]    # 正确结果应为1
    print("用训练好的模型进行预测")
    predict(model, test_data)
    print("实际正确结果应为：[5, 3, 1, 1]")
    print("=" * 100)


if __name__ == "__main__":
    main()
