import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os

# 定义模型保存路径
MODEL_PATH = os.path.join(os.path.dirname(__file__), "torch_model.pth")

# 测试的向量维度
INPUT_DIM = 10

class TorchModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(TorchModel, self).__init__()
        # 定义一个线性层
        self.linear = nn.Linear(input_size, output_size)
        # loss 函数(交叉熵损失)
        self.loss_fn = nn.CrossEntropyLoss()

    # 当输入真实标签，返回loss值；无真实标签，返回预测值
    def forward(self, x, y=None):
        x = self.linear(x)
        if y is not None:
            return self.loss_fn(x, y)
        return torch.softmax(x, dim=-1)

# 生成一个样本, 样本的生成方法，代表了我们要学习的规律
# 随机生成一个 INPUT_DIM 维向量，哪个数最大，这个向量就属于第几类
def build_sample():
    x = np.random.rand(INPUT_DIM)
    y = np.argmax(x)
    return x, y

# 随机生成一批样本
# 正负样本均匀生成
def build_dataset(num_samples):
    X = []
    Y = []
    for _ in range(num_samples):
        x, y = build_sample()
        X.append(x)
        Y.append(y)

    # print("X:", X)
    # print("np.array(X):", np.array(X))
    # print("torch.FloatTensor(X):", torch.FloatTensor(np.array(X)))
    return np.array(X), np.array(Y)

# 测试代码
# 用来测试每轮模型的准确率
def evaluate(model):
    model.eval() # 切换到评估模式
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    correct, wrong = 0, 0
    with torch.no_grad(): # 评估时不需要计算梯度
        y_pred = model(torch.FloatTensor(x)) # 模型预测
        for y_t, y_p in zip(y, y_pred):
            if y_t == torch.argmax(y_p).item(): # 预测正确
                correct += 1
            else: # 预测错误
                wrong += 1
    print("正确预测: %d, 错误预测: %d, 正确率: %.4f" % (correct, wrong, correct / (correct + wrong)))
    return correct / (correct + wrong)


def main():
    # 配置参数
    epoch_num = 100 # 训练轮数
    batch_size = 64 # 每次训练使用的样本数量
    train_sample_num = 10000 # 每轮训练总共训练的样本总数
    input_size = INPUT_DIM # 输入特征的维度
    output_size = INPUT_DIM # 输出类别的数量
    lr = 0.01 # 学习率

    # 建立模型
    main_model = TorchModel(input_size, output_size)
    # 定义优化器
    optimizer = torch.optim.Adam(main_model.parameters(), lr=lr)

    log = []

    # 创建训练集
    X, Y = build_dataset(train_sample_num)
    # 训练模型
    for epoch in range(epoch_num):
        main_model.train()
        watch_loss = [] # 记录每轮训练的loss值，观察训练过程

        for batch_index in range(train_sample_num // batch_size):
            # 获取当前批次的样本
            start_index = batch_index * batch_size
            end_index = start_index + batch_size
            x_batch = torch.FloatTensor(X[start_index:end_index]) # 转换为FloatTensor
            y_batch = torch.LongTensor(Y[start_index:end_index]) # 转换为LongTensor，适用于交叉熵损失函数的标签格式

            # 前向传播计算损失
            loss = main_model(x_batch, y_batch)

            # 反向传播和优化
            loss.backward() # 计算梯度
            optimizer.step() # 更新参数
            optimizer.zero_grad() # 清零梯度，准备下一次迭代

            watch_loss.append(loss.item()) # 记录当前批次的loss值
        print(f'Epoch {epoch+1}/{epoch_num}, Loss: {np.mean(watch_loss):.4f}')
        acc = evaluate(main_model) # 每轮训练结束后评估模型准确率
        log.append((epoch+1, np.mean(watch_loss), acc)) # 记录每轮的epoch、loss和准确率

    # 保存模型参数
    torch.save(main_model.state_dict(), MODEL_PATH)

    # 画图观察训练结果
    epochs = [item[0] for item in log]
    losses = [item[1] for item in log]
    accuracies = [item[2] for item in log]

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(epochs, accuracies)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Training Accuracy') # 训练准确率

    plt.tight_layout()
    plt.show()

# 训练结束后，加载模型参数进行测试
def test(model_path, input_vec):
    # 创建模型实例
    model = TorchModel(input_size=INPUT_DIM, output_size=INPUT_DIM)
    # 加载训练好的模型参数
    model.load_state_dict(torch.load(model_path))
    print(model.state_dict())

    model.eval() # 切换到评估模式
    with torch.no_grad(): # 测试时不需要计算梯度
        result = model.forward(torch.FloatTensor(input_vec))

    for vec, res in zip(input_vec, result):
        print("✅预测成功" if np.argmax(vec) == torch.argmax(res).item() else "❌预测失败")
        print(f'实际类别: {np.argmax(vec)}, 输入: {np.round(vec, 4)}')
        print(f'预测类别: {torch.argmax(res).item()}, 预测概率分布: {res.numpy()}')
        print("-" * 50)

if __name__ == "__main__":
    main()

    # 根据 INPUT_DIM 构建测试向量，测试模型的预测结果
    test_vec = []
    for _ in range(10): # 生成10个测试向量
        vec = np.random.rand(INPUT_DIM)
        test_vec.append(vec.tolist())
    test(MODEL_PATH, test_vec)
