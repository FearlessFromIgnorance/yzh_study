# coding:utf8
import os

from sklearn.model_selection import learning_curve

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

#nn.Module 是神经网络模型的父类
"""

多分类任务的训练:一个随机向量，哪一维数字最大就属于第几类

"""
#模型名字
class MultiClassficationModel(nn.Module):
    #self初始化函数，input_size 是输入向量维度
    def __init__(self,input_size,output_size):
        #启动父类nn。module功能，必写
        super(MultiClassficationModel,self).__init__()
        #线性层 nn.Linear（输入维度，输出维度） 因为我们是二分类任务，输出是一个数字
        #nn.Linear里面是数学公式，代表y=w1x1+...+w5x5+b
        #多分类任务，输出结果和输入内容向量一样多
        self.linear = nn.Linear(input_size,output_size)
        #数字压缩工具 最后结果是0-1的数，0=0.5
        # self.activation = torch.sigmoid
        #损失函数 nn提供的工具
        # loss函数采用交叉熵损失 nn.CrossEntropyLoss()
        #预测一个具体数字（房价、分数、大小）👉 MSELoss
        #分类别（3 分类、5 分类、10 分类）👉 CrossEntropyLoss
        self.loss = nn.functional.cross_entropy
    def forward(self,x,y=None):
        y_pred = self.linear(x)
        #传入y的时候就是训练的过程，来修改线性层的公式，
        # 不传的时候是使用的时候，根据公式算出来预测的结果
        if y is not None:
            return self.loss(y_pred,y)
        else:
            #dim=1对第一行取最大值，返回时是下标
            return torch.softmax(y_pred,dim=1)# 输出预测结果

#打造数据 这个是打造一个5维度的向量，完事判断当0》4时候标签为1反之为0
def build_sample():
    x = np.random.random(5)
    #获取最大值的下表
    max_index = np.argmax(x)
    #返回打造的5维向量和最大值的下标
    return x,max_index

#对打造向量处理最后结果
def build_dataset(total_sample_num):
    #用来存生成的数据，和结果
    X=[]
    Y=[]
    #开始跑数据
    for i in range(total_sample_num):
        x,y = build_sample()
        X.append(x)
        Y.append(y)
    #将x和ylist转换为Tensor张量数据才能处理
    return torch.FloatTensor(X),torch.LongTensor(Y)

#训练函数
def evaluate(model):
    #eval = evaluation 评估
    #不写这段话会随即遗忘数据，结果会乱飘、不稳定、不准确！写了这个每次预测结果都稳定、准确
    #默认是model.train()
    model.eval()
    #造数据
    test_sample_num = 100
    x, y = build_dataset(test_sample_num)
    #对了几个，错了几个
    correct, wrong = 0, 0
    #和model.eval()配套，用来说明是预测数据减少内存
    with torch.no_grad():
        #预测模型 model.forward(x) 调用模型算值
        #第一种写法
        # y_pred = torch.argmax(model(x), dim=1)
        #第二种写法
        y_pred=model(x)
        #y_pred是我预测值，y是最终结果
        for y_p,y_t in zip(y_pred,y):
            #如果预测y_p=y_下标相同
            if torch.argmax(y_p)==int(y_t):
                correct += 1
            else:
                wrong += 1
    print("正确预测个数：%d, 正确率：%f" % (correct, correct / (correct + wrong)))
    #训练出来的正确率
    return correct / (correct + wrong)

def main():
    #配置参数
    epoch_num = 20
    batch_size = 20
    train_sample = 5000
    input_size = 5
    output_size = 5
    learning_rate = 0.01
    #     # 配置参数
    #     epoch_num = 20  # 训练轮数
    #     batch_size = 20  # 每次训练样本个数
    #     train_sample = 5000  # 每轮训练总共训练的样本总数
    #     input_size = 5  # 输入向量维度
    #     learning_rate = 0.01  # 学习率
    #建立模型
    model = MultiClassficationModel(input_size,output_size)
    #选择优化器 Adam修改model模型的权重代码
    #optim.step()  就在这里修改 model.parameters() 里的所有参数
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    log = []
    #创建训练集了
    train_x,train_y = build_dataset(train_sample)
    #训练过程 训练轮数20轮 5000个数据
    for epoch in range(epoch_num):
        #修改成训练模式
        model.train()
        #装每一组训练完事的loss值
        watch_loss = []
        for batch_index in range(train_sample//batch_size):
            #取出一个batch数据作为输入   train_x[0:20]  train_y[0:20] train_x[20:40]  train_y[20:40]
            x = train_x[batch_index*batch_size:(batch_index+1)*batch_size]
            y = train_y[batch_index*batch_size:(batch_index+1)*batch_size]
            loss = model(x, y) #model(x, y)=model.forward(x,y)
            #根据错误，反向计算出：模型里每个权重该怎么调整（梯度）
            loss.backward() #计算梯度
            optim.step()  # 更新权重
            optim.zero_grad()  # 梯度归零
            #.item()是把张量换成能存的值
            watch_loss.append(loss.item())
            #求本轮所有 batch 的 loss 的【平均值】
            print("=========\n第%d轮平均loss:%f" % (epoch + 1, np.mean(watch_loss)))
            acc = evaluate(model)
            log.append([acc, float(np.mean(watch_loss))])
            # 保存模型
    torch.save(model.state_dict(), "model.bin")
    # 画图
    print(log)
    plt.plot(range(len(log)), [l[0] for l in log], label="acc")  # 画acc曲线
    plt.plot(range(len(log)), [l[1] for l in log], label="loss")  # 画loss曲线
    plt.legend()
    plt.show()
    return

# 使用训练好的模型做预测
def predict(model_path, input_vec):
    input_size = 5
    output_size = 5
    model = MultiClassficationModel(input_size,output_size)
    model.load_state_dict(torch.load(model_path))  # 加载训练好的权重
    print(model.state_dict())

    model.eval()  # 测试模式
    with torch.no_grad():  # 不计算梯度
        result = model.forward(torch.FloatTensor(input_vec))  # 模型预测
    for vec, res in zip(input_vec, result):
        print("输入：%s, 预测类别：%s, 概率值：%s" % (vec, torch.argmax(res), res))  # 打印结果


if __name__ == "__main__":
    main()
    test_vec = [[0.47889086, 0.15229675, 0.31082123, 0.03504317, 0.18920843],
                [0.4963533, 0.5524256, 0.95758807, 0.65520434, 0.84890681],
                [0.48797868, 0.67482528, 0.13625847, 0.34675372, 0.09871392],
                [0.49349776, 0.59416669, 0.92579291, 0.41567412, 0.7358894]]
    predict("model.pt", test_vec)
# "model.bin", test_vec)