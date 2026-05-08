import torch
import torch.nn as nn
import jieba
import csv
import random
from collections import Counter
from encode import text_to_ids
from build_vocab import load_vocabulary
from news_classification import category_mapping
from news_classification import LSTMClassifier
MAXLEN      = 96
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

category_id2name = {v: k for k, v in category_mapping.items()}

news_text = ["这是一个测试新闻文本,包含关键词足球和篮球。",
             "京城再落重磅靴子！发改委大佬落马、远比想的可怕，圈内集体沉默",
             "扮猪吃虎？忍耐四个月，委代总统撕破伪装，率几十万大军硬刚美国",
             "被要求滚出主持界，沦为众矢之的的央视主持董倩到底做错了什么",
             "中国女排3-0横扫巴西，夺得世界杯冠军！郎平：我们是最棒的！",
             "暗黑破坏神4清算赛季4月28日 无专属剧情",
             "日前，我们获悉，长安福特新款福睿斯正式上市，售价区间为9.68-12.23万元",
             "日本放宽学校留学生就职条件 再入境仍可就职",
             "戴秉国在拉奎拉会见美国总统奥巴马中新网拉奎拉",
             "公安部出台驾照使用新规 放宽残疾人申领条件",
             "万像华语电影节闭幕 范伟马伊�黄奕助阵",
             "成龙入行48年 第100部电影看他的成“龙”之路",
             "海量存储全高清格式 索尼XR500E促销中"]


def load_random_news():
    news_data = []
    with open('random_news.csv', 'r', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            news_data.append(row)
    return news_data

def test_encoding(vocab):
    input_tensors = []
    for text in news_text:
        encoded_ids, ids_len = text_to_ids(text, "", vocab, MAXLEN)
        input_tensor = torch.tensor([encoded_ids], dtype=torch.long)  # 添加 batch 维度
        input_tensors.append(input_tensor)
    print("编码测试完成！")
    return input_tensors

def test_classification(vocab):
    test_data = test_encoding(vocab)
    model = LSTMClassifier(len(vocab))
    model.load_state_dict(torch.load('lstm_news_classifier.pth', map_location=device))
    model.to(device)
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        for i, input_tensor in enumerate(test_data):
            input_tensor = input_tensor.to(device)
            output = model(input_tensor)
            print(f"预测内容: {news_text[i]}")
            print(f"预测分类: {category_id2name[output.item()]}")
            print("-" * 50)

def main():
    vocab = load_vocabulary('vocabulary.txt')
    test_classification(vocab)
    
    
if __name__ == "__main__":
    main()