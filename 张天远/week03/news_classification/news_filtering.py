""""
数据格式 6552431613437805063_!_102_!_news_entertainment_!_谢娜为李浩菲澄清网络谣言，之后她的两个行为给自己加分_!_佟丽娅,网络谣言,快乐大本营,李浩菲,谢娜,观众们
每行为一条数据，以_!_分割的个字段，从前往后分别是 新闻ID，分类code（见下文），分类名称（见下文），新闻字符串（仅含标题），新闻关键词
分类code与名称：
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

我请读取txt数据文件并进行分类，统计每个分类的新闻数量，并输出结果。
"""
import csv
from collections import defaultdict

# 读取txt数据文件
with open('toutiao_cat_data.txt', 'r', encoding='utf-8') as f:
    lines = f.readlines()

# 统计每个分类的新闻数量，统计新闻字符串的最大长度
category_count = defaultdict(int)

max_news_length = 0
max_title_length = 0
for line in lines:
    fields = line.strip().split('_!_')
    if len(fields) >= 4:
        category = fields[2]
        category_count[category] += 1
        news_string = fields[3]
        max_news_length = max(max_news_length, len(news_string))

# 输出结果
for category, count in category_count.items():
    print(f"{category}: {count}")
print(f"最长的新闻字符串长度: {max_news_length}")

#将其中属于体育、汽车、游戏、军事这四个分类的新闻单独统计出来，并单独保存成一个新的csv文件，文件名为sports_car_game_military_news.csv，文件内容包括新闻ID、分类名称、新闻字符串、新闻关键词四个字段，以逗号分隔。

# 定义需要单独统计的分类
# categories_to_save = {'news_sports', 'news_car', 'news_game', 'news_military'}
# 打开新的csv文件进行写入
with open('news.csv', 'w', encoding='utf-8', newline='') as csvfile:
    writer = csv.writer(csvfile)
    # 写入表头
    writer.writerow(['新闻ID', '分类名称', '新闻字符串', '新闻关键词'])
    # 遍历原始数据，筛选出需要的分类并写入新的csv文件
    for line in lines:
        fields = line.strip().split('_!_')
        if len(fields) >= 4:
            category_code = fields[1]
            category_name = fields[2]
            news_string = fields[3]
            news_keywords = fields[4] if len(fields) > 4 else ''
            # if category_name in categories_to_save:
            writer.writerow([fields[0], category_name, news_string, news_keywords])
print("保存文件完成！")


    

        