import csv
import jieba
import torch
import json
from collections import Counter
from build_vocab import load_vocabulary

# 配置参数
CSV_FILE = 'news.csv'
VOCAB_PATH = 'vocabulary.txt' # 假设你之前保存的是 JSON 格式
OUTPUT_FILE = 'processed_news_dataset.pt' # 保存的二进制文件名
MAXLEN = 96
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


def text_to_ids(text, keywords, vocab, maxlen=MAXLEN):
    """核心编码函数"""
    # 1. 分词
    tokens = jieba.lcut(text)
    # 2. 转 ID (OOV 用 1, PAD 用 0)
    ids = [vocab.get(t, 1) for t in tokens]
    
    # 3. 处理关键词 (如果有)
    if keywords.strip():
        kw_tokens = [k.strip() for k in keywords.split(',')]
        kw_ids = [vocab.get(k, 1) for k in kw_tokens]
        ids = ids + kw_ids # 拼接
    
    # 4. 截断或填充,统计最大长度并返回
    ids_len = len(ids)
    if len(ids) < maxlen:
        ids += [0] * (maxlen - len(ids))
    else:
        ids = ids[:maxlen]
    
    return ids, ids_len

def main():
    # 1. 加载词表
    print("正在加载词表...")
    vocab = load_vocabulary(VOCAB_PATH)
    
    # 2. 准备数据列表
    print("正在读取并处理 CSV 数据...")
    processed_data = []
    max_len = 0
    with open(CSV_FILE, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        next(reader) # 跳过表头
        
        for row in reader:
            news_id, category_name, news_string, news_keywords = row
            
            # 编码文本
            encoded_ids, ids_len = text_to_ids(news_string, news_keywords, vocab, MAXLEN)
            max_len = max(max_len, ids_len)
            
            # 处理标签 (这里需要你定义的 category_mapping)
            category_id = category_mapping.get(category_name, category_mapping["unknown"]) # 15 是 unknown
            
            # 转成 Tensor
            input_tensor = torch.tensor(encoded_ids, dtype=torch.long)
            label_tensor = torch.tensor(category_id, dtype=torch.long)
            
            processed_data.append((input_tensor, label_tensor))
    
    # 3. 保存所有数据
    # 将所有的 Tensor 拼成一个巨大的 Tensor (或者保存为列表)
    # 这里我们保存为一个包含两个 Tensor 的字典：一个是数据，一个是标签
    # 也可以直接保存为 List，取决于数据量大小
    
    # 方案 A：如果数据量很大 (几十 GB)，保存为 List (兼容性好)
    # torch.save(processed_data, OUTPUT_FILE)
    
    # 方案 B：如果数据量适中 (几百 MB)，拼成 Tensor (加载速度最快)
    data_tensors = torch.stack([item[0] for item in processed_data])
    label_tensors = torch.stack([item[1] for item in processed_data])
    
    torch.save({
        'data': data_tensors,
        'labels': label_tensors
    }, OUTPUT_FILE)
    
    print(f"预处理完成！共处理 {len(processed_data)} 条数据。")
    print(f"已保存至 {OUTPUT_FILE}")
    print(f"最大文本长度 (含关键词): {max_len}")

if __name__ == "__main__":
    main()