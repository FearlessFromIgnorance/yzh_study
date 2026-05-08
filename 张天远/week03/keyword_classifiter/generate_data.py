import random
import csv
from collections import Counter
KEYWORDS = ["你好", "再见", "谢谢", "抱歉"]
KEYWORD_IDS = {"你好": 0, "再见": 1, "谢谢": 2, "抱歉": 3}


SEARCH_LEN = 5
SAMPLE_NUM = 10000
TRAIN_RATIO = 0.8
POSTIVE_RATIO = 0.8
TEST_NUM = 2000
with open("cnews.vocab.txt", "r", encoding="utf-8") as f:
    vocab = [line.strip() for line in f if line.strip()]

vocab_chars = [c for c in vocab if len(c) == 1 and c != "<PAD>"]

KEYWORD_CHARS = set()
for kw in KEYWORDS:
    for c in kw:
        KEYWORD_CHARS.add(c)
        

def random_text_from_vocab(length):
    # 从词表中剔除关键词字符，只保留“安全”的字符，防止干扰
    safe_chars = [c for c in vocab_chars if c not in KEYWORD_CHARS]
    if not safe_chars: # 防止列表为空
        safe_chars = vocab_chars
    return "".join(random.choices(safe_chars, k=length))


def generate_sample(keyword_id, keyword, position,max_len,min_len):
    text_len = random.randint(min_len, max_len)
    body_len = text_len - len(keyword)

    if position == "head":
        text = keyword + random_text_from_vocab(body_len)
    else:
        text = random_text_from_vocab(body_len) + keyword

    return text, keyword_id


def generate_negative_sample(max_len,min_len):
    text_len = random.randint(min_len, max_len)
    return random_text_from_vocab(text_len), len(KEYWORD_IDS)


def generate_dataset(num_samples,max_len,min_len,train_ratio=TRAIN_RATIO):
    data = []
    for _ in range(num_samples):
        if random.random() < POSTIVE_RATIO:
            keyword = random.choice(KEYWORDS)
            keyword_id = KEYWORD_IDS[keyword]
            # position = random.choice(["head", "tail"])
            position = "head"
            text, label = generate_sample(keyword_id, keyword, position,max_len,min_len)
        else:
            text, label = generate_negative_sample(max_len,min_len)
        data.append((text, label))

    random.shuffle(data)
    split_idx = int(num_samples * train_ratio)
    return data[:split_idx], data[split_idx:]

def generate_all_data(max_len,sample_num = SAMPLE_NUM):
    random.seed(42)
    min_len = max(max_len//4,10)
    train_data, val_data = generate_dataset(sample_num,max_len,min_len)
    # test_data,_ = generate_dataset(TEST_NUM,1)
    with open("train.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text"])
        for text, label in train_data:
            writer.writerow([label, text])

    with open("val.csv", "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "text"])
        for text, label in val_data:
            writer.writerow([label, text])

    # with open("test.csv", "w", encoding="utf-8", newline="") as f:
    #     writer = csv.writer(f)
    #     writer.writerow(["label", "text"])
    #     for text, label in test_data:
    #         writer.writerow([label, text])
    
    print(f"Train: {len(train_data)}, Test: {len(val_data)}")
    print("\nLabel mapping:", KEYWORD_IDS)
    print("\nSamples:")
    for i in range(5):
        text, label = train_data[i]
        preview = text[:30] + "..." if len(text) > 30 else text
        print(f"  Label: {label}, Text: {preview}")
    
    labels = [label for _, label in train_data]
    print("Label Distribution:", Counter(labels))

if __name__ == "__main__":
    generate_all_data(128,SAMPLE_NUM)
