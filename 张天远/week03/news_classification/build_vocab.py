import csv
import re
import pickle
import jieba
from collections import Counter


def tokenize(text):
    if not isinstance(text, str) or text == "":
        return []
    text = re.sub(r"[^\w\u4e00-\u9fff]", " ", text)
    tokens = jieba.lcut(text)
    return [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]


def load_csv(filepath):
    data = []
    with open(filepath, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        print(f"Header: {header}")
        for row in reader:
            if len(row) >= 4:
                news_id = row[0]
                category = row[1]
                news_text = row[2]
                keywords = row[3] if len(row) > 3 else ""
                data.append(
                    {
                        "news_id": news_id,
                        "category": category,
                        "news_text": news_text,
                        "keywords": keywords,
                    }
                )
    return data


def tokenize_keywords(keywords):
    if not keywords or not isinstance(keywords, str):
        return []
    tokens = [t.strip() for t in keywords.split(",") if t.strip()]
    return tokens


def build_vocabulary(data):
    word_counter = Counter()
    for item in data:
        text_tokens = tokenize(item["news_text"])
        keyword_tokens = tokenize_keywords(item["keywords"])
        all_tokens = text_tokens + keyword_tokens
        word_counter.update(all_tokens)

    vocabulary = {
        word: idx + 2 for idx, (word, count) in enumerate(word_counter.most_common())
    }
    vocabulary["<PAD>"] = 0
    vocabulary["<UNK>"] = 1

    return vocabulary, word_counter


# 将词表保存到文本文件，每行一个词
def save_vocabulary(vocabulary, output_path):
    with open(output_path, "wb") as f:
        for word in sorted(vocabulary.keys(), key=lambda w: vocabulary[w]):
            f.write(f"{word}\n".encode("utf-8"))
    print(f"Vocabulary saved to {output_path}")


def main():
    filepath = "news.csv"
    output_path = "vocabulary.txt"

    print("Loading CSV data...")
    data = load_csv(filepath)
    print(f"Loaded {len(data)} records")

    categories = set(item["category"] for item in data)
    print(f"Categories: {sorted(categories)}")

    print("Building vocabulary...")
    vocabulary, word_counter = build_vocabulary(data)
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Top 20 words: {word_counter.most_common(20)}")

    save_vocabulary(vocabulary, output_path)


# 构建一个函数载入词表文件，返回一个词到id的字典
def load_vocabulary(vocab_path):
    vocab = {}
    with open(vocab_path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            word = line.strip()
            vocab[word] = idx
    return vocab


if __name__ == "__main__":
    main()
