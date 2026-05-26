# 单向 Transformer 语言模型（作业）

本项目实现基于 **Decoder-only Transformer**（因果注意力）的**字符级**单向语言模型，并完成训练与文本生成。

## 环境

```bash
cd E:/Work/week5
pip install -r requirements.txt
```

## 数据

默认使用 `data/sample_zh.txt`。可替换为你的语料（UTF-8 纯文本）。

## 训练

```bash
python train.py --data data/sample_zh.txt --epochs 150 --batch_size 32
```

权重与词典元数据保存在 `checkpoints/lm.pt` 与 `checkpoints/meta.json`。

主要参数：`--block_size`（上下文长度）、`--n_layers`、`--n_heads`、`--d_model`、`--d_ff`。

## 生成

```bash
python generate.py --prompt "机器学习" --max_new_tokens 100 --temperature 0.9
```

`temperature` 越大采样越随机；越小越尖锐（接近贪心）。

## 结构说明

- `model.py`：多头因果自注意力、`DecoderBlock`、`DecoderOnlyLM`（embedding + 位置编码 + LayerNorm）。
- `train.py`：字符词典、数据集、交叉熵与 AdamW。
- `generate.py`：加载 checkpoint，自回归 multinomial 采样扩展序列。
