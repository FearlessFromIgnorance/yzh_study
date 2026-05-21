import torch
import torch.nn as nn
import math
import numpy as np


class PositionalEncoding(nn.Module):
    def __init__(self, hidden_size, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, hidden_size)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, hidden_size, 2).float() * (-math.log(10000.0) / hidden_size))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states, attention_mask=None):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = qk / math.sqrt(self.attention_head_size)

        if attention_mask is not None:
            qk = qk + attention_mask

        qk = torch.softmax(qk, dim=-1)
        qk = self.dropout(qk)

        qkv = torch.matmul(qk, v)
        qkv = qkv.permute(0, 2, 1, 3).contiguous()
        qkv = qkv.view(batch_size, seq_len, self.hidden_size)

        attention_output = self.dense(qkv)
        attention_output = self.dropout(attention_output)
        return attention_output


class FeedForward(nn.Module):
    def __init__(self, hidden_size=768, intermediate_size=3072, dropout=0.1):
        super().__init__()
        self.dense1 = nn.Linear(hidden_size, intermediate_size)
        self.dense2 = nn.Linear(intermediate_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, hidden_states):
        hidden_states = self.dense1(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        return hidden_states


class TransformerEncoderLayer(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, intermediate_size=3072, dropout=0.1):
        super().__init__()
        self.attention = MultiHeadSelfAttention(hidden_size, num_attention_heads, dropout)
        self.attention_norm = nn.LayerNorm(hidden_size)

        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)
        self.feed_forward_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.attention(hidden_states, attention_mask)
        hidden_states = self.attention_norm(hidden_states + attention_output)

        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_norm(hidden_states + feed_forward_output)

        return hidden_states


class SimpleTransformer(nn.Module):
    def __init__(self, vocab_size=21128, hidden_size=768, num_layers=6,
                 num_attention_heads=12, intermediate_size=3072, dropout=0.1, max_len=512):
        super().__init__()
        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_encoding = PositionalEncoding(hidden_size, max_len)
        self.token_type_embeddings = nn.Embedding(2, hidden_size)

        self.embeddings_layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(hidden_size, num_attention_heads, intermediate_size, dropout)
            for _ in range(num_layers)
        ])

        self.pooler = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh()
        )

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        batch_size, seq_len = input_ids.shape

        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_encoding(embeddings)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = embeddings + position_embeddings + token_type_embeddings
        embeddings = self.embeddings_layer_norm(embeddings)
        hidden_states = self.dropout(embeddings)

        if attention_mask is not None:
            attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attention_mask = (1.0 - attention_mask) * -10000.0

        for layer in self.encoder_layers:
            hidden_states = layer(hidden_states, attention_mask)

        sequence_output = hidden_states
        pooler_input = hidden_states[:, 0, :]
        pooler_output = self.pooler(pooler_input)

        return sequence_output, pooler_output


if __name__ == "__main__":
    vocab_size = 21128
    hidden_size = 768
    num_layers = 6
    num_attention_heads = 12

    model = SimpleTransformer(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads
    )
    model.eval()

    x = torch.LongTensor([[2450, 15486, 102, 2110]])
    print(f"输入形状: {x.shape}")

    with torch.no_grad():
        sequence_output, pooler_output = model(x)

    print(f"Sequence output 形状: {sequence_output.shape}")
    print(f"Pooler output 形状: {pooler_output.shape}")

    print(f"\n模型总参数量: {sum(p.numel() for p in model.parameters()):,}")
