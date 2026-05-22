import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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


class CausalSelfAttention(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, dropout=0.1, max_len=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_size = hidden_size // num_attention_heads

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)

        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)

        causal_mask = torch.triu(torch.ones(max_len, max_len), diagonal=1)
        causal_mask = causal_mask * -10000.0
        self.register_buffer('causal_mask', causal_mask)

    def transpose_for_scores(self, x):
        new_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        batch_size, seq_len, _ = hidden_states.shape

        q = self.query(hidden_states)
        k = self.key(hidden_states)
        v = self.value(hidden_states)

        q = self.transpose_for_scores(q)
        k = self.transpose_for_scores(k)
        v = self.transpose_for_scores(v)

        qk = torch.matmul(q, k.transpose(-2, -1))
        qk = qk / math.sqrt(self.attention_head_size)

        qk = qk + self.causal_mask[:seq_len, :seq_len].unsqueeze(0).unsqueeze(0)

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


class TransformerDecoderLayer(nn.Module):
    def __init__(self, hidden_size=768, num_attention_heads=12, intermediate_size=3072, dropout=0.1, max_len=512):
        super().__init__()
        self.attention = CausalSelfAttention(hidden_size, num_attention_heads, dropout, max_len)
        self.attention_norm = nn.LayerNorm(hidden_size)

        self.feed_forward = FeedForward(hidden_size, intermediate_size, dropout)
        self.feed_forward_norm = nn.LayerNorm(hidden_size)

    def forward(self, hidden_states):
        attention_output = self.attention(hidden_states)
        hidden_states = self.attention_norm(hidden_states + attention_output)

        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_norm(hidden_states + feed_forward_output)

        return hidden_states


class GPTModel(nn.Module):
    def __init__(self, vocab_size=21128, hidden_size=768, num_layers=6,
                 num_attention_heads=12, intermediate_size=3072, dropout=0.1, max_len=512):
        super().__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size

        self.word_embeddings = nn.Embedding(vocab_size, hidden_size)
        self.position_encoding = PositionalEncoding(hidden_size, max_len)

        self.embeddings_layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(hidden_size, num_attention_heads, intermediate_size, dropout, max_len)
            for _ in range(num_layers)
        ])

        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        self.lm_head.weight = self.word_embeddings.weight

    def forward(self, input_ids):
        batch_size, seq_len = input_ids.shape

        embeddings = self.word_embeddings(input_ids)
        embeddings = self.position_encoding(embeddings)
        embeddings = self.embeddings_layer_norm(embeddings)
        hidden_states = self.dropout(embeddings)

        for layer in self.decoder_layers:
            hidden_states = layer(hidden_states)

        logits = self.lm_head(hidden_states)
        return logits

    @torch.no_grad()
    def generate(self, input_ids, max_new_tokens=50, temperature=1.0, top_k=None, top_p=None):
        self.eval()
        for _ in range(max_new_tokens):
            if input_ids.size(1) > 512:
                input_ids = input_ids[:, -512:]

            logits = self(input_ids)
            logits = logits[:, -1, :] / temperature

            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                logits[indices_to_remove] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids


class SimpleTokenizer:
    def __init__(self):
        self.char2id = {}
        self.id2char = {}
        self.vocab_size = 0

    def train(self, text):
        chars = sorted(list(set(text)))
        self.char2id = {ch: i for i, ch in enumerate(chars)}
        self.char2id['<PAD>'] = len(self.char2id)
        self.char2id['<UNK>'] = len(self.char2id)
        self.id2char = {i: ch for ch, i in self.char2id.items()}
        self.vocab_size = len(self.char2id)
        return self.vocab_size

    def encode(self, text):
        return [self.char2id.get(ch, self.char2id['<UNK>']) for ch in text]

    def decode(self, ids):
        return ''.join([self.id2char.get(i, '<UNK>') for i in ids])


if __name__ == "__main__":
    print("=" * 60)
    print("单向 Transformer (GPT) 文本生成演示")
    print("=" * 60)

    sample_text = """
    自然语言处理是人工智能的重要分支。
    深度学习改变了自然语言处理的方式。
    Transformer模型是自然语言处理的核心。
    语言模型可以生成连贯的文本内容。
    """

    tokenizer = SimpleTokenizer()
    vocab_size = tokenizer.train(sample_text)
    print(f"\n词表大小: {vocab_size}")

    model = GPTModel(
        vocab_size=vocab_size,
        hidden_size=128,
        num_layers=4,
        num_attention_heads=4,
        intermediate_size=512,
        dropout=0.1,
        max_len=256
    )
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    print("\n" + "=" * 60)
    print("训练模型...")
    print("=" * 60)

    encoded = tokenizer.encode(sample_text)
    data = torch.tensor(encoded, dtype=torch.long).unsqueeze(0)

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(500):
        for i in range(data.size(1) - 1):
            input_seq = data[:, :i+1]
            target = data[:, i+1]

            logits = model(input_seq)
            logits = logits[:, -1, :]

            loss = F.cross_entropy(logits, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

    print("\n" + "=" * 60)
    print("文本生成测试")
    print("=" * 60)

    model.eval()
    prompts = ["自然", "深度", "语言"]

    for prompt in prompts:
        input_ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long)
        output_ids = model.generate(input_ids, max_new_tokens=20, temperature=0.8, top_k=10)
        generated_text = tokenizer.decode(output_ids[0].tolist())
        print(f"\n输入: '{prompt}'")
        print(f"生成: '{generated_text}'")

    print("\n" + "=" * 60)
    print("因果掩码验证 (单向性)")
    print("=" * 60)

    test_input = torch.tensor([[1, 2, 3, 4, 5]], dtype=torch.long)
    with torch.no_grad():
        logits_all = model(test_input)
        logits_partial = model(test_input[:, :3])

    print(f"\n完整输入 [1,2,3,4,5] 时，位置2的输出: {logits_all[0, 2, :5].tolist()}")
    print(f"部分输入 [1,2,3] 时，位置2的输出: {logits_partial[0, 2, :5].tolist()}")
    print(f"两者是否相近: {torch.allclose(logits_all[0, 2, :], logits_partial[0, 2, :], atol=1e-5)}")
    print("\n这证明了模型是单向的：位置2的输出只依赖于位置0-2，不受后续位置影响。")
