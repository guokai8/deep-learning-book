# 第八章：Attention 与 Transformer

## 📌 章节目标
- 理解注意力机制的动机和原理
- 掌握 Self-Attention 的计算
- 深入学习 Transformer 架构
- 了解 BERT 和 GPT 的设计
- 实战：文本分类、翻译

---

## 8.1 为什么需要 Attention？

### 🚨 Seq2Seq 的瓶颈

**传统 Seq2Seq**：

```
编码器：源句子 → 固定长度向量 h
          ↓
解码器：h → 目标句子

问题：所有信息压缩到固定向量 h
     长句子信息丢失！
```

**例子**：翻译长句

```
英文：The quick brown fox jumps over the lazy dog.
     (9个词)

编码成单个向量 h (512维)
     ↓
解码时，早期词的信息已经模糊
```

### ✅ Attention 解决方案

**核心思想**：解码每个词时，动态关注源句子的不同部分

```
编码器：产生所有时刻的隐藏状态 h₁, h₂, ..., hₙ

解码第 t 个词时：
  1. 计算与每个 hᵢ 的相关性
  2. 加权求和得到 context vector c_t
  3. 使用 c_t 生成输出
```

---

## 8.2 Attention 机制

### 📐 计算过程

**1. 计算注意力分数**

```
score(h_t, h_s) = h_t^T · W · h_s

或简化版：
score(h_t, h_s) = h_t^T · h_s  (点积)
```

**2. 归一化（Softmax）**

```
α_t,s = exp(score(h_t, h_s)) / Σ_i exp(score(h_t, h_i))

α_t = [α_t,1, α_t,2, ..., α_t,n]  (注意力权重)
```

**3. 加权求和**

```
c_t = Σ_s α_t,s · h_s

c_t: context vector（上下文向量）
```

**4. 生成输出**

```
output_t = f(h_t, c_t)
```

### 💻 实现

```python
class BahdanauAttention(nn.Module):
    """Bahdanau (Additive) Attention"""

    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.W_h = nn.Linear(hidden_size, hidden_size)
        self.W_s = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, 1)

    def forward(self, query, keys, values):
        """
        参数：
            query: (batch, hidden_size) 解码器当前状态
            keys: (batch, seq_len, hidden_size) 编码器所有状态
            values: (batch, seq_len, hidden_size) 同 keys
        """
        # query: (batch, 1, hidden_size)
        query = query.unsqueeze(1)

        # 计算分数
        score = self.v(torch.tanh(
            self.W_h(keys) + self.W_s(query)
        ))  # (batch, seq_len, 1)

        # 注意力权重
        attention_weights = torch.softmax(score, dim=1)

        # 加权求和
        context = torch.sum(attention_weights * values, dim=1)
        # (batch, hidden_size)

        return context, attention_weights

class LuongAttention(nn.Module):
    """Luong (Multiplicative) Attention"""

    def __init__(self, hidden_size, method='dot'):
        super(LuongAttention, self).__init__()
        self.method = method

        if method == 'general':
            self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == 'concat':
            self.W = nn.Linear(hidden_size * 2, hidden_size)
            self.v = nn.Linear(hidden_size, 1)

    def forward(self, query, keys, values):
        """
        参数同上
        """
        query = query.unsqueeze(1)  # (batch, 1, hidden)

        if self.method == 'dot':
            # 点积注意力
            score = torch.bmm(query, keys.transpose(1, 2))
        elif self.method == 'general':
            # 一般注意力
            score = torch.bmm(self.W(query), keys.transpose(1, 2))
        elif self.method == 'concat':
            # 拼接注意力
            query_expanded = query.expand(-1, keys.size(1), -1)
            score = self.v(torch.tanh(
                self.W(torch.cat([query_expanded, keys], dim=2))
            ))

        # (batch, 1, seq_len)
        attention_weights = torch.softmax(score, dim=2)

        # 加权求和
        context = torch.bmm(attention_weights, values)
        # (batch, 1, hidden_size)

        return context.squeeze(1), attention_weights.squeeze(1)
```

---

## 8.3 Self-Attention ⭐

### 🎯 动机

**普通 Attention**：查询和键来自不同序列（编码器和解码器）

**Self-Attention**：查询、键、值都来自同一序列

**作用**：
- 捕捉序列内部的依赖关系
- 并行计算（不像 RNN 需要串行）
- 全局感受野（每个位置都能看到所有位置）

### 📐 计算机制

**Query, Key, Value (Q, K, V)**

```
给定输入序列 X = [x₁, x₂, ..., xₙ]

通过线性变换得到：
Q = X · W_Q  (queries)
K = X · W_K  (keys)
V = X · W_V  (values)

每个都是 (seq_len, d_k) 矩阵
```

**Scaled Dot-Product Attention**

```
Attention(Q, K, V) = softmax(Q·K^T / √d_k) · V

步骤：
1. 计算 Q 和 K 的点积：Q·K^T
   结果：(seq_len, seq_len) 注意力矩阵

2. 缩放：除以 √d_k
   防止点积过大导致 softmax 饱和

3. Softmax：对每一行归一化
   得到注意力权重

4. 加权求和：乘以 V
   得到输出
```

### 💻 实现

```python
class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        """
        参数：
            Q: (batch, n_heads, seq_len, d_k)
            K: (batch, n_heads, seq_len, d_k)
            V: (batch, n_heads, seq_len, d_v)
            mask: (batch, 1, seq_len, seq_len) 可选
        """
        d_k = Q.size(-1)

        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) / np.sqrt(d_k)
        # (batch, n_heads, seq_len, seq_len)

        # 应用 mask（可选，用于 padding 或未来信息）
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # Softmax
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)

        # 加权求和
        output = torch.matmul(attention_weights, V)
        # (batch, n_heads, seq_len, d_v)

        return output, attention_weights
```

---

## 8.4 Multi-Head Attention

### 🎯 为什么需要多头？

**单头注意力**：只有一组 Q, K, V
- 可能只关注某一种模式

**多头注意力**：多组 Q, K, V 并行
- 不同的头可以关注不同的模式
- 类似 CNN 的多个卷积核

### 📐 计算

```
Multi-Head Attention(Q, K, V) = Concat(head₁, ..., head_h) · W_O

其中：
head_i = Attention(Q·W_Q^i, K·W_K^i, V·W_V^i)

参数：
  W_Q^i ∈ ℝ^(d_model × d_k)
  W_K^i ∈ ℝ^(d_model × d_k)
  W_V^i ∈ ℝ^(d_model × d_v)
  W_O ∈ ℝ^(h·d_v × d_model)

通常：d_k = d_v = d_model / h
```

### 💻 实现

```python
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        # 线性变换
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention(dropout)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        """
        (batch, seq_len, d_model)
        → (batch, n_heads, seq_len, d_k)
        """
        batch_size, seq_len, d_model = x.size()
        return x.view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)

    def combine_heads(self, x):
        """
        (batch, n_heads, seq_len, d_k)
        → (batch, seq_len, d_model)
        """
        batch_size, n_heads, seq_len, d_k = x.size()
        return x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)

    def forward(self, Q, K, V, mask=None):
        """
        参数：
            Q, K, V: (batch, seq_len, d_model)
            mask: 可选
        """
        # 线性变换
        Q = self.W_Q(Q)  # (batch, seq_len, d_model)
        K = self.W_K(K)
        V = self.W_V(V)

        # 分成多个头
        Q = self.split_heads(Q)  # (batch, n_heads, seq_len, d_k)
        K = self.split_heads(K)
        V = self.split_heads(V)

        # 计算注意力
        attn_output, attn_weights = self.attention(Q, K, V, mask)
        # (batch, n_heads, seq_len, d_k)

        # 合并多个头
        output = self.combine_heads(attn_output)
        # (batch, seq_len, d_model)

        # 最后的线性层
        output = self.W_O(output)
        output = self.dropout(output)

        return output, attn_weights
```

---

## 8.5 Transformer 架构 🌟

### 🏗️ 整体结构

```
输入序列
    ↓
[Positional Encoding]
    ↓
┌─────────────────────┐
│  Encoder (×N层)    │
│  - Multi-Head Attn  │
│  - Feed Forward     │
└─────────────────────┘
    ↓
┌─────────────────────┐
│  Decoder (×N层)    │
│  - Masked Attn      │
│  - Cross Attn       │
│  - Feed Forward     │
└─────────────────────┘
    ↓
  输出概率
```

### 🔹 位置编码 (Positional Encoding)

**问题**：Self-Attention 没有顺序信息

**解决**：添加位置编码

```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))

其中：
  pos: 位置（0, 1, 2, ...）
  i: 维度索引（0, 1, ..., d_model/2）
```

**实现**：

```python
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()

        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() *
            (-np.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        参数：
            x: (batch, seq_len, d_model)
        """
        seq_len = x.size(1)
        x = x + self.pe[:, :seq_len, :]
        return x
```

---

### 🔹 Encoder 层

```python
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        # Multi-Head Self-Attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # Feed Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        """
        参数：
            x: (batch, seq_len, d_model)
        """
        # Self-Attention + Residual + Norm
        attn_output, _ = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed Forward + Residual + Norm
        ff_output = self.ff(x)
        x = self.norm2(x + self.dropout(ff_output))

        return x
```

---

### 🔹 Decoder 层

```python
class TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        # Masked Self-Attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # Cross-Attention (Encoder-Decoder Attention)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)

        # Feed Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model)
        )

        # Layer Normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, encoder_output, src_mask=None, tgt_mask=None):
        """
        参数：
            x: (batch, tgt_len, d_model) 目标序列
            encoder_output: (batch, src_len, d_model) 编码器输出
            src_mask: source mask
            tgt_mask: target mask（防止看到未来信息）
        """
        # Masked Self-Attention
        attn_output, _ = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Cross-Attention
        attn_output, _ = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(attn_output))

        # Feed Forward
        ff_output = self.ff(x)
        x = self.norm3(x + self.dropout(ff_output))

        return x
```

---

### 🔹 完整 Transformer

```python
class Transformer(nn.Module):
    def __init__(self, src_vocab_size, tgt_vocab_size,
                 d_model=512, n_heads=8, n_layers=6,
                 d_ff=2048, dropout=0.1, max_len=5000):
        super().__init__()

        self.d_model = d_model

        # Embedding
        self.src_embedding = nn.Embedding(src_vocab_size, d_model)
        self.tgt_embedding = nn.Embedding(tgt_vocab_size, d_model)

        # Positional Encoding
        self.pos_encoding = PositionalEncoding(d_model, max_len)

        # Encoder
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Decoder
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Output
        self.fc_out = nn.Linear(d_model, tgt_vocab_size)
        self.dropout = nn.Dropout(dropout)

    def make_src_mask(self, src):
        """创建 source mask (padding)"""
        src_mask = (src != 0).unsqueeze(1).unsqueeze(2)
        # (batch, 1, 1, src_len)
        return src_mask

    def make_tgt_mask(self, tgt):
        """创建 target mask (padding + future)"""
        tgt_len = tgt.size(1)

        # Padding mask
        tgt_pad_mask = (tgt != 0).unsqueeze(1).unsqueeze(2)
        # (batch, 1, 1, tgt_len)

        # Future mask
        tgt_sub_mask = torch.tril(
            torch.ones((tgt_len, tgt_len), device=tgt.device)
        ).bool()
        # (tgt_len, tgt_len)

        tgt_mask = tgt_pad_mask & tgt_sub_mask
        return tgt_mask

    def encode(self, src, src_mask):
        """Encoder"""
        x = self.src_embedding(src) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.encoder_layers:
            x = layer(x, src_mask)

        return x

    def decode(self, tgt, encoder_output, src_mask, tgt_mask):
        """Decoder"""
        x = self.tgt_embedding(tgt) * np.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)

        for layer in self.decoder_layers:
            x = layer(x, encoder_output, src_mask, tgt_mask)

        return x

    def forward(self, src, tgt):
        """
        参数：
            src: (batch, src_len)
            tgt: (batch, tgt_len)
        """
        src_mask = self.make_src_mask(src)
        tgt_mask = self.make_tgt_mask(tgt)

        encoder_output = self.encode(src, src_mask)
        decoder_output = self.decode(tgt, encoder_output, src_mask, tgt_mask)

        output = self.fc_out(decoder_output)
        return output
```

---

## 8.6 BERT (Bidirectional Encoder Representations from Transformers)

### 🎯 核心思想

**只使用 Transformer 的 Encoder**

**训练任务**：
1. **Masked Language Model (MLM)**
   - 随机 mask 15% 的词
   - 让模型预测被 mask 的词

2. **Next Sentence Prediction (NSP)**
   - 判断两个句子是否连续

### 📐 架构

```
输入：[CLS] 句子1 [SEP] 句子2 [SEP]

Embedding = Token Emb + Segment Emb + Position Emb
    ↓
Transformer Encoder (×12 or 24层)
    ↓
输出：每个 token 的表示

[CLS] 的输出用于分类任务
```

### 💻 使用 BERT（Hugging Face）

```python
from transformers import BertTokenizer, BertModel, BertForSequenceClassification
import torch

# ==================== 加载预训练模型 ====================

# Tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# 模型
model = BertModel.from_pretrained('bert-base-uncased')
model.eval()

# ==================== 文本编码 ====================

text = "Hello, my name is BERT."
inputs = tokenizer(text, return_tensors='pt')

# inputs['input_ids']: token IDs
# inputs['attention_mask']: mask (1=real, 0=padding)

# ==================== 获取表示 ====================

with torch.no_grad():
    outputs = model(**inputs)

# outputs.last_hidden_state: (1, seq_len, 768)
# outputs.pooler_output: (1, 768) [CLS] token

# ==================== 微调用于分类 ====================

class BERTClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(768, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )

        # 使用 [CLS] token
        pooled_output = outputs.pooler_output
        output = self.dropout(pooled_output)
        output = self.fc(output)

        return output

# 使用
model = BERTClassifier(n_classes=2)

# 训练
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
```

---

## 8.7 GPT (Generative Pre-trained Transformer)

### 🎯 核心思想

**只使用 Transformer 的 Decoder**

**训练任务**：**自回归语言模型**
- 给定前面的词，预测下一个词
- P(w_t | w_1, ..., w_{t-1})

### 📐 架构

```
输入：w_1, w_2, ..., w_{t-1}
    ↓
Token Embedding + Position Embedding
    ↓
Transformer Decoder (仅自注意力，×12/24/48层)
    ↓
预测：w_t
```

**与 BERT 的区别**：

| 特性 | BERT | GPT |
|------|------|-----|
| 架构 | Encoder only | Decoder only |
| 注意力 | 双向 | 单向（causal） |
| 训练 | MLM + NSP | 语言建模 |
| 应用 | 理解任务 | 生成任务 |

### 💻 使用 GPT-2

```python
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# 加载模型
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')

# 文本生成
def generate_text(prompt, max_length=50):
    inputs = tokenizer(prompt, return_tensors='pt')

    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_length=max_length,
            num_return_sequences=1,
            no_repeat_ngram_size=2,
            temperature=0.7,
            top_k=50,
            top_p=0.95
        )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return text

# 使用
prompt = "Once upon a time,"
generated_text = generate_text(prompt)
print(generated_text)
```

---

## 📝 本章作业

### 作业 1：从零实现 Transformer

```python
# TODO:
# 1. 实现完整的 Transformer
# 2. 在机器翻译任务上训练
# 3. 可视化注意力权重
# 4. 对比不同层数和头数的效果
```

### 作业 2：BERT 微调

```python
# 使用 Hugging Face Transformers
# 任务：文本分类（IMDB 或 AG News）
# 1. 加载预训练 BERT
# 2. 添加分类头
# 3. 微调
# 4. 评估性能
# 5. 对比从头训练 vs 微调
```

### 作业 3：文本生成

```python
# 使用 GPT-2 或训练自己的模型
# 1. 实现不同的采样策略
#    - Greedy
#    - Beam Search
#    - Top-K
#    - Nucleus (Top-P)
# 2. 对比生成质量
# 3. 实现条件生成
```

---

## 🔑 本章关键概念

| 概念 | 说明 |
|------|------|
| Attention | 动态关注相关信息 |
| Self-Attention | 序列内部的注意力 |
| Multi-Head | 多组注意力并行 |
| Positional Encoding | 位置信息编码 |
| Transformer | 完全基于注意力的架构 |
| Encoder-Decoder | 序列到序列转换 |
| BERT | 预训练双向编码器 |
| GPT | 预训练自回归解码器 |
| MLM | 掩码语言模型 |
| Fine-tuning | 微调预训练模型 |

---

