# 第七章：循环神经网络 (Recurrent Neural Networks)

## 📌 章节目标
- 理解序列数据和循环结构
- 掌握基本 RNN 及其梯度问题
- 深入学习 LSTM 和 GRU 的设计
- 了解双向 RNN 和多层 RNN
- 实战：文本分类、情感分析、序列预测

---

## 7.1 为什么需要 RNN？

### 🎯 序列数据的特性

**什么是序列数据？**

```
普通数据（独立）：
  图片、房价、医学诊断
  每个样本是独立的

序列数据（有依赖）：
  文本：今天天气→很好→适合→出去
  语音：音频帧 t₁, t₂, ..., tₙ
  时间序列：股票价格、温度记录

特性：
  ✓ 长度可变
  ✓ 前后有依赖关系
  ✓ 顺序很重要
```

### ❌ CNN 和 FC 的局限

**全连接网络**：
- 固定输入大小
- 忽视序列顺序
- 无法处理可变长度

**CNN**：
- 虽然有局部连接，但感受野有限
- 需要很多层才能捕获长距离依赖
- 不够自然

### ✅ RNN 的优势

```
设计用于序列数据：
  ✓ 可变长度输入
  ✓ 循环结构保留序列信息
  ✓ 参数共享（所有时间步共用）
  ✓ 可以建模长期依赖（理论上）
```

---

## 7.2 基本 RNN (Vanilla RNN)

### 🔄 RNN 的循环结构

**展开视图**：

```
y₁      y₂      y₃      y₄
↑       ↑       ↑       ↑
h₁      h₂      h₃      h₄
↑       ↑       ↑       ↑
x₁  →  h₁  →  h₂  →  h₃  →  h₄
        ↓       ↓       ↓
        (循环)

隐藏状态作为信息载体传递
```

**折叠视图**（参数共享）：

```
      x(t)
        ↓
    [U]  [W]  [V]
      ↓    ↓    ↓
   h(t-1) → RNN → h(t) → y(t)
            单元
```

### 📐 RNN 计算

**单时刻计算**：

```
h(t) = tanh(U·x(t) + W·h(t-1) + b)
y(t) = V·h(t) + c

或用激活函数 σ：
h(t) = σ(U·x(t) + W·h(t-1) + b)
```

**符号说明**：
- `x(t)`: t 时刻的输入
- `h(t)`: t 时刻的隐藏状态
- `y(t)`: t 时刻的输出
- `U`: 输入到隐藏的权重
- `W`: 隐藏到隐藏的权重（循环）
- `V`: 隐藏到输出的权重
- `b, c`: 偏置

### 💻 从零实现 RNN

```python
import numpy as np

class VanillaRNN:
    def __init__(self, input_size, hidden_size, output_size,
                 learning_rate=0.01):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.learning_rate = learning_rate

        # 初始化权重
        self.U = np.random.randn(hidden_size, input_size) * 0.01
        self.W = np.random.randn(hidden_size, hidden_size) * 0.01
        self.V = np.random.randn(output_size, hidden_size) * 0.01
        self.b = np.zeros((hidden_size, 1))
        self.c = np.zeros((output_size, 1))

    def forward(self, X):
        """
        前向传播

        参数：
            X: 序列输入 [(seq_len, input_size), ...]

        返回：
            Y: 输出序列
            cache: 用于反向传播的中间值
        """
        seq_len = len(X)

        # 初始化隐藏状态
        h = np.zeros((self.hidden_size, 1))

        # 存储中间值
        cache = {
            'X': X,
            'h': [h],  # 包括初始 h
            'z': [],
            'y': []
        }

        # 前向传播
        for t in range(seq_len):
            # 隐藏状态计算
            z = self.U @ X[t] + self.W @ h + self.b
            h = np.tanh(z)

            # 输出计算
            y = self.V @ h + self.c

            # 保存中间值
            cache['z'].append(z)
            cache['h'].append(h)
            cache['y'].append(y)

        return np.array(cache['y']), cache

    def backward(self, dY, cache):
        """
        反向传播（BPTT）

        参数：
            dY: 输出梯度 [(output_size, 1), ...]
            cache: 前向传播的中间值
        """
        seq_len = len(dY)

        # 初始化梯度
        dU = np.zeros_like(self.U)
        dW = np.zeros_like(self.W)
        dV = np.zeros_like(self.V)
        db = np.zeros_like(self.b)
        dc = np.zeros_like(self.c)

        # 初始隐藏状态梯度
        dh_next = np.zeros((self.hidden_size, 1))

        # 反向遍历时间步
        for t in reversed(range(seq_len)):
            # 输出层梯度
            dV += dY[t] @ cache['h'][t+1].T
            dc += dY[t]

            # 隐藏层梯度
            dh = self.V.T @ dY[t] + dh_next

            # tanh 的梯度
            dz = dh * (1 - np.tanh(cache['z'][t])**2)

            # 权重梯度
            dU += dz @ cache['X'][t].T
            dW += dz @ cache['h'][t].T
            db += dz

            # 传递到前一时刻
            dh_next = self.W.T @ dz

        # 梯度裁剪（防止梯度爆炸）
        for dparam in [dU, dW, dV, db, dc]:
            np.clip(dparam, -5, 5, out=dparam)

        return dU, dW, dV, db, dc

    def update_parameters(self, dU, dW, dV, db, dc):
        """更新参数"""
        self.U -= self.learning_rate * dU
        self.W -= self.learning_rate * dW
        self.V -= self.learning_rate * dV
        self.b -= self.learning_rate * db
        self.c -= self.learning_rate * dc

# 示例：预测数字序列
def train_rnn():
    # 超参数
    input_size = 1
    hidden_size = 10
    output_size = 1
    seq_len = 5

    rnn = VanillaRNN(input_size, hidden_size, output_size,
                     learning_rate=0.01)

    # 生成简单数据（t 时刻预测 t+1）
    X_train = [np.array([[i]]) for i in range(5)]
    y_train = [np.array([[i+1]]) for i in range(5)]

    # 训练
    for epoch in range(100):
        Y_pred, cache = rnn.forward(X_train)

        # 计算损失
        loss = np.mean((Y_pred - np.array(y_train))**2)

        # 梯度计算
        dY = 2 * (Y_pred - np.array(y_train)) / len(y_train)
        dU, dW, dV, db, dc = rnn.backward(dY, cache)

        # 更新参数
        rnn.update_parameters(dU, dW, dV, db, dc)

        if epoch % 20 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")

train_rnn()
```

---

## 7.3 梯度问题：消失和爆炸

### 🚨 梯度消失 (Vanishing Gradient)

**问题**：长期依赖难以学习

```
h(t) = tanh(U·x(t) + W·h(t-1) + b)

∂h(t)/∂h(t-1) = W·diag(1 - tanh²(...))

对 t 步之前的梯度：
∂h(T)/∂h(t) = ∏(τ=t+1 to T) [W·diag(1-tanh²(...))]

如果 ||W|| < 1，则：
||∂h(T)/∂h(t)|| ≈ ||W||^(T-t)

T-t 很大时，梯度接近 0 → 梯度消失
```

**后果**：
- 早期权重几乎不更新
- 无法学习长期依赖

### 💥 梯度爆炸 (Exploding Gradient)

**问题**：如果 ||W|| > 1

```
梯度 ∝ ||W||^(T-t) → ∞

导致：
- 参数更新不稳定
- NaN/Inf 值
- 训练崩溃
```

### ✅ 解决方案

#### **1. 梯度裁剪 (Gradient Clipping)**

**防止梯度爆炸**

```python
def clip_gradients(gradients, max_norm=5):
    """
    L2 范数裁剪
    """
    total_norm = 0
    for g in gradients:
        total_norm += np.sum(g**2)
    total_norm = np.sqrt(total_norm)

    clip_ratio = max_norm / (total_norm + 1e-8)
    clip_ratio = min(clip_ratio, 1.0)

    clipped_grads = []
    for g in gradients:
        clipped_grads.append(g * clip_ratio)

    return clipped_grads
```

#### **2. 权重初始化**

```python
# 正交初始化
def orthogonal_init(shape):
    """正交初始化 W 矩阵"""
    Q, R = np.linalg.qr(np.random.randn(*shape))
    return Q

# 使用
W = orthogonal_init((hidden_size, hidden_size))
```

#### **3. 激活函数选择**

```
ReLU 的梯度恒为 1（在正区域）
不容易梯度消失

tanh 的梯度最大为 0.25
容易梯度消失
```

---

## 7.4 LSTM (Long Short-Term Memory) ⭐

### 🎯 核心思想

**使用"记忆单元"代替隐藏状态**

```
传统 RNN：
  信息通过隐藏状态传递
  每步都被破坏性地改变

LSTM：
  有专门的"细胞状态"C(t)
  信息可以长期保留
  通过门控机制有选择地更新
```

### 📐 LSTM 的四个门

**1. 遗忘门 (Forget Gate)**

```
f(t) = σ(W_f·[h(t-1), x(t)] + b_f)

作用：决定哪些信息被丢弃
f(t) ≈ 0: 丢弃
f(t) ≈ 1: 保留
```

**2. 输入门 (Input Gate)**

```
i(t) = σ(W_i·[h(t-1), x(t)] + b_i)
C̃(t) = tanh(W_c·[h(t-1), x(t)] + b_c)

作用：决定新信息
i(t): 有多少新信息进入
C̃(t): 新信息的内容
```

**3. 细胞状态更新 (Cell State Update)**

```
C(t) = f(t) ⊙ C(t-1) + i(t) ⊙ C̃(t)

⊙ 表示逐元素乘法（Hadamard 积）

过程：
  前一个细胞状态 × 遗忘门
  + 新信息 × 输入门
```

**4. 输出门 (Output Gate)**

```
o(t) = σ(W_o·[h(t-1), x(t)] + b_o)
h(t) = o(t) ⊙ tanh(C(t))

作用：决定输出多少信息
```

### 📊 LSTM 单元图

```
        ┌─────────────────────┐
        │  细胞状态 C(t)      │ ← 长期记忆
        └──────┬──────────────┘
               │
        ┌──────▼──────┐
        │   ⊙ f(t)    │ ← 遗忘门（保留多少）
        └──────┬───────┘
               │
        ┌──────▼──────────┐
        │ ⊙ + ⊙ i(t) C̃(t) │ ← 新信息加入
        └──────┬──────────┘
               │
               ▼
        ┌──────────────┐
        │ tanh ⊙ o(t) │ ← 输出门
        └──────┬───────┘
               │
               ▼
            h(t) ← 短期记忆
```

### 💻 PyTorch 实现

```python
import torch
import torch.nn as nn

# 方式1：使用高级 LSTM
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # 输入格式 (batch, seq, feature)
            dropout=0.3 if num_layers > 1 else 0
        )

        # 输出层
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        参数：
            x: (batch_size, seq_len, input_size)
        """
        # LSTM 输出
        lstm_out, (h_n, c_n) = self.lstm(x)
        # lstm_out: (batch, seq_len, hidden_size)
        # h_n: (num_layers, batch, hidden_size)
        # c_n: (num_layers, batch, hidden_size)

        # 使用最后一个时刻的输出
        last_out = lstm_out[:, -1, :]  # (batch, hidden_size)

        # 全连接层
        output = self.fc(last_out)  # (batch, output_size)

        return output

# 方式2：从零实现 LSTM 单元
class LSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 四个门的权重矩阵
        self.W_f = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_i = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_c = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_o = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev, c_prev):
        """
        参数：
            x: (batch, input_size)
            h_prev: (batch, hidden_size)
            c_prev: (batch, hidden_size)

        返回：
            h: (batch, hidden_size)
            c: (batch, hidden_size)
        """
        # 拼接输入和前一隐藏状态
        combined = torch.cat([x, h_prev], dim=1)

        # 四个门
        f = torch.sigmoid(self.W_f(combined))  # 遗忘门
        i = torch.sigmoid(self.W_i(combined))  # 输入门
        c_tilde = torch.tanh(self.W_c(combined))  # 候选值
        o = torch.sigmoid(self.W_o(combined))  # 输出门

        # 更新细胞状态
        c = f * c_prev + i * c_tilde

        # 计算隐藏状态
        h = o * torch.tanh(c)

        return h, c

# 使用自定义 LSTM Cell
class CustomLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(CustomLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 多层 LSTM
        self.lstm_cells = nn.ModuleList([
            LSTMCell(
                input_size if layer == 0 else hidden_size,
                hidden_size
            )
            for layer in range(num_layers)
        ])

        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        """
        参数：
            x: (batch, seq_len, input_size)
        """
        batch_size, seq_len, _ = x.size()

        # 初始化隐藏状态和细胞状态
        h = [torch.zeros(batch_size, self.hidden_size)
             for _ in range(self.num_layers)]
        c = [torch.zeros(batch_size, self.hidden_size)
             for _ in range(self.num_layers)]

        # 前向传播
        for t in range(seq_len):
            x_t = x[:, t, :]  # (batch, input_size)

            for layer in range(self.num_layers):
                h[layer], c[layer] = self.lstm_cells[layer](
                    x_t, h[layer], c[layer]
                )
                x_t = h[layer]

        # 使用最后一个时刻的隐藏状态
        output = self.fc(h[-1])

        return output
```

---

## 7.5 GRU (Gated Recurrent Unit)

### 🎯 简化的 LSTM

**LSTM 问题**：参数多，计算复杂

**GRU 解决**：只用两个门，结构更简洁

### 📐 GRU 的两个门

**1. 重置门 (Reset Gate)**

```
r(t) = σ(W_r·[h(t-1), x(t)] + b_r)

作用：决定有多少历史信息被遗忘
```

**2. 更新门 (Update Gate)**

```
z(t) = σ(W_z·[h(t-1), x(t)] + b_z)

作用：决定新旧信息的比例
```

**3. 候选隐藏状态**

```
h̃(t) = tanh(W·[r(t) ⊙ h(t-1), x(t)] + b)

使用重置门来选择历史信息
```

**4. 隐藏状态更新**

```
h(t) = (1 - z(t)) ⊙ h̃(t) + z(t) ⊙ h(t-1)

= 新信息比例 × 候选值 + 历史信息比例 × 前值
```

### 📊 LSTM vs GRU

```
LSTM：
  - 细胞状态 C(t) 用于长期记忆
  - 隐藏状态 h(t) 用于短期输出
  - 3个门（遗忘、输入、输出）
  - 参数多，表达能力强

GRU：
  - 细胞状态和隐藏状态合并
  - 2个门（重置、更新）
  - 参数少（约 LSTM 的 2/3）
  - 计算速度快
  - 在大多数任务上性能相当
```

### 💻 实现

```python
class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(GRUCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # 重置门和更新门
        self.W_r = nn.Linear(input_size + hidden_size, hidden_size)
        self.W_z = nn.Linear(input_size + hidden_size, hidden_size)
        # 候选隐藏状态
        self.W_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev):
        """
        参数：
            x: (batch, input_size)
            h_prev: (batch, hidden_size)

        返回：
            h: (batch, hidden_size)
        """
        combined = torch.cat([x, h_prev], dim=1)

        # 重置门
        r = torch.sigmoid(self.W_r(combined))

        # 更新门
        z = torch.sigmoid(self.W_z(combined))

        # 候选隐藏状态
        combined_reset = torch.cat([x, r * h_prev], dim=1)
        h_tilde = torch.tanh(self.W_h(combined_reset))

        # 更新隐藏状态
        h = (1 - z) * h_tilde + z * h_prev

        return h

# PyTorch 高级接口
model = nn.GRU(
    input_size=10,
    hidden_size=20,
    num_layers=2,
    batch_first=True
)
```

---

## 7.6 双向 RNN (Bidirectional RNN)

### 🎯 问题：前向 RNN 的局限

```
前向 RNN：
  x₁ → x₂ → x₃ → x₄

h₃ 无法看到 x₄ 的信息
但有些任务需要完整的上下文！
```

### ✅ 双向 RNN 解决

**同时运行前向和后向 RNN**

```
前向：x₁ → x₂ → x₃ → x₄
      →h₁→ →h₂→ →h₃→ →h₄

后向：        ← ← ← ←
      ←h̄₁←  ←h̄₂←  ←h̄₃←  ←h̄₄←

输出：[h₃, h̄₃] = 结合两个方向的信息
```

**计算**：

```
h₃ = [h₃_forward, h₃_backward]
   = [LSTM_fwd(x₁:x₃), LSTM_bwd(x₃:x₁)]
```

### 💻 实现

```python
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True  # 关键！
        )

        # 双向 LSTM 输出大小是 2×hidden_size
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        """
        参数：
            x: (batch, seq_len, input_size)
        """
        # LSTM 输出
        lstm_out, _ = self.lstm(x)
        # lstm_out: (batch, seq_len, 2×hidden_size)

        # 使用最后时刻
        last_out = lstm_out[:, -1, :]

        output = self.fc(last_out)
        return output

# 或者用所有时刻（如 NER 标签）
class BiLSTMSequence(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size,
                           batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)  # (batch, seq, 2×hidden)

        # 对每个时刻做分类
        output = self.fc(lstm_out)  # (batch, seq, output_size)
        return output
```

---

## 7.7 实战 1：文本情感分析

### 📋 任务设定

**数据**：电影评论
```
"这部电影太棒了！" → 正面 (1)
"完全是浪费时间。" → 负面 (0)
```

### 💻 完整实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from collections import Counter
import re

# ==================== 数据预处理 ====================

class Tokenizer:
    def __init__(self, vocab_size=5000):
        self.vocab_size = vocab_size
        self.word2idx = {'<PAD>': 0, '<UNK>': 1}
        self.idx2word = {0: '<PAD>', 1: '<UNK>'}

    def build_vocab(self, texts):
        """构建词汇表"""
        word_freq = Counter()
        for text in texts:
            words = self.tokenize(text)
            word_freq.update(words)

        idx = 2
        for word, freq in word_freq.most_common(self.vocab_size - 2):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            idx += 1

    def tokenize(self, text):
        """分词"""
        text = text.lower()
        text = re.sub(r'[^a-z\s]', '', text)
        return text.split()

    def encode(self, text, max_len=100):
        """编码文本"""
        words = self.tokenize(text)
        ids = [self.word2idx.get(w, 1) for w in words]

        # Padding 或截断
        if len(ids) < max_len:
            ids = ids + [0] * (max_len - len(ids))
        else:
            ids = ids[:max_len]

        return ids

    def decode(self, ids):
        """解码"""
        return ' '.join([self.idx2word.get(i, '<UNK>') for i in ids])

# ==================== 模型定义 ====================

class SentimentLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_size, hidden_size,
                 num_layers=2, dropout=0.3):
        super(SentimentLSTM, self).__init__()

        # Embedding 层
        self.embedding = nn.Embedding(vocab_size, embedding_size,
                                     padding_idx=0)

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=embedding_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=True
        )

        # 注意力机制（可选）
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size * 2,
            num_heads=4,
            batch_first=True,
            dropout=dropout
        )

        # 全连接层
        self.fc1 = nn.Linear(hidden_size * 2, 128)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(128, 1)

        # Batch Normalization
        self.bn = nn.BatchNorm1d(128)

    def forward(self, x):
        """
        参数：
            x: (batch, seq_len)
        """
        # Embedding
        embedded = self.embedding(x)  # (batch, seq_len, embedding_size)

        # LSTM
        lstm_out, (h_n, c_n) = self.lstm(embedded)
        # lstm_out: (batch, seq_len, hidden_size*2)

        # 使用注意力机制
        attn_out, attn_weights = self.attention(
            lstm_out, lstm_out, lstm_out
        )

        # 取最后一个时刻或池化
        # 方式1：最后时刻
        # last_hidden = lstm_out[:, -1, :]

        # 方式2：平均池化
        # last_hidden = torch.mean(lstm_out, dim=1)

        # 方式3：最大池化
        # last_hidden, _ = torch.max(lstm_out, dim=1)

        # 方式4：使用注意力输出
        last_hidden = torch.mean(attn_out, dim=1)

        # 全连接层
        x = torch.relu(self.fc1(last_hidden))
        x = self.bn(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return torch.sigmoid(x)

# ==================== 训练代码 ====================

def train_sentiment_model():
    # 超参数
    VOCAB_SIZE = 5000
    EMBEDDING_SIZE = 128
    HIDDEN_SIZE = 64
    NUM_LAYERS = 2
    MAX_LEN = 100
    BATCH_SIZE = 64
    EPOCHS = 20
    LEARNING_RATE = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 示例数据（实际应使用真实数据集如 IMDB）
    texts_train = [
        "This movie is great and wonderful",
        "I love this film so much",
        "Amazing performance by the actors",
        "Terrible waste of time",
        "Boring and dull movie",
        "I hate this film"
    ] * 100  # 复制以增加数据量

    labels_train = [1, 1, 1, 0, 0, 0] * 100

    # 构建词汇表
    tokenizer = Tokenizer(vocab_size=VOCAB_SIZE)
    tokenizer.build_vocab(texts_train)

    # 编码数据
    X_train = torch.tensor([tokenizer.encode(t, MAX_LEN) for t in texts_train])
    y_train = torch.tensor(labels_train, dtype=torch.float32).unsqueeze(1)

    # 数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # 模型
    model = SentimentLSTM(
        vocab_size=VOCAB_SIZE,
        embedding_size=EMBEDDING_SIZE,
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS
    ).to(device)

    # 损失和优化器
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=3, factor=0.5
    )

    # 训练
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5)

            optimizer.step()

            # 统计
            total_loss += loss.item()
            predicted = (outputs > 0.5).float()
            correct += (predicted == y_batch).sum().item()
            total += y_batch.size(0)

        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total

        print(f'Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.4f}, Acc: {accuracy:.2f}%')

        scheduler.step(avg_loss)

    # 保存模型
    torch.save(model.state_dict(), 'sentiment_lstm.pth')

    return model, tokenizer

# ==================== 预测函数 ====================

def predict_sentiment(model, tokenizer, text, device):
    """预测单个文本的情感"""
    model.eval()

    # 编码
    encoded = tokenizer.encode(text, max_len=100)
    x = torch.tensor([encoded]).to(device)

    # 预测
    with torch.no_grad():
        output = model(x)
        prob = output.item()
        sentiment = "正面" if prob > 0.5 else "负面"

    return sentiment, prob

# 使用
if __name__ == "__main__":
    model, tokenizer = train_sentiment_model()

    # 测试
    test_texts = [
        "This is an amazing movie!",
        "Terrible and boring film.",
        "Not bad, quite enjoyable."
    ]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for text in test_texts:
        sentiment, prob = predict_sentiment(model, tokenizer, text, device)
        print(f"\n文本: {text}")
        print(f"情感: {sentiment} (置信度: {prob:.4f})")
```

---

## 7.8 实战 2：时间序列预测

### 📋 任务：预测股票价格

```python
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# ==================== 数据准备 ====================

def create_sequences(data, seq_length):
    """
    创建序列数据

    参数：
        data: 原始数据 (n_samples,)
        seq_length: 序列长度

    返回：
        X: (n_samples - seq_length, seq_length, 1)
        y: (n_samples - seq_length, 1)
    """
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length])

    return np.array(X), np.array(y)

# 生成模拟数据（实际应使用真实股票数据）
def generate_stock_data(n_samples=1000):
    """生成模拟股票价格"""
    t = np.linspace(0, 100, n_samples)
    # 趋势 + 季节性 + 噪声
    trend = 0.02 * t
    seasonal = 10 * np.sin(2 * np.pi * t / 50)
    noise = np.random.randn(n_samples) * 2

    price = 100 + trend + seasonal + noise
    return price

# ==================== 模型定义 ====================

class StockLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2, dropout=0.2):
        super(StockLSTM, self).__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM 层
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        """
        参数：
            x: (batch, seq_len, input_size)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)

        # 取最后一个时刻
        last_out = lstm_out[:, -1, :]

        # 预测
        output = self.fc(last_out)

        return output

# ==================== 训练代码 ====================

def train_stock_predictor():
    # 超参数
    SEQ_LENGTH = 30  # 使用30天预测下一天
    BATCH_SIZE = 32
    EPOCHS = 100
    LEARNING_RATE = 0.001

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 生成数据
    data = generate_stock_data(n_samples=1000)

    # 归一化
    scaler = MinMaxScaler()
    data_normalized = scaler.fit_transform(data.reshape(-1, 1)).flatten()

    # 创建序列
    X, y = create_sequences(data_normalized, SEQ_LENGTH)
    X = X.reshape(-1, SEQ_LENGTH, 1)

    # 训练集和测试集分割
    train_size = int(0.8 * len(X))
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # 转换为 Tensor
    X_train = torch.FloatTensor(X_train)
    y_train = torch.FloatTensor(y_train).unsqueeze(1)
    X_test = torch.FloatTensor(X_test)
    y_test = torch.FloatTensor(y_test).unsqueeze(1)

    # 数据加载器
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True
    )

    # 模型
    model = StockLSTM(
        input_size=1,
        hidden_size=64,
        num_layers=2,
        dropout=0.2
    ).to(device)

    # 损失和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5, verbose=True
    )

    # 训练
    train_losses = []

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            # 前向传播
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(train_loader)
        train_losses.append(avg_loss)

        # 验证
        model.eval()
        with torch.no_grad():
            X_test_device = X_test.to(device)
            test_pred = model(X_test_device)
            test_loss = criterion(test_pred, y_test.to(device))

        scheduler.step(test_loss)

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{EPOCHS}], '
                  f'Train Loss: {avg_loss:.6f}, Test Loss: {test_loss:.6f}')

    # ==================== 可视化预测结果 ====================

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train.to(device)).cpu().numpy()
        test_pred = model(X_test.to(device)).cpu().numpy()

    # 反归一化
    train_pred = scaler.inverse_transform(train_pred)
    test_pred = scaler.inverse_transform(test_pred)
    y_train_actual = scaler.inverse_transform(y_train.numpy())
    y_test_actual = scaler.inverse_transform(y_test.numpy())

    # 绘图
    plt.figure(figsize=(15, 6))

    # 训练集
    plt.subplot(1, 2, 1)
    plt.plot(y_train_actual, label='真实值', alpha=0.7)
    plt.plot(train_pred, label='预测值', alpha=0.7)
    plt.title('训练集预测')
    plt.xlabel('时间步')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 测试集
    plt.subplot(1, 2, 2)
    plt.plot(y_test_actual, label='真实值', alpha=0.7)
    plt.plot(test_pred, label='预测值', alpha=0.7)
    plt.title('测试集预测')
    plt.xlabel('时间步')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 计算指标
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    test_mse = mean_squared_error(y_test_actual, test_pred)
    test_mae = mean_absolute_error(y_test_actual, test_pred)
    test_r2 = r2_score(y_test_actual, test_pred)

    print(f"\n测试集指标：")
    print(f"MSE: {test_mse:.4f}")
    print(f"MAE: {test_mae:.4f}")
    print(f"R²: {test_r2:.4f}")

    return model, scaler

# 运行
if __name__ == "__main__":
    model, scaler = train_stock_predictor()
```

---

## 7.9 实战 3：序列到序列 (Seq2Seq)

### 📋 任务：机器翻译

```python
class Seq2SeqLSTM(nn.Module):
    def __init__(self, input_vocab_size, output_vocab_size,
                 embedding_size, hidden_size, num_layers=2):
        super(Seq2SeqLSTM, self).__init__()

        # Encoder
        self.encoder_embedding = nn.Embedding(input_vocab_size, embedding_size)
        self.encoder = nn.LSTM(
            embedding_size, hidden_size, num_layers,
            batch_first=True
        )

        # Decoder
        self.decoder_embedding = nn.Embedding(output_vocab_size, embedding_size)
        self.decoder = nn.LSTM(
            embedding_size, hidden_size, num_layers,
            batch_first=True
        )

        # 输出层
        self.fc = nn.Linear(hidden_size, output_vocab_size)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        参数：
            src: (batch, src_seq_len) 源语言
            tgt: (batch, tgt_seq_len) 目标语言
            teacher_forcing_ratio: 使用真实目标的概率
        """
        batch_size = src.size(0)
        tgt_len = tgt.size(1)
        tgt_vocab_size = self.fc.out_features

        # 编码器
        embedded_src = self.encoder_embedding(src)
        encoder_outputs, (hidden, cell) = self.encoder(embedded_src)

        # 解码器初始输入（<SOS> token）
        decoder_input = tgt[:, 0].unsqueeze(1)

        # 存储输出
        outputs = torch.zeros(batch_size, tgt_len, tgt_vocab_size)

        # 逐步解码
        for t in range(1, tgt_len):
            # 解码一步
            embedded_tgt = self.decoder_embedding(decoder_input)
            decoder_output, (hidden, cell) = self.decoder(
                embedded_tgt, (hidden, cell)
            )

            # 预测
            output = self.fc(decoder_output.squeeze(1))
            outputs[:, t, :] = output

            # Teacher forcing
            use_teacher_forcing = np.random.random() < teacher_forcing_ratio
            top1 = output.argmax(1)

            decoder_input = tgt[:, t].unsqueeze(1) if use_teacher_forcing else top1.unsqueeze(1)

        return outputs

# 使用
model = Seq2SeqLSTM(
    input_vocab_size=5000,
    output_vocab_size=5000,
    embedding_size=256,
    hidden_size=512,
    num_layers=2
)
```

---

## 📝 本章作业

### 作业 1：RNN 梯度分析

```python
# TODO:
# 1. 实现 vanilla RNN
# 2. 在长序列上训练
# 3. 可视化梯度流
# 4. 观察梯度消失现象
# 5. 对比 LSTM 的梯度流
```

### 作业 2：情感分析完整项目

```python
# 使用真实数据集（IMDB 或中文评论）
# 要求：
# 1. 数据预处理和 EDA
# 2. 实现 LSTM 和 GRU 模型
# 3. 对比双向和单向
# 4. 加入注意力机制
# 5. 超参数调优
# 6. 可视化注意力权重
# 7. 编写完整报告
```

### 作业 3：文本生成

```python
# 字符级语言模型
# 1. 使用莎士比亚文本训练
# 2. 实现 LSTM 生成模型
# 3. 尝试不同的采样策略（贪心、top-k、nucleus）
# 4. 生成新文本并评估质量
```

---

## 🔑 本章关键概念

| 概念 | 说明 |
|------|------|
| 序列数据 | 前后有依赖关系的数据 |
| RNN | 循环神经网络，处理序列 |
| 隐藏状态 | 序列信息的载体 |
| BPTT | 反向传播穿越时间 |
| 梯度消失/爆炸 | RNN 的核心问题 |
| LSTM | 长短期记忆网络 |
| 门控机制 | 控制信息流 |
| 细胞状态 | LSTM 的长期记忆 |
| GRU | 简化的 LSTM |
| 双向 RNN | 同时看前后文 |

---
