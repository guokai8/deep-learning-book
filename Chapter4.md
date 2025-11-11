# 第四章：深度神经网络基础 (Deep Neural Networks)

## 📌 章节目标
- 理解从逻辑回归到神经网络的演进
- 掌握前向传播和反向传播算法
- 了解各种激活函数及其作用
- 学习深度网络的初始化和训练技巧
- 实战：用 PyTorch/TensorFlow 构建神经网络

---

## 4.1 从逻辑回归到神经网络

### 🔄 回顾：逻辑回归

```
输入 x → 线性组合 z = w·x + b → Sigmoid σ(z) → 输出 ŷ
```

**局限性**：只能学习线性决策边界

```
示例：XOR 问题

输入          输出
x₁  x₂       y
0   0    →   0
0   1    →   1
1   0    →   1
1   1    →   0

x₂
↑
1 | 0   1    无法用一条直线分开！
0 | 1   0
  |_____→ x₁
  0     1
```

### 💡 解决方案：堆叠多层

**神经网络的核心思想**：

> 一层逻辑回归太简单？那就叠很多层！

```
输入层 → 隐藏层1 → 隐藏层2 → ... → 输出层
```

### 🧠 神经元 (Neuron)

**单个神经元 = 逻辑回归单元**

```
       x₁ ─┐
       x₂ ─┤
       x₃ ─┼→ z = Σwᵢxᵢ + b → a = σ(z) → 输出
      ...  ┤
       xₙ ─┘

输入 → 加权求和 → 激活函数 → 输出
```

**数学表示**：

```
z = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
a = σ(z)

向量形式：
z = wᵀx + b
a = σ(z)
```

### 🕸️ 神经网络 = 很多神经元的组合

```
输入层        隐藏层1         隐藏层2        输出层
  x₁ ───┬──→ h₁⁽¹⁾ ───┬──→ h₁⁽²⁾ ───┬──→ ŷ₁
        │              │              │
  x₂ ───┼──→ h₂⁽¹⁾ ───┼──→ h₂⁽²⁾ ───┼──→ ŷ₂
        │              │              │
  x₃ ───┼──→ h₃⁽¹⁾ ───┼──→ h₃⁽²⁾ ───┼──→ ŷ₃
        │              │              │
  ...   └──→ ...   ───└──→ ...   ───┘

第0层         第1层           第2层         第3层
(输入)       (隐藏)         (隐藏)       (输出)
```

---

## 4.2 前向传播 (Forward Propagation)

### 📐 数学推导

**符号定义**：

```
L: 网络层数
n⁽ˡ⁾: 第 l 层的神经元数量
w⁽ˡ⁾: 第 l 层的权重矩阵
b⁽ˡ⁾: 第 l 层的偏置向量
a⁽ˡ⁾: 第 l 层的激活值（输出）
z⁽ˡ⁾: 第 l 层的加权输入
```

**单层计算**：

```
z⁽ˡ⁾ = W⁽ˡ⁾·a⁽ˡ⁻¹⁾ + b⁽ˡ⁾
a⁽ˡ⁾ = σ(z⁽ˡ⁾)
```

### 🔢 具体例子

**网络结构**：2 → 3 → 1

```
输入层：2个神经元 [x₁, x₂]
隐藏层：3个神经元 [h₁, h₂, h₃]
输出层：1个神经元 [ŷ]
```

**第1层（输入→隐藏）**：

```
z₁⁽¹⁾ = w₁₁⁽¹⁾x₁ + w₁₂⁽¹⁾x₂ + b₁⁽¹⁾
z₂⁽¹⁾ = w₂₁⁽¹⁾x₁ + w₂₂⁽¹⁾x₂ + b₂⁽¹⁾
z₃⁽¹⁾ = w₃₁⁽¹⁾x₁ + w₃₂⁽¹⁾x₂ + b₃⁽¹⁾

a₁⁽¹⁾ = σ(z₁⁽¹⁾)
a₂⁽¹⁾ = σ(z₂⁽¹⁾)
a₃⁽¹⁾ = σ(z₃⁽¹⁾)

矩阵形式：
z⁽¹⁾ = W⁽¹⁾·x + b⁽¹⁾

其中：
     [w₁₁ w₁₂]       [b₁]
W⁽¹⁾=[w₂₁ w₂₂]  b⁽¹⁾=[b₂]
     [w₃₁ w₃₂]       [b₃]
```

**第2层（隐藏→输出）**：

```
z⁽²⁾ = W⁽²⁾·a⁽¹⁾ + b⁽²⁾
ŷ = a⁽²⁾ = σ(z⁽²⁾)

其中：
W⁽²⁾ = [w₁ w₂ w₃]  (1×3 矩阵)
b⁽²⁾ = [b]          (标量)
```

### 💻 代码实现

```python
import numpy as np

def sigmoid(z):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-z))

def forward_propagation(X, parameters):
    """
    前向传播

    参数：
        X: 输入数据 (n_features, m_samples)
        parameters: 字典，包含 W1, b1, W2, b2

    返回：
        A2: 输出层激活值
        cache: 中间值，用于反向传播
    """
    # 获取参数
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # 第1层
    Z1 = np.dot(W1, X) + b1
    A1 = sigmoid(Z1)

    # 第2层
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)

    # 保存中间值
    cache = {
        'Z1': Z1,
        'A1': A1,
        'Z2': Z2,
        'A2': A2
    }

    return A2, cache

# 示例
np.random.seed(42)

# 初始化参数
parameters = {
    'W1': np.random.randn(3, 2) * 0.01,  # 3×2
    'b1': np.zeros((3, 1)),              # 3×1
    'W2': np.random.randn(1, 3) * 0.01,  # 1×3
    'b2': np.zeros((1, 1))               # 1×1
}

# 输入数据
X = np.array([[1.0, 2.0],
              [0.5, 1.5]]).T  # 2×2 (2个样本)

# 前向传播
A2, cache = forward_propagation(X, parameters)

print("输入 X:")
print(X)
print("\n输出 A2:")
print(A2)
print("\n隐藏层激活 A1:")
print(cache['A1'])
```

---

## 4.3 激活函数 (Activation Functions)

### 🤔 为什么需要激活函数？

**如果没有激活函数**（或使用线性激活）：

```
z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾
a⁽¹⁾ = z⁽¹⁾              ← 线性

z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾
    = W⁽²⁾(W⁽¹⁾x + b⁽¹⁾) + b⁽²⁾
    = (W⁽²⁾W⁽¹⁾)x + (W⁽²⁾b⁽¹⁾ + b⁽²⁾)
    = W'x + b'           ← 还是线性！

多层线性变换 = 单层线性变换
深度网络退化成浅层网络！
```

**结论**：激活函数引入非线性，让神经网络能学习复杂函数

---

### 🔹 常见激活函数

#### **1. Sigmoid**

```
σ(z) = 1 / (1 + e⁻ᶻ)

范围：(0, 1)
导数：σ'(z) = σ(z)·(1 - σ(z))
```

**图形**：
```
  1 |         ____
    |       /
0.5 |      /
    |     /
  0 |____/
    |___________ z
   -5  0   5
```

**优点**：
- 输出范围 (0,1)，适合表示概率
- 平滑可导

**缺点**：
- **梯度消失**：z 很大或很小时，梯度接近 0
- **输出不是零中心**：都是正数，导致权重更新效率低
- **计算量大**：有指数运算

**使用场景**：
- 输出层（二元分类）
- 不推荐在隐藏层使用

#### **2. Tanh (双曲正切)**

```
tanh(z) = (eᶻ - e⁻ᶻ) / (eᶻ + e⁻ᶻ)
        = 2·σ(2z) - 1

范围：(-1, 1)
导数：tanh'(z) = 1 - tanh²(z)
```

**图形**：
```
  1 |         ____
    |       /
  0 |      /
    |     /
 -1 |____/
    |___________ z
   -5  0   5
```

**优点**：
- 零中心输出
- 比 Sigmoid 好

**缺点**：
- 仍有梯度消失问题
- 计算量大

**使用场景**：
- RNN/LSTM 中常用
- 隐藏层（但现在更推荐 ReLU）

#### **3. ReLU (Rectified Linear Unit)** ⭐

```
ReLU(z) = max(0, z)

       ⎧ z,  如果 z > 0
     = ⎨
       ⎩ 0,  如果 z ≤ 0

导数：ReLU'(z) = ⎧ 1, 如果 z > 0
                 ⎩ 0, 如果 z ≤ 0
```

**图形**：
```
    |    /
    |   /
    |  /
    | /
____|/_______ z
    0
```

**优点**：
- ✅ **计算简单**：不涉及指数运算
- ✅ **缓解梯度消失**：正区域梯度恒为1
- ✅ **收敛快**：比 Sigmoid/Tanh 快很多
- ✅ **稀疏激活**：约50%神经元被激活

**缺点**：
- ❌ **Dead ReLU**：负区域梯度为0，神经元可能"死亡"
- ❌ 输出不是零中心

**使用场景**：
- 🌟 **默认首选**！隐藏层的标准选择
- CNN 中广泛使用

#### **4. Leaky ReLU**

```
Leaky ReLU(z) = max(αz, z)

              ⎧ z,   如果 z > 0
            = ⎨
              ⎩ αz,  如果 z ≤ 0

通常 α = 0.01
```

**图形**：
```
    |    /
    |   /
    |  /
    | /
___/|_______ z
  / 0
```

**优点**：
- 解决 Dead ReLU 问题
- 负区域仍有小梯度

**变种**：
- **PReLU** (Parametric ReLU)：α 是可学习的参数
- **ELU** (Exponential Linear Unit)

#### **5. Softmax** (输出层)

```
对于 K 个类别：

Softmax(zᵢ) = e^zᵢ / Σⱼ e^zⱼ

性质：
- Σᵢ Softmax(zᵢ) = 1
- 输出可解释为概率
```

**使用场景**：
- 多分类问题的输出层

#### **6. 其他激活函数**

**Swish** (Google 2017):
```
Swish(z) = z·σ(z)

优点：无界、平滑、非单调
```

**GELU** (Gaussian Error Linear Unit):
```
GELU(z) ≈ 0.5z(1 + tanh(√(2/π)(z + 0.044715z³)))

用于 BERT、GPT
```

---

### 📊 激活函数对比

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def tanh(z):
    return np.tanh(z)

def relu(z):
    return np.maximum(0, z)

def leaky_relu(z, alpha=0.01):
    return np.maximum(alpha * z, z)

def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(z) - 1))

# 生成数据
z = np.linspace(-5, 5, 1000)

# 绘图
plt.figure(figsize=(15, 10))

# 激活函数
plt.subplot(2, 2, 1)
plt.plot(z, sigmoid(z), label='Sigmoid', linewidth=2)
plt.plot(z, tanh(z), label='Tanh', linewidth=2)
plt.plot(z, relu(z), label='ReLU', linewidth=2)
plt.plot(z, leaky_relu(z), label='Leaky ReLU', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('z')
plt.ylabel('Activation')
plt.title('激活函数')
plt.legend()
plt.grid(True, alpha=0.3)

# 导数
plt.subplot(2, 2, 2)
sigmoid_derivative = sigmoid(z) * (1 - sigmoid(z))
tanh_derivative = 1 - tanh(z)**2
relu_derivative = (z > 0).astype(float)
leaky_relu_derivative = np.where(z > 0, 1, 0.01)

plt.plot(z, sigmoid_derivative, label='Sigmoid\'', linewidth=2)
plt.plot(z, tanh_derivative, label='Tanh\'', linewidth=2)
plt.plot(z, relu_derivative, label='ReLU\'', linewidth=2)
plt.plot(z, leaky_relu_derivative, label='Leaky ReLU\'', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('z')
plt.ylabel('Derivative')
plt.title('激活函数的导数')
plt.legend()
plt.grid(True, alpha=0.3)

# ReLU 变种对比
plt.subplot(2, 2, 3)
plt.plot(z, relu(z), label='ReLU', linewidth=2)
plt.plot(z, leaky_relu(z), label='Leaky ReLU', linewidth=2)
plt.plot(z, elu(z), label='ELU', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('z')
plt.ylabel('Activation')
plt.title('ReLU 变种对比')
plt.legend()
plt.grid(True, alpha=0.3)

# 梯度消失问题演示
plt.subplot(2, 2, 4)
z_grad = np.linspace(-10, 10, 1000)
sigmoid_grad = sigmoid(z_grad) * (1 - sigmoid(z_grad))
tanh_grad = 1 - tanh(z_grad)**2
relu_grad = (z_grad > 0).astype(float)

plt.plot(z_grad, sigmoid_grad, label='Sigmoid', linewidth=2)
plt.plot(z_grad, tanh_grad, label='Tanh', linewidth=2)
plt.plot(z_grad, relu_grad, label='ReLU', linewidth=2)
plt.axhline(y=0, color='k', linestyle='--', alpha=0.3)
plt.xlabel('z')
plt.ylabel('Gradient')
plt.title('梯度消失问题')
plt.legend()
plt.grid(True, alpha=0.3)
plt.ylim(-0.1, 1.1)

plt.tight_layout()
plt.show()
```

### 💡 激活函数选择指南

```
隐藏层：
  ├─ 默认：ReLU ⭐
  ├─ 尝试：Leaky ReLU / PReLU / ELU
  └─ 避免：Sigmoid / Tanh (除非特殊需求)

输出层：
  ├─ 二元分类：Sigmoid
  ├─ 多元分类：Softmax
  ├─ 回归：Linear (无激活函数)
  └─ 特殊范围：Tanh (输出[-1,1])
```

---

## 4.4 损失函数 (Loss Functions)

### 🎯 常见损失函数

#### **1. 回归问题**

**Mean Squared Error (MSE)**:
```
L = (1/N) Σ(ŷⁿ - yⁿ)²
```

**Mean Absolute Error (MAE)**:
```
L = (1/N) Σ|ŷⁿ - yⁿ|
```

**Huber Loss** (结合 MSE 和 MAE):
```
        ⎧ 0.5·(y - ŷ)²,           if |y - ŷ| ≤ δ
L_δ(y,ŷ)=⎨
        ⎩ δ·(|y - ŷ| - 0.5·δ),   otherwise
```

#### **2. 二元分类**

**Binary Cross Entropy**:
```
L = -(1/N) Σ[yⁿ·log(ŷⁿ) + (1-yⁿ)·log(1-ŷⁿ)]
```

#### **3. 多元分类**

**Categorical Cross Entropy**:
```
L = -(1/N) ΣΣ yᵢⁿ·log(ŷᵢⁿ)
           n i

其中 yᵢⁿ 是 one-hot 编码
```

---

## 4.5 反向传播 (Backpropagation)

### 🎯 核心思想

> 反向传播 = 链式法则 + 从后往前计算梯度

**目标**：计算 ∂L/∂W⁽ˡ⁾ 和 ∂L/∂b⁽ˡ⁾

### 📐 链式法则回顾

```
如果 y = f(u) 且 u = g(x)
则 dy/dx = (dy/du)·(du/dx)

例子：
y = (x² + 1)³
令 u = x² + 1, 则 y = u³

dy/dx = (dy/du)·(du/dx)
      = 3u²·2x
      = 3(x²+1)²·2x
```

### 🔄 反向传播推导

**网络结构**：输入 → 隐藏层 → 输出层

```
前向传播：
  z⁽¹⁾ = W⁽¹⁾x + b⁽¹⁾
  a⁽¹⁾ = σ(z⁽¹⁾)
  z⁽²⁾ = W⁽²⁾a⁽¹⁾ + b⁽²⁾
  a⁽²⁾ = σ(z⁽²⁾) = ŷ

损失：
  L = (y - ŷ)²
```

**反向传播**：

**输出层**：
```
∂L/∂ŷ = -2(y - ŷ)

∂L/∂z⁽²⁾ = ∂L/∂ŷ · ∂ŷ/∂z⁽²⁾
         = ∂L/∂ŷ · σ'(z⁽²⁾)

∂L/∂W⁽²⁾ = ∂L/∂z⁽²⁾ · ∂z⁽²⁾/∂W⁽²⁾
         = ∂L/∂z⁽²⁾ · a⁽¹⁾ᵀ

∂L/∂b⁽²⁾ = ∂L/∂z⁽²⁾
```

**隐藏层**：
```
∂L/∂a⁽¹⁾ = (W⁽²⁾)ᵀ · ∂L/∂z⁽²⁾

∂L/∂z⁽¹⁾ = ∂L/∂a⁽¹⁾ ⊙ σ'(z⁽¹⁾)
         (⊙ 表示逐元素乘法)

∂L/∂W⁽¹⁾ = ∂L/∂z⁽¹⁾ · xᵀ

∂L/∂b⁽¹⁾ = ∂L/∂z⁽¹⁾
```

### 💻 代码实现

```python
def backward_propagation(X, Y, parameters, cache):
    """
    反向传播

    参数：
        X: 输入 (n_features, m_samples)
        Y: 真实标签 (1, m_samples)
        parameters: 权重和偏置
        cache: 前向传播的中间值

    返回：
        gradients: 梯度字典
    """
    m = X.shape[1]  # 样本数

    # 获取参数
    W1 = parameters['W1']
    W2 = parameters['W2']

    # 获取前向传播的值
    A1 = cache['A1']
    A2 = cache['A2']

    # 输出层梯度
    dZ2 = A2 - Y  # 对于 sigmoid + MSE
    dW2 = (1/m) * np.dot(dZ2, A1.T)
    db2 = (1/m) * np.sum(dZ2, axis=1, keepdims=True)

    # 隐藏层梯度
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * A1 * (1 - A1)  # sigmoid 的导数
    dW1 = (1/m) * np.dot(dZ1, X.T)
    db1 = (1/m) * np.sum(dZ1, axis=1, keepdims=True)

    gradients = {
        'dW1': dW1,
        'db1': db1,
        'dW2': dW2,
        'db2': db2
    }

    return gradients

def update_parameters(parameters, gradients, learning_rate):
    """
    更新参数
    """
    parameters['W1'] -= learning_rate * gradients['dW1']
    parameters['b1'] -= learning_rate * gradients['db1']
    parameters['W2'] -= learning_rate * gradients['dW2']
    parameters['b2'] -= learning_rate * gradients['db2']

    return parameters
```

### 🔁 完整训练循环

```python
def train_neural_network(X, Y, hidden_size=4, learning_rate=0.01, epochs=10000):
    """
    训练神经网络
    """
    n_x = X.shape[0]  # 输入特征数
    n_y = Y.shape[0]  # 输出数

    # 初始化参数
    np.random.seed(42)
    parameters = {
        'W1': np.random.randn(hidden_size, n_x) * 0.01,
        'b1': np.zeros((hidden_size, 1)),
        'W2': np.random.randn(n_y, hidden_size) * 0.01,
        'b2': np.zeros((n_y, 1))
    }

    losses = []

    for epoch in range(epochs):
        # 前向传播
        A2, cache = forward_propagation(X, parameters)

        # 计算损失
        loss = np.mean((A2 - Y) ** 2)
        losses.append(loss)

        # 反向传播
        gradients = backward_propagation(X, Y, parameters, cache)

        # 更新参数
        parameters = update_parameters(parameters, gradients, learning_rate)

        # 打印进度
        if epoch % 1000 == 0:
            print(f"Epoch {epoch}: Loss = {loss:.6f}")

    return parameters, losses

# 示例：解决 XOR 问题
X = np.array([[0, 0, 1, 1],
              [0, 1, 0, 1]])
Y = np.array([[0, 1, 1, 0]])

parameters, losses = train_neural_network(
    X, Y,
    hidden_size=4,
    learning_rate=0.5,
    epochs=10000
)

# 测试
A2, _ = forward_propagation(X, parameters)
print("\n输入:")
print(X.T)
print("\n预测:")
print(A2.T)
print("\n真实标签:")
print(Y.T)
```

**输出**：
```
Epoch 0: Loss = 0.250615
Epoch 1000: Loss = 0.062439
Epoch 2000: Loss = 0.013152
Epoch 3000: Loss = 0.005862
Epoch 4000: Loss = 0.003494
Epoch 5000: Loss = 0.002388
Epoch 6000: Loss = 0.001756
Epoch 7000: Loss = 0.001353
Epoch 8000: Loss = 0.001081
Epoch 9000: Loss = 0.000887

输入:
[[0 0]
 [0 1]
 [1 0]
 [1 1]]

预测:
[[0.02458917]
 [0.97201347]
 [0.97412658]
 [0.02907213]]

真实标签:
[[0]
 [1]
 [1]
 [0]]
```

成功解决了 XOR 问题！🎉

---

## 4.6 初始化策略

### 🤔 为什么初始化重要？

**全零初始化**：
```python
W = np.zeros((n_out, n_in))
```

**问题**：所有神经元学到相同的特征（对称性问题）

**随机初始化**：
```python
W = np.random.randn(n_out, n_in)
```

**问题**：方差可能太大或太小

### ✅ 好的初始化方法

#### **1. Xavier 初始化** (Glorot)

**适用于**：Sigmoid / Tanh

```python
W = np.random.randn(n_out, n_in) * np.sqrt(1 / n_in)

# 或

W = np.random.randn(n_out, n_in) * np.sqrt(2 / (n_in + n_out))
```

**原理**：保持方差在各层之间平衡

#### **2. He 初始化**

**适用于**：ReLU 及其变种

```python
W = np.random.randn(n_out, n_in) * np.sqrt(2 / n_in)
```

**原理**：考虑 ReLU 会"杀死"一半神经元

### 💻 实现

```python
def initialize_parameters_xavier(layer_dims):
    """
    Xavier 初始化

    参数：
        layer_dims: 列表，每层的神经元数量
                   例如 [784, 128, 64, 10]
    """
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(
            layer_dims[l],
            layer_dims[l-1]
        ) * np.sqrt(1 / layer_dims[l-1])

        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))

    return parameters

def initialize_parameters_he(layer_dims):
    """
    He 初始化
    """
    parameters = {}
    L = len(layer_dims)

    for l in range(1, L):
        parameters[f'W{l}'] = np.random.randn(
            layer_dims[l],
            layer_dims[l-1]
        ) * np.sqrt(2 / layer_dims[l-1])

        parameters[f'b{l}'] = np.zeros((layer_dims[l], 1))

    return parameters
```

---

## 4.7 梯度检验 (Gradient Checking)

### 🎯 目的

验证反向传播的实现是否正确

### 📐 数值梯度

```
f'(θ) ≈ [f(θ + ε) - f(θ - ε)] / (2ε)

通常 ε = 10⁻⁷
```

### 💻 实现

```python
def gradient_check(parameters, gradients, X, Y, epsilon=1e-7):
    """
    梯度检验

    返回：
        difference: 数值梯度和解析梯度的相对差异
    """
    # 将参数展平为向量
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]

    # 计算数值梯度
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    for i in range(num_parameters):
        # 计算 J_plus[i]
        thetaplus = np.copy(parameters_values)
        thetaplus[i][0] += epsilon
        J_plus[i] = forward_propagation_cost(X, Y, vector_to_dictionary(thetaplus))

        # 计算 J_minus[i]
        thetaminus = np.copy(parameters_values)
        thetaminus[i][0] -= epsilon
        J_minus[i] = forward_propagation_cost(X, Y, vector_to_dictionary(thetaminus))

        # 计算数值梯度
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    # 计算相对差异
    numerator = np.linalg.norm(grad - gradapprox)
    denominator = np.linalg.norm(grad) + np.linalg.norm(gradapprox)
    difference = numerator / denominator

    if difference > 2e-7:
        print(f"⚠️  梯度检验失败！差异 = {difference}")
    else:
        print(f"✅ 梯度检验通过！差异 = {difference}")

    return difference
```

---

## 4.8 使用深度学习框架

### 🔥 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# 1. 定义网络
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.sigmoid(x)
        return x

# 2. 准备数据
X_train = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=torch.float32)
y_train = torch.tensor([[0], [1], [1], [0]], dtype=torch.float32)

# 3. 创建模型
model = NeuralNetwork(input_size=2, hidden_size=4, output_size=1)

# 4. 定义损失函数和优化器
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.1)

# 5. 训练
epochs = 5000
for epoch in range(epochs):
    # 前向传播
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # 反向传播
    optimizer.zero_grad()  # 清零梯度
    loss.backward()        # 计算梯度
    optimizer.step()       # 更新参数

    if epoch % 500 == 0:
        print(f'Epoch [{epoch}/{epochs}], Loss: {loss.item():.6f}')

# 6. 测试
with torch.no_grad():
    predictions = model(X_train)
    print("\n预测结果:")
    print(predictions.numpy())
```

### 🌐 TensorFlow/Keras 实现

```python
import tensorflow as tf
from tensorflow import keras
import numpy as np

# 1. 准备数据
X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_train = np.array([[0], [1], [1], [0]])

# 2. 构建模型
model = keras.Sequential([
    keras.layers.Dense(4, activation='relu', input_shape=(2,)),
    keras.layers.Dense(1, activation='sigmoid')
])

# 3. 编译模型
model.compile(
    optimizer='adam',
    loss='mse',
    metrics=['accuracy']
)

# 4. 训练
history = model.fit(
    X_train, y_train,
    epochs=5000,
    verbose=0  # 不打印训练过程
)

# 5. 评估
loss, accuracy = model.evaluate(X_train, y_train, verbose=0)
print(f'Loss: {loss:.6f}')

# 6. 预测
predictions = model.predict(X_train)
print("\n预测结果:")
print(predictions)

# 7. 查看模型结构
model.summary()
```

---

## 4.9 实战：MNIST 手写数字识别（深度网络）

### 💻 完整实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# 1. 数据加载和预处理
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # MNIST 的均值和标准差
])

train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

test_dataset = datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 2. 定义深度神经网络
class DeepNN(nn.Module):
    def __init__(self):
        super(DeepNN, self).__init__()
        self.fc1 = nn.Linear(28*28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 10)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x):
        x = x.view(-1, 28*28)  # 展平

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = self.relu(x)
        x = self.dropout(x)

        x = self.fc4(x)
        return x

# 3. 初始化模型
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = DeepNN().to(device)

# 4. 定义损失和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()

    train_loss /= len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)

    print(f'Epoch: {epoch}, Loss: {train_loss:.4f}, Accuracy: {accuracy:.2f}%')
    return train_loss, accuracy

# 6. 测试函数
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader)
    accuracy = 100. * correct / len(test_loader.dataset)

    print(f'Test Loss: {test_loss:.4f}, Accuracy: {accuracy:.2f}%\n')
    return test_loss, accuracy

# 7. 训练模型
epochs = 10
train_losses, train_accs = [], []
test_losses, test_accs = [], []

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
    test_loss, test_acc = test(model, device, test_loader)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

# 8. 可视化训练过程
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

ax1.plot(train_losses, label='Train Loss')
ax1.plot(test_losses, label='Test Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Test Loss')
ax1.legend()
ax1.grid(True)

ax2.plot(train_accs, label='Train Accuracy')
ax2.plot(test_accs, label='Test Accuracy')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Test Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()

# 9. 可视化一些预测结果
model.eval()
with torch.no_grad():
    data, target = next(iter(test_loader))
    data, target = data.to(device), target.to(device)
    output = model(data)
    pred = output.argmax(dim=1)

    # 显示前16个样本
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        image = data[i].cpu().squeeze()
        true_label = target[i].item()
        pred_label = pred[i].item()

        ax.imshow(image, cmap='gray')
        color = 'green' if true_label == pred_label else 'red'
        ax.set_title(f'True: {true_label}, Pred: {pred_label}', color=color)
        ax.axis('off')

    plt.tight_layout()
    plt.show()
```

---

## 📝 本章作业

### 作业 1：理论题

1. **激活函数选择**
   - 为什么 ReLU 比 Sigmoid 更常用？
   - 什么是 "Dead ReLU" 问题？如何解决？
   - 在什么情况下仍然使用 Sigmoid？

2. **反向传播理解**
   - 用自己的话解释反向传播
   - 为什么叫"反向"传播？
   - 画出一个3层网络的计算图

3. **初始化策略**
   - 为什么不能全零初始化？
   - Xavier 和 He 初始化的区别？
   - 偏置需要特殊初始化吗？

### 作业 2：编程实践

#### 任务 1：从零实现多层网络

```python
# 实现一个L层全连接网络
class DeepNeuralNetwork:
    def __init__(self, layer_dims):
        """
        layer_dims: 每层的神经元数量
                   例如 [784, 128, 64, 10]
        """
        pass

    def forward(self, X):
        """前向传播"""
        pass

    def backward(self, X, Y):
        """反向传播"""
        pass

    def train(self, X, Y, epochs, learning_rate):
        """训练"""
        pass

# TODO:
# 1. 支持任意层数
# 2. 支持不同激活函数（ReLU, Sigmoid, Tanh）
# 3. 实现梯度检验
# 4. 在 MNIST 上测试
```

#### 任务 2：激活函数对比实验

```python
# 在相同数据集上对比不同激活函数的效果
# 1. Sigmoid
# 2. Tanh
# 3. ReLU
# 4. Leaky ReLU

# 记录：
# - 训练速度（达到90%准确率需要的epoch数）
# - 最终准确率
# - 训练稳定性

# 画出训练曲线对比图
```

#### 任务 3：深度网络实验

```python
# 在 Fashion-MNIST 上对比不同深度的网络
# 1. 2层：784 → 128 → 10
# 2. 3层：784 → 256 → 128 → 10
# 3. 4层：784 → 512 → 256 → 128 → 10
# 4. 5层：784 → 512 → 256 → 128 → 64 → 10

# 分析：
# - 是否越深越好？
# - 观察梯度消失/爆炸现象
# - 尝试不同的初始化方法
```

---

## 🔑 本章关键概念

| 概念 | 说明 |
|------|------|
| 神经网络 | 多层感知机的堆叠 |
| 前向传播 | 从输入计算到输出 |
| 反向传播 | 用链式法则计算梯度 |
| 激活函数 | 引入非线性 |
| ReLU | 最常用的激活函数 |
| Sigmoid | 输出层（二元分类） |
| Softmax | 输出层（多元分类） |
| Xavier初始化 | 适用于 Sigmoid/Tanh |
| He初始化 | 适用于 ReLU |
| 梯度检验 | 验证反向传播正确性 |

---

## 🎯 下一章预告

**第五章：优化算法与训练技巧**
- Mini-batch 梯度下降
- 动量 (Momentum)
- RMSprop, Adam
- Learning Rate Scheduling
- Batch Normalization
- Dropout
- 正则化技术
- 调参技巧

---

-----

