# 第五章：优化算法与训练技巧 (Optimization & Training Tricks)

## 📌 章节目标
- 理解不同梯度下降变种的原理
- 掌握现代优化算法（Momentum、Adam等）
- 学习正则化技术防止过拟合
- 了解批归一化和学习率调度
- 掌握深度学习训练的实用技巧

---

## 5.1 梯度下降的变种

### 🔄 三种梯度下降

#### **1. Batch Gradient Descent (批量梯度下降)**

**每次使用全部训练数据**

```python
for epoch in range(epochs):
    # 使用所有数据计算梯度
    gradients = compute_gradients(X_train, y_train, parameters)
    parameters = update_parameters(parameters, gradients, lr)
```

**优点**：
- ✅ 收敛稳定
- ✅ 可以利用矩阵运算加速

**缺点**：
- ❌ 数据量大时计算慢
- ❌ 内存占用大
- ❌ 容易陷入局部最优

---

#### **2. Stochastic Gradient Descent (随机梯度下降，SGD)**

**每次使用一个样本**

```python
for epoch in range(epochs):
    # 随机打乱数据
    indices = np.random.permutation(len(X_train))

    for i in indices:
        # 每次用一个样本
        gradients = compute_gradients(X_train[i:i+1], y_train[i:i+1], parameters)
        parameters = update_parameters(parameters, gradients, lr)
```

**优点**：
- ✅ 更新频繁，收敛快
- ✅ 可以逃离局部最优
- ✅ 可以在线学习

**缺点**：
- ❌ 波动大，不稳定
- ❌ 难以并行化
- ❌ 可能不收敛

**Loss 曲线对比**：

```
Batch GD:        SGD:
Loss             Loss
 ↓                ↓
 |\                |  *
 | \               | * *
 |  \              |*   *
 |   \___          | *   *
 |_______→        |___*___→
   Epoch            Epoch
 (平滑下降)        (震荡下降)
```

---

#### **3. Mini-batch Gradient Descent (小批量梯度下降)** ⭐

**每次使用一小批数据（通常 32-256）**

```python
batch_size = 64

for epoch in range(epochs):
    # 随机打乱
    indices = np.random.permutation(len(X_train))

    # 分批处理
    for i in range(0, len(X_train), batch_size):
        batch_indices = indices[i:i+batch_size]
        X_batch = X_train[batch_indices]
        y_batch = y_train[batch_indices]

        # 计算梯度和更新
        gradients = compute_gradients(X_batch, y_batch, parameters)
        parameters = update_parameters(parameters, gradients, lr)
```

**优点**：
- ✅ 平衡了速度和稳定性
- ✅ 可以利用 GPU 并行
- ✅ 更好的泛化能力

**缺点**：
- 需要调整 batch_size

**Batch Size 选择**：

```
小 batch (16-32):
  + 泛化能力强
  + 适合小数据集
  - 训练不稳定
  - 速度慢

大 batch (256-512):
  + 训练稳定
  + 充分利用 GPU
  + 速度快
  - 可能过拟合
  - 泛化能力较弱

常用: 32, 64, 128
```

---

### 💻 实现 Mini-batch

```python
def create_mini_batches(X, y, batch_size):
    """
    创建 mini-batches

    返回：
        mini_batches: 列表，每个元素是 (X_batch, y_batch)
    """
    m = X.shape[0]
    mini_batches = []

    # 随机打乱
    permutation = np.random.permutation(m)
    shuffled_X = X[permutation]
    shuffled_y = y[permutation]

    # 分批
    num_complete_batches = m // batch_size

    for k in range(num_complete_batches):
        X_batch = shuffled_X[k*batch_size:(k+1)*batch_size]
        y_batch = shuffled_y[k*batch_size:(k+1)*batch_size]
        mini_batches.append((X_batch, y_batch))

    # 处理剩余的数据
    if m % batch_size != 0:
        X_batch = shuffled_X[num_complete_batches*batch_size:]
        y_batch = shuffled_y[num_complete_batches*batch_size:]
        mini_batches.append((X_batch, y_batch))

    return mini_batches

# 使用
def train_with_mini_batch(X, y, parameters, epochs, batch_size, learning_rate):
    """使用 mini-batch 训练"""
    losses = []

    for epoch in range(epochs):
        epoch_loss = 0
        mini_batches = create_mini_batches(X, y, batch_size)

        for X_batch, y_batch in mini_batches:
            # 前向传播
            y_pred, cache = forward_propagation(X_batch, parameters)

            # 计算损失
            loss = compute_loss(y_pred, y_batch)
            epoch_loss += loss

            # 反向传播
            gradients = backward_propagation(X_batch, y_batch, parameters, cache)

            # 更新参数
            parameters = update_parameters(parameters, gradients, learning_rate)

        # 平均损失
        avg_loss = epoch_loss / len(mini_batches)
        losses.append(avg_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

    return parameters, losses
```

---

## 5.2 动量法 (Momentum)

### 🎯 问题：梯度下降的震荡

```
Loss
 ↑
 |     *
 |    * *
 |   *   *     ← 垂直方向震荡
 |  *     *
 | *       *
 |__________→
   参数空间

理想：横向快速前进，纵向减少震荡
```

### 💡 动量法原理

**物理类比**：滚下山的小球

```
小球不会立刻改变方向
而是累积动量，平滑地滚动
```

**数学公式**：

```
v_t = β·v_{t-1} + (1-β)·∇L_t

θ_t = θ_{t-1} - α·v_t

其中：
  v_t: 速度（动量）
  β: 动量系数（通常 0.9）
  ∇L_t: 当前梯度
  α: 学习率
```

**指数加权移动平均**：

```
v_t = β·v_{t-1} + (1-β)·g_t
    = (1-β)·g_t + β·(1-β)·g_{t-1} + β²·(1-β)·g_{t-2} + ...

权重：
  g_t:   (1-β) = 0.1
  g_{t-1}: β(1-β) = 0.09
  g_{t-2}: β²(1-β) = 0.081
  ...

越近的梯度权重越大
```

### 📊 效果对比

```
不使用动量:        使用动量:
    *                 ──→
   * *               ──→
  *   *             ──→
 *     *           ──→
*震荡  *          平滑
```

### 💻 实现

```python
def initialize_momentum(parameters):
    """
    初始化动量

    返回：
        v: 字典，与 parameters 结构相同，初始化为 0
    """
    v = {}
    L = len(parameters) // 2  # W 和 b 的对数

    for l in range(1, L + 1):
        v[f'dW{l}'] = np.zeros_like(parameters[f'W{l}'])
        v[f'db{l}'] = np.zeros_like(parameters[f'b{l}'])

    return v

def update_parameters_with_momentum(parameters, gradients, v, learning_rate, beta=0.9):
    """
    使用动量更新参数

    参数：
        parameters: 当前参数
        gradients: 梯度
        v: 动量
        learning_rate: 学习率
        beta: 动量系数
    """
    L = len(parameters) // 2

    for l in range(1, L + 1):
        # 更新动量
        v[f'dW{l}'] = beta * v[f'dW{l}'] + (1 - beta) * gradients[f'dW{l}']
        v[f'db{l}'] = beta * v[f'db{l}'] + (1 - beta) * gradients[f'db{l}']

        # 更新参数
        parameters[f'W{l}'] -= learning_rate * v[f'dW{l}']
        parameters[f'b{l}'] -= learning_rate * v[f'db{l}']

    return parameters, v

# 使用示例
v = initialize_momentum(parameters)

for epoch in range(epochs):
    for X_batch, y_batch in mini_batches:
        # 前向传播和反向传播
        y_pred, cache = forward_propagation(X_batch, parameters)
        gradients = backward_propagation(X_batch, y_batch, parameters, cache)

        # 使用动量更新
        parameters, v = update_parameters_with_momentum(
            parameters, gradients, v, learning_rate, beta=0.9
        )
```

---

## 5.3 RMSprop (Root Mean Square Propagation)

### 🎯 问题：不同参数需要不同学习率

```
参数 w₁: 梯度范围 [-100, 100]  ← 需要小学习率
参数 w₂: 梯度范围 [-0.01, 0.01] ← 需要大学习率

固定学习率无法同时满足
```

### 💡 RMSprop 原理

**自适应调整每个参数的学习率**

```
s_t = β·s_{t-1} + (1-β)·(∇L_t)²

θ_t = θ_{t-1} - α·∇L_t / √(s_t + ε)

其中：
  s_t: 梯度平方的指数加权移动平均
  β: 衰减率（通常 0.999）
  ε: 防止除零（通常 10⁻⁸）
```

**直觉**：
- 梯度大 → s 大 → 步长小（除以大数）
- 梯度小 → s 小 → 步长大（除以小数）

### 💻 实现

```python
def initialize_rmsprop(parameters):
    """初始化 RMSprop"""
    s = {}
    L = len(parameters) // 2

    for l in range(1, L + 1):
        s[f'dW{l}'] = np.zeros_like(parameters[f'W{l}'])
        s[f'db{l}'] = np.zeros_like(parameters[f'b{l}'])

    return s

def update_parameters_with_rmsprop(parameters, gradients, s, learning_rate,
                                   beta=0.999, epsilon=1e-8):
    """使用 RMSprop 更新参数"""
    L = len(parameters) // 2

    for l in range(1, L + 1):
        # 更新平方梯度的移动平均
        s[f'dW{l}'] = beta * s[f'dW{l}'] + (1 - beta) * gradients[f'dW{l}']**2
        s[f'db{l}'] = beta * s[f'db{l}'] + (1 - beta) * gradients[f'db{l}']**2

        # 更新参数
        parameters[f'W{l}'] -= learning_rate * gradients[f'dW{l}'] / (np.sqrt(s[f'dW{l}']) + epsilon)
        parameters[f'b{l}'] -= learning_rate * gradients[f'db{l}'] / (np.sqrt(s[f'db{l}']) + epsilon)

    return parameters, s
```

---

## 5.4 Adam (Adaptive Moment Estimation) ⭐

### 🎯 Adam = Momentum + RMSprop

**结合两者优点**：
- Momentum：平滑梯度方向
- RMSprop：自适应学习率

### 📐 算法

```
初始化：
  v₀ = 0  (一阶矩估计，动量)
  s₀ = 0  (二阶矩估计，RMSprop)

每次迭代：
  1. 计算梯度 g_t = ∇L_t

  2. 更新动量：
     v_t = β₁·v_{t-1} + (1-β₁)·g_t

  3. 更新平方梯度：
     s_t = β₂·s_{t-1} + (1-β₂)·g_t²

  4. 偏差修正：
     v̂_t = v_t / (1 - β₁ᵗ)
     ŝ_t = s_t / (1 - β₂ᵗ)

  5. 更新参数：
     θ_t = θ_{t-1} - α·v̂_t / (√ŝ_t + ε)

默认超参数：
  α = 0.001
  β₁ = 0.9
  β₂ = 0.999
  ε = 10⁻⁸
```

### 🤔 为什么需要偏差修正？

```
初始时 v₀ = 0, s₀ = 0

第一步：
  v₁ = 0.9·0 + 0.1·g₁ = 0.1·g₁

问题：v₁ 远小于真实期望！
  (因为初始化为 0，有偏差)

修正：
  v̂₁ = v₁ / (1 - 0.9¹) = 0.1·g₁ / 0.1 = g₁  ✓

随着 t 增大：
  (1 - β₁ᵗ) → 1
  修正效果逐渐消失
```

### 💻 完整实现

```python
def initialize_adam(parameters):
    """
    初始化 Adam 优化器

    返回：
        v: 一阶矩估计（动量）
        s: 二阶矩估计（RMSprop）
    """
    v = {}
    s = {}
    L = len(parameters) // 2

    for l in range(1, L + 1):
        v[f'dW{l}'] = np.zeros_like(parameters[f'W{l}'])
        v[f'db{l}'] = np.zeros_like(parameters[f'b{l}'])
        s[f'dW{l}'] = np.zeros_like(parameters[f'W{l}'])
        s[f'db{l}'] = np.zeros_like(parameters[f'b{l}'])

    return v, s

def update_parameters_with_adam(parameters, gradients, v, s, t,
                                learning_rate=0.001,
                                beta1=0.9, beta2=0.999, epsilon=1e-8):
    """
    使用 Adam 更新参数

    参数：
        t: 当前迭代次数（从 1 开始）
    """
    L = len(parameters) // 2
    v_corrected = {}
    s_corrected = {}

    for l in range(1, L + 1):
        # 更新动量
        v[f'dW{l}'] = beta1 * v[f'dW{l}'] + (1 - beta1) * gradients[f'dW{l}']
        v[f'db{l}'] = beta1 * v[f'db{l}'] + (1 - beta1) * gradients[f'db{l}']

        # 更新平方梯度
        s[f'dW{l}'] = beta2 * s[f'dW{l}'] + (1 - beta2) * (gradients[f'dW{l}']**2)
        s[f'db{l}'] = beta2 * s[f'db{l}'] + (1 - beta2) * (gradients[f'db{l}']**2)

        # 偏差修正
        v_corrected[f'dW{l}'] = v[f'dW{l}'] / (1 - beta1**t)
        v_corrected[f'db{l}'] = v[f'db{l}'] / (1 - beta1**t)
        s_corrected[f'dW{l}'] = s[f'dW{l}'] / (1 - beta2**t)
        s_corrected[f'db{l}'] = s[f'db{l}'] / (1 - beta2**t)

        # 更新参数
        parameters[f'W{l}'] -= learning_rate * v_corrected[f'dW{l}'] / (np.sqrt(s_corrected[f'dW{l}']) + epsilon)
        parameters[f'b{l}'] -= learning_rate * v_corrected[f'db{l}'] / (np.sqrt(s_corrected[f'db{l}']) + epsilon)

    return parameters, v, s

# 使用示例
def train_with_adam(X_train, y_train, layer_dims, epochs=1000, batch_size=64):
    """使用 Adam 训练网络"""
    # 初始化参数
    parameters = initialize_parameters(layer_dims)
    v, s = initialize_adam(parameters)

    losses = []
    t = 0  # 全局迭代计数器

    for epoch in range(epochs):
        mini_batches = create_mini_batches(X_train, y_train, batch_size)
        epoch_loss = 0

        for X_batch, y_batch in mini_batches:
            t += 1  # 每个 mini-batch 增加计数

            # 前向传播
            AL, caches = forward_propagation_deep(X_batch, parameters)

            # 计算损失
            loss = compute_loss(AL, y_batch)
            epoch_loss += loss

            # 反向传播
            gradients = backward_propagation_deep(AL, y_batch, caches)

            # Adam 更新
            parameters, v, s = update_parameters_with_adam(
                parameters, gradients, v, s, t
            )

        avg_loss = epoch_loss / len(mini_batches)
        losses.append(avg_loss)

        if epoch % 100 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss:.6f}")

    return parameters, losses
```

---

## 5.5 优化器对比

### 📊 可视化对比

```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 定义一个非凸函数（类似 Rosenbrock）
def f(x, y):
    return (1 - x)**2 + 100 * (y - x**2)**2

def grad_f(x, y):
    """计算梯度"""
    dx = -2 * (1 - x) - 400 * x * (y - x**2)
    dy = 200 * (y - x**2)
    return np.array([dx, dy])

# 创建等高线图
x = np.linspace(-2, 2, 100)
y = np.linspace(-1, 3, 100)
X, Y = np.meshgrid(x, y)
Z = f(X, Y)

# 测试不同优化器
def sgd(pos, grad, lr=0.001):
    return pos - lr * grad

def momentum(pos, grad, v, beta=0.9, lr=0.001):
    v = beta * v + (1 - beta) * grad
    return pos - lr * v, v

def rmsprop(pos, grad, s, beta=0.999, lr=0.001, eps=1e-8):
    s = beta * s + (1 - beta) * grad**2
    return pos - lr * grad / (np.sqrt(s) + eps), s

def adam(pos, grad, v, s, t, beta1=0.9, beta2=0.999, lr=0.001, eps=1e-8):
    v = beta1 * v + (1 - beta1) * grad
    s = beta2 * s + (1 - beta2) * grad**2
    v_hat = v / (1 - beta1**t)
    s_hat = s / (1 - beta2**t)
    return pos - lr * v_hat / (np.sqrt(s_hat) + eps), v, s

# 运行优化
def run_optimizer(optimizer_name, steps=200):
    pos = np.array([-1.5, 2.5])
    trajectory = [pos.copy()]

    if optimizer_name == 'SGD':
        for _ in range(steps):
            grad = grad_f(pos[0], pos[1])
            pos = sgd(pos, grad)
            trajectory.append(pos.copy())

    elif optimizer_name == 'Momentum':
        v = np.zeros(2)
        for _ in range(steps):
            grad = grad_f(pos[0], pos[1])
            pos, v = momentum(pos, grad, v)
            trajectory.append(pos.copy())

    elif optimizer_name == 'RMSprop':
        s = np.zeros(2)
        for _ in range(steps):
            grad = grad_f(pos[0], pos[1])
            pos, s = rmsprop(pos, grad, s)
            trajectory.append(pos.copy())

    elif optimizer_name == 'Adam':
        v = np.zeros(2)
        s = np.zeros(2)
        for t in range(1, steps + 1):
            grad = grad_f(pos[0], pos[1])
            pos, v, s = adam(pos, grad, v, s, t)
            trajectory.append(pos.copy())

    return np.array(trajectory)

# 绘制对比图
fig, axes = plt.subplots(2, 2, figsize=(14, 12))
optimizers = ['SGD', 'Momentum', 'RMSprop', 'Adam']

for ax, opt_name in zip(axes.flat, optimizers):
    ax.contour(X, Y, Z, levels=np.logspace(-1, 3, 20), cmap='viridis', alpha=0.3)

    trajectory = run_optimizer(opt_name, steps=200)
    ax.plot(trajectory[:, 0], trajectory[:, 1], 'r-', linewidth=2, alpha=0.7)
    ax.plot(trajectory[0, 0], trajectory[0, 1], 'go', markersize=10, label='起点')
    ax.plot(1, 1, 'r*', markersize=15, label='最优点')

    ax.set_title(f'{opt_name} Optimizer', fontsize=14, fontweight='bold')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 📈 性能对比表

| 优化器 | 收敛速度 | 稳定性 | 内存开销 | 超参数敏感度 | 推荐度 |
|--------|---------|--------|---------|------------|--------|
| SGD | ★★☆☆☆ | ★★☆☆☆ | ★★★★★ | ★★★★☆ | ★★☆☆☆ |
| SGD+Momentum | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★★★☆ |
| RMSprop | ★★★★☆ | ★★★★☆ | ★★★★☆ | ★★★☆☆ | ★★★★☆ |
| Adam | ★★★★★ | ★★★★★ | ★★★☆☆ | ★★★★★ | ★★★★★ |
| AdaGrad | ★★★☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★☆ | ★★☆☆☆ |
| AdamW | ★★★★★ | ★★★★★ | ★★★☆☆ | ★★★★★ | ★★★★★ |

### 💡 选择指南

```
默认选择：Adam ⭐
  - 几乎适用于所有场景
  - 不需要太多调参
  - 收敛快且稳定

需要最佳性能：SGD + Momentum
  - 训练时间足够时
  - 配合 Learning Rate Schedule
  - 通常泛化能力更好

计算机视觉：SGD + Momentum
  - ResNet, VGG 等经典模型
  - 需要仔细调整学习率

NLP / Transformer：Adam / AdamW
  - BERT, GPT 标配
  - AdamW 加入权重衰减

内存受限：SGD
  - 不需要额外存储动量
```

---

## 5.6 学习率调度 (Learning Rate Scheduling)

### 🎯 为什么需要调整学习率？

```
训练初期：
  - 离最优点远
  - 可以用大学习率快速接近

训练后期：
  - 接近最优点
  - 需要小学习率精细调整

固定学习率：
  太大 → 震荡，不收敛
  太小 → 训练慢
```

### 📊 常见调度策略

#### **1. Step Decay (阶梯衰减)**

```
每隔固定 epoch，学习率乘以衰减因子

lr_t = lr_0 · γ^⌊epoch/step_size⌋

例：
  lr_0 = 0.1
  γ = 0.1
  step_size = 30

  epoch 0-29:  lr = 0.1
  epoch 30-59: lr = 0.01
  epoch 60-89: lr = 0.001
```

```python
def step_decay_schedule(epoch, lr, drop=0.5, epochs_drop=10):
    """阶梯衰减"""
    return lr * (drop ** (epoch // epochs_drop))
```

#### **2. Exponential Decay (指数衰减)**

```
lr_t = lr_0 · e^(-λt)

或

lr_t = lr_0 · γ^t
```

```python
def exponential_decay(epoch, lr_0, decay_rate=0.96):
    """指数衰减"""
    return lr_0 * np.exp(-decay_rate * epoch)
```

#### **3. Cosine Annealing (余弦退火)**

```
lr_t = lr_min + (lr_max - lr_min) · (1 + cos(πt/T)) / 2

平滑下降，常用于训练后期 fine-tune
```

```python
def cosine_annealing(epoch, lr_max, lr_min, T_max):
    """余弦退火"""
    return lr_min + (lr_max - lr_min) * (1 + np.cos(np.pi * epoch / T_max)) / 2
```

#### **4. Warm-up + Cosine (现代 Transformer 标配)**

```
Warm-up 阶段（前几个 epoch）：
  线性增加学习率 0 → lr_max

主训练阶段：
  余弦退火 lr_max → lr_min
```

```python
def warmup_cosine_schedule(epoch, lr_max, warmup_epochs, total_epochs, lr_min=0):
    """Warm-up + Cosine"""
    if epoch < warmup_epochs:
        # Warm-up 阶段
        return lr_max * (epoch + 1) / warmup_epochs
    else:
        # Cosine 阶段
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return lr_min + (lr_max - lr_min) * 0.5 * (1 + np.cos(np.pi * progress))
```

#### **5. Reduce on Plateau (基于验证集)**

```
监控验证集 Loss：

如果 N 个 epoch 没有改进：
  lr = lr * factor

很实用！
```

```python
class ReduceLROnPlateau:
    def __init__(self, lr_init, factor=0.1, patience=10, min_lr=1e-7):
        self.lr = lr_init
        self.factor = factor
        self.patience = patience
        self.min_lr = min_lr
        self.best_loss = float('inf')
        self.counter = 0

    def step(self, val_loss):
        """
        根据验证集 Loss 调整学习率
        """
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.lr *= self.factor
                self.lr = max(self.lr, self.min_lr)
                self.counter = 0
                print(f"学习率已降低到 {self.lr:.6f}")

        return self.lr
```

### 📈 可视化对比

```python
import matplotlib.pyplot as plt
import numpy as np

epochs = 200
lr_0 = 0.1

# 生成不同调度的学习率
epochs_arr = np.arange(epochs)

lr_constant = np.ones(epochs) * lr_0
lr_step = np.array([step_decay_schedule(e, lr_0, drop=0.5, epochs_drop=50)
                    for e in epochs_arr])
lr_exp = np.array([exponential_decay(e, lr_0, decay_rate=0.02)
                   for e in epochs_arr])
lr_cosine = np.array([cosine_annealing(e, lr_0, 1e-5, 200)
                      for e in epochs_arr])
lr_warmup_cosine = np.array([warmup_cosine_schedule(e, lr_0, 10, 200, 1e-5)
                             for e in epochs_arr])

# 绘图
plt.figure(figsize=(12, 6))
plt.semilogy(epochs_arr, lr_constant, label='Constant', linewidth=2)
plt.semilogy(epochs_arr, lr_step, label='Step Decay', linewidth=2)
plt.semilogy(epochs_arr, lr_exp, label='Exponential Decay', linewidth=2)
plt.semilogy(epochs_arr, lr_cosine, label='Cosine Annealing', linewidth=2)
plt.semilogy(epochs_arr, lr_warmup_cosine, label='Warm-up + Cosine', linewidth=2)

plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedules Comparison')
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

---

## 5.7 正则化技术 (Regularization)

### 🎯 问题：过拟合

```
训练集 Loss: 0.01  ✓
测试集 Loss: 0.5   ✗

模型过度学习了训练数据的噪声
```

### 🔹 L1 和 L2 正则化

**L2 正则化（权重衰减）**：

```
L_total = L_origin + λ·(1/2)·Σw²

效果：倾向于让权重变小
```

```python
# PyTorch 中直接在优化器中指定
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.01)

# 或手动添加到 Loss
L_total = L_origin + weight_decay * sum(p**2 for p in model.parameters())
```

**L1 正则化**：

```
L_total = L_origin + λ·Σ|w|

效果：让一些权重变成 0（稀疏性）
```

---

### 🔹 Dropout ⭐

**核心思想**：训练时随机"关闭"一些神经元

```
训练时 (dropout = 0.5):
  x₁ ── w₁ ──┐
  x₂ ── ✗   ├─ z  (随机关闭部分连接)
  x₃ ── w₃ ──┘

预测时：
  使用所有连接，但权重乘以 (1-p)
  或使用 inverted dropout，训练时就调整
```

**Inverted Dropout** (推荐):

```
训练时：
  a_dropped = a / (1 - p)  with probability (1-p)
              0             with probability p

预测时：
  使用 a 直接（不需要调整）
```

**效果**：
- 防止共适应（co-adaptation）
- 减少过拟合
- 集成效果

```python
# PyTorch
class NeuralNetworkWithDropout(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_rate=0.5):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout1 = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.dropout2 = nn.Dropout(dropout_rate)
        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)  # 训练时随机关闭，预测时自动调整

        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)

        x = self.fc3(x)
        return x

# 使用
model = NeuralNetworkWithDropout(784, 128, dropout_rate=0.5)
model.train()   # 训练模式，Dropout 有效
model.eval()    # 评估模式，Dropout 无效
```

### 📊 Dropout 效果

```
不使用 Dropout:          使用 Dropout:
训练Loss ↓              训练Loss ↘
测试Loss ↗              测试Loss ↘

过拟合                  泛化更好
```

---

### 🔹 Early Stopping

**思想**：监控验证集，当验证集 Loss 停止改进时停止训练

```python
class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        self.restore_best_weights = restore_best_weights
        self.best_weights = None

    def __call__(self, val_loss, model):
        """
        返回 True 表示应该停止训练
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = model.state_dict().copy()
        else:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights and self.best_weights is not None:
                    model.load_state_dict(self.best_weights)
                return True  # 停止训练
        return False

# 使用
early_stopping = EarlyStopping(patience=10)

for epoch in range(num_epochs):
    # 训练
    train_loss = train_one_epoch()

    # 验证
    val_loss = validate()

    # 检查是否停止
    if early_stopping(val_loss, model):
        print(f"在第 {epoch} 个 epoch 停止")
        break
```

---

## 5.8 Batch Normalization (BN)

### 🎯 为什么需要 BN？

**问题：内部协变量转移 (Internal Covariate Shift)**

```
第1层的输出变化 → 第2层的输入分布变化
→ 第2层需要不断适应 → 训练变慢
```

### 💡 Batch Normalization 原理

**标准化每个 batch**：

```
对每个特征：
  1. 计算均值：μ_B = (1/m)·Σxᵢ
  2. 计算方差：σ_B² = (1/m)·Σ(xᵢ - μ_B)²
  3. 标准化：x̂ᵢ = (xᵢ - μ_B) / √(σ_B² + ε)
  4. 尺度和平移：yᵢ = γ·x̂ᵢ + β

其中 γ 和 β 是可学习的参数
```

### 📍 BN 在哪里放？

```
常见位置：
  1. Linear → BN → Activation
  2. Linear → Activation → BN
  3. Conv → BN → ReLU (推荐)

一般：BN 在激活函数之前
```

### 💻 PyTorch 实现

```python
class NeuralNetworkWithBN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.bn2 = nn.BatchNorm1d(hidden_size)

        self.fc3 = nn.Linear(hidden_size, 10)

    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        x = self.fc3(x)
        return x

# 训练时 BN 使用 batch 统计
# 推理时 BN 使用运行均值和方差
model.train()    # 使用 batch 统计
model.eval()     # 使用运行统计
```

### ✅ Batch Normalization 的优点

1. **加速收敛**：减少内部协变量转移
2. **允许更大学习率**：更稳定的梯度
3. **减少初始化敏感性**：对初始化不敏感
4. **轻微的正则化效果**
5. **简化后续网络**：可以移除 Dropout

### ⚠️ BN 的缺点和限制

```
问题：
  - Batch size 太小时效果差
  - 训练和推理时行为不同
  - 在 RNN 中使用困难

解决：
  - Layer Normalization (LN)
  - Group Normalization (GN)
  - Instance Normalization (IN)
```

---

## 5.9 实战：完整训练流程

### 💻 PyTorch 完整示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

# ==================== 超参数 ====================
BATCH_SIZE = 128
LEARNING_RATE = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 50
DROPOUT_RATE = 0.3
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 数据加载 ====================
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))
])

train_dataset = datasets.MNIST(root='./data', train=True,
                               download=True, transform=transform)
val_dataset = datasets.MNIST(root='./data', train=False,
                             download=True, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                        shuffle=False, num_workers=2)

# ==================== 定义模型 ====================
class ImprovedNN(nn.Module):
    def __init__(self, dropout_rate=0.3):
        super().__init__()

        # 第1层
        self.fc1 = nn.Linear(28*28, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.dropout1 = nn.Dropout(dropout_rate)

        # 第2层
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(dropout_rate)

        # 第3层
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.dropout3 = nn.Dropout(dropout_rate)

        # 输出层
        self.fc4 = nn.Linear(128, 10)

    def forward(self, x):
        # 展平
        x = x.view(-1, 28*28)

        # 第1层
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)

        # 第2层
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)

        # 第3层
        x = self.fc3(x)
        x = self.bn3(x)
        x = torch.relu(x)
        x = self.dropout3(x)

        # 输出层
        x = self.fc4(x)
        return x

model = ImprovedNN(dropout_rate=DROPOUT_RATE).to(DEVICE)

# ==================== 损失和优化器 ====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(),
                       lr=LEARNING_RATE,
                       weight_decay=WEIGHT_DECAY)

# 学习率调度
scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer, T_max=EPOCHS, eta_min=1e-6
)

# Early Stopping
early_stopping = EarlyStopping(patience=10)

# ==================== 训练函数 ====================
def train_epoch(model, train_loader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc='Training')

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, labels)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 统计
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

        # 更新进度条
        accuracy = 100. * correct / total
        progress_bar.set_postfix({
            'loss': f'{total_loss/(total):.3f}',
            'acc': f'{accuracy:.2f}%'
        })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy

# ==================== 验证函数 ====================
def validate(model, val_loader, criterion, device):
    """验证"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(val_loader, desc='Validation'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    avg_loss = total_loss / len(val_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy

# ==================== 训练循环 ====================
train_losses = []
train_accs = []
val_losses = []
val_accs = []

for epoch in range(EPOCHS):
    print(f'\n=== Epoch {epoch+1}/{EPOCHS} ===')
    print(f'学习率: {optimizer.param_groups[0]["lr"]:.6f}')

    # 训练
    train_loss, train_acc = train_epoch(
        model, train_loader, criterion, optimizer, DEVICE
    )
    train_losses.append(train_loss)
    train_accs.append(train_acc)

    # 验证
    val_loss, val_acc = validate(model, val_loader, criterion, DEVICE)
    val_losses.append(val_loss)
    val_accs.append(val_acc)

    print(f'训练: Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
    print(f'验证: Loss={val_loss:.4f}, Acc={val_acc:.2f}%')

    # 学习率调度
    scheduler.step()

    # Early Stopping
    if early_stopping(val_loss, model):
        print(f'\n在第 {epoch+1} 个 epoch 停止训练（验证集无改进）')
        break

# ==================== 可视化 ====================
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

epochs_range = range(1, len(train_losses) + 1)

ax1.plot(epochs_range, train_losses, label='Train Loss', marker='o')
ax1.plot(epochs_range, val_losses, label='Val Loss', marker='o')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Loss Over Epochs')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, train_accs, label='Train Acc', marker='o')
ax2.plot(epochs_range, val_accs, label='Val Acc', marker='o')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Accuracy Over Epochs')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== 最终评估 ====================
print('\n' + '='*50)
print('最终结果：')
print(f'最佳验证准确率: {max(val_accs):.2f}% (Epoch {val_accs.index(max(val_accs))+1})')
print('='*50)
```

---

## 📝 本章作业

### 作业 1：优化器对比

实现以下优化器，在 MNIST 上对比：

```python
# TODO:
# 1. SGD
# 2. SGD + Momentum
# 3. RMSprop
# 4. Adam

# 记录：
#   - 达到 95% 准确率需要的 epoch 数
#   - 最终准确率
#   - 训练时间
#   - Loss 曲线平滑度

# 绘制对比图表
```

### 作业 2：学习率调度实验

```python
# 在同一个模型上对比不同的学习率调度：

# 1. 固定学习率
# 2. Step Decay
# 3. Exponential Decay
# 4. Cosine Annealing
# 5. Warm-up + Cosine

# 分析：
#   - 最终准确率
#   - 收敛速度
#   - 训练稳定性
```

### 作业 3：正则化技术对比

```python
# 在相同架构上对比：

# 1. 无正则化
# 2. L2 正则化 (λ=0.001, 0.01, 0.1)
# 3. Dropout (p=0.3, 0.5)
# 4. Batch Normalization
# 5. 组合（BN + Dropout）

# 观察：
#   - 训练集 vs 测试集性能差距
#   - 过拟合情况
#   - 每种方法的影响
```

### 作业 4：完整项目

在 CIFAR-10 数据集上构建完整的训练流程

**要求**：
1. 实现一个 4-5 层的深度网络
2. 使用 Batch Normalization
3. 使用 Dropout
4. 选择合适的优化器（Adam 或 SGD+Momentum）
5. 实现学习率调度
6. 实现 Early Stopping
7. 记录训练过程和评估结果
8. 可视化训练曲线
9. 分析模型性能和改进方向

---

## 🔑 本章关键概念

| 概念 | 说明 |
|------|------|
| Mini-batch GD | 批量梯度下降的折中 |
| Momentum | 使用历史梯度加速收敛 |
| RMSprop | 自适应学习率 |
| Adam | Momentum + RMSprop 的组合 |
| Learning Rate Schedule | 动态调整学习率 |
| Batch Normalization | 标准化中间层输出 |
| Dropout | 随机禁用神经元防止过拟合 |
| L1/L2 正则化 | 惩罚大权重 |
| Early Stopping | 监控验证集提前停止 |
| Weight Decay | 权重衰减（等同 L2） |

---

## 🎯 下一章预告

**第六章：卷积神经网络 (Convolutional Neural Networks)**
- 卷积操作的原理
- 感受野和参数共享
- 池化和特征图
- 经典 CNN 架构（LeNet, AlexNet, VGG, ResNet）
- 实战：图像分类

---
