# 第三章：分类与逻辑回归 (Classification & Logistic Regression)

## 📌 章节目标
- 理解分类问题与回归问题的区别
- 掌握逻辑回归的原理和实现
- 学习 Sigmoid 和 Softmax 函数
- 理解交叉熵损失函数
- 实战：手写数字识别、信用卡欺诈检测

---

## 3.1 什么是分类？

### 🎯 分类 vs 回归

```
回归 (Regression):
  预测连续数值
  例：房价 $350,000, 温度 25.3°C
  输出：实数 ℝ

分类 (Classification):
  预测离散类别
  例：猫/狗, 垃圾邮件/正常邮件
  输出：类别标签
```

### 📊 分类问题的类型

#### **1. 二元分类 (Binary Classification)**
只有两个类别

**例子**：
- 邮件：垃圾邮件 (1) / 正常邮件 (0)
- 医学：有病 (1) / 没病 (0)
- 信用卡：欺诈 (1) / 正常 (0)
- 客户：会购买 (1) / 不会购买 (0)

#### **2. 多元分类 (Multi-class Classification)**
多个类别（但只能属于一个）

**例子**：
- 手写数字识别：0, 1, 2, ..., 9
- 新闻分类：体育、政治、娱乐、科技
- 动物分类：猫、狗、鸟、鱼

#### **3. 多标签分类 (Multi-label Classification)**
可以同时属于多个类别

**例子**：
- 电影标签：[动作, 喜剧, 爱情]
- 文章标签：[机器学习, Python, 深度学习]

---

## 3.2 为什么不能用线性回归做分类？

### 🤔 尝试用回归做分类

假设我们要分类：猫 (0) / 狗 (1)

```
训练数据：
x (特征) | y (标签)
  1.0    |   0  (猫)
  2.0    |   0  (猫)
  3.0    |   1  (狗)
  4.0    |   1  (狗)
```

**线性回归**：`y = b + w·x`

```
y
↑
1 |         ●  ●   (狗)
  |       /
  |     /
0 |   ●  ●       (猫)
  |_____________→ x
```

看起来还不错？

### ⚠️ 问题来了！

**新数据点**：x = 10

```
y
↑
3 |               ●  (预测值 = 3？？)
2 |             /
1 |         ●  ●
  |       /
  |     /
0 |   ●  ●
  |_____________→ x
            10
```

**问题**：
1. 输出不是 0 或 1（可能是 3, -1, 0.7...）
2. 远离训练数据的点会影响决策边界
3. 无法表示概率

### 💡 我们需要什么？

```
理想的分类器：
  输出范围在 [0, 1]
  可以解释为概率
  0.9 → 90% 确定是狗
  0.1 → 10% 确定是狗（90% 是猫）
```

---

## 3.3 Logistic Regression

### 🔹 核心思想

**改造线性回归**：

```
Step 1: 计算线性组合
  z = b + w·x

Step 2: 通过 Sigmoid 函数压缩到 (0, 1)
  y = σ(z) = 1 / (1 + e^(-z))
```

### 📐 Sigmoid 函数

```python
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

z = np.linspace(-10, 10, 100)
y = sigmoid(z)

plt.figure(figsize=(10, 6))
plt.plot(z, y, 'b-', linewidth=2)
plt.axhline(y=0.5, color='r', linestyle='--', label='阈值 = 0.5')
plt.axvline(x=0, color='g', linestyle='--', alpha=0.5)
plt.xlabel('z = b + wx')
plt.ylabel('σ(z)')
plt.title('Sigmoid Function')
plt.grid(True, alpha=0.3)
plt.legend()
plt.show()
```

**图形**：
```
σ(z)
 1 |         ______
   |       /
0.5|      * (z=0, σ=0.5)
   |     /
 0 |____/
   |_____________→ z
  -10      0     10
```

### 🔍 Sigmoid 的性质

```
σ(z) = 1 / (1 + e^(-z))

性质：
1. 输出范围：(0, 1)
2. σ(0) = 0.5 (中点)
3. z → +∞, σ(z) → 1
4. z → -∞, σ(z) → 0
5. 关于 (0, 0.5) 中心对称
6. 导数：σ'(z) = σ(z)·(1 - σ(z))
```

### 🎯 决策规则

```
给定输入 x，计算：
  z = b + w·x
  P(y=1|x) = σ(z)

决策：
  如果 P(y=1|x) ≥ 0.5  → 预测为类别 1
  如果 P(y=1|x) < 0.5  → 预测为类别 0

等价于：
  如果 z ≥ 0  → 预测为类别 1
  如果 z < 0  → 预测为类别 0
```

### 📊 决策边界 (Decision Boundary)

**一维情况**：

```
z = b + w·x = 0
→ x = -b/w  (决策边界)

例：b = -3, w = 1
  x < 3 → 预测类别 0
  x > 3 → 预测类别 1
```

**二维情况**：

```
z = b + w₁x₁ + w₂x₂ = 0
→ x₂ = -(b + w₁x₁)/w₂  (一条直线)

x₂
↑
|     /
|    /  类别 1
|   /
|  /_________ 决策边界
| /
|/ 类别 0
|________→ x₁
```

---

## 3.4 Loss Function for Classification

### 🚫 为什么不用 MSE？

**尝试**：`L = (y - σ(z))²`

**问题**：
1. **非凸函数**：有很多局部最优解
2. **梯度消失**：当预测很错时，梯度反而很小

```
Loss
 ↑
 |  *     *
 | / \   / \   多个局部最优！
 |/   \_/   \
 |___________→ w
```

### ✅ Cross Entropy Loss

**公式**：

```
对于单个样本：
L(y, ŷ) = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

其中：
  y ∈ {0, 1}     真实标签
  ŷ = σ(z)       预测概率
```

**理解**：

```
情况1：真实标签 y = 1
  L = -log(ŷ)
  如果 ŷ → 1 (预测正确)  → L → 0   (损失小)
  如果 ŷ → 0 (预测错误)  → L → ∞   (损失大)

情况2：真实标签 y = 0
  L = -log(1-ŷ)
  如果 ŷ → 0 (预测正确)  → L → 0   (损失小)
  如果 ŷ → 1 (预测错误)  → L → ∞   (损失大)
```

### 📊 可视化 Cross Entropy

```python
import numpy as np
import matplotlib.pyplot as plt

y_pred = np.linspace(0.01, 0.99, 100)

# y = 1 时的损失
loss_y1 = -np.log(y_pred)

# y = 0 时的损失
loss_y0 = -np.log(1 - y_pred)

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(y_pred, loss_y1, 'b-', linewidth=2)
plt.title('真实标签 y = 1')
plt.xlabel('预测概率 ŷ')
plt.ylabel('Loss = -log(ŷ)')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(y_pred, loss_y0, 'r-', linewidth=2)
plt.title('真实标签 y = 0')
plt.xlabel('预测概率 ŷ')
plt.ylabel('Loss = -log(1-ŷ)')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
```

### 🧮 完整的 Loss Function

对于 N 个训练样本：

```
L(w, b) = -(1/N) Σ[yⁿ·log(ŷⁿ) + (1-yⁿ)·log(1-ŷⁿ)]

其中：
  ŷⁿ = σ(b + w·xⁿ)
```

---

## 3.5 梯度下降求解

### 📐 计算梯度

```
ŷ = σ(z) = σ(b + w·x)

L = -[y·log(ŷ) + (1-y)·log(1-ŷ)]

求导（链式法则）：

∂L/∂w = (ŷ - y)·x
∂L/∂b = (ŷ - y)

惊喜！形式和线性回归一样简单！
```

### 💻 从零实现 Logistic Regression

```python
import numpy as np

class LogisticRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.w = None
        self.b = None
        self.losses = []

    def sigmoid(self, z):
        """Sigmoid 函数"""
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        """训练模型"""
        n_samples, n_features = X.shape

        # 初始化参数
        self.w = np.zeros(n_features)
        self.b = 0

        # 梯度下降
        for epoch in range(self.epochs):
            # 前向传播
            z = np.dot(X, self.w) + self.b
            y_pred = self.sigmoid(z)

            # 计算损失
            loss = -np.mean(y * np.log(y_pred + 1e-9) +
                           (1 - y) * np.log(1 - y_pred + 1e-9))
            self.losses.append(loss)

            # 计算梯度
            dw = np.dot(X.T, (y_pred - y)) / n_samples
            db = np.mean(y_pred - y)

            # 更新参数
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # 打印进度
            if epoch % 100 == 0:
                print(f"Epoch {epoch}: Loss = {loss:.4f}")

    def predict_proba(self, X):
        """预测概率"""
        z = np.dot(X, self.w) + self.b
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        """预测类别"""
        proba = self.predict_proba(X)
        return (proba >= threshold).astype(int)

# 使用示例
if __name__ == "__main__":
    # 生成模拟数据
    np.random.seed(42)

    # 类别 0
    X0 = np.random.randn(100, 2) + np.array([2, 2])
    y0 = np.zeros(100)

    # 类别 1
    X1 = np.random.randn(100, 2) + np.array([5, 5])
    y1 = np.ones(100)

    # 合并
    X = np.vstack([X0, X1])
    y = np.hstack([y0, y1])

    # 训练
    model = LogisticRegression(learning_rate=0.1, epochs=1000)
    model.fit(X, y)

    # 预测
    predictions = model.predict(X)
    accuracy = np.mean(predictions == y)
    print(f"\n准确率: {accuracy:.4f}")
```

**输出**：
```
Epoch 0: Loss = 0.6931
Epoch 100: Loss = 0.2156
Epoch 200: Loss = 0.1398
Epoch 300: Loss = 0.1045
Epoch 400: Loss = 0.0850
Epoch 500: Loss = 0.0722
Epoch 600: Loss = 0.0632
Epoch 700: Loss = 0.0565
Epoch 800: Loss = 0.0513
Epoch 900: Loss = 0.0472

准确率: 1.0000
```

### 📊 可视化决策边界

```python
import matplotlib.pyplot as plt

def plot_decision_boundary(model, X, y):
    # 创建网格
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx1, xx2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 100),
        np.linspace(x2_min, x2_max, 100)
    )

    # 预测每个网格点
    Z = model.predict(np.c_[xx1.ravel(), xx2.ravel()])
    Z = Z.reshape(xx1.shape)

    # 绘图
    plt.figure(figsize=(10, 8))
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap='RdYlBu')
    plt.contour(xx1, xx2, Z, levels=[0.5], colors='black', linewidths=2)

    # 绘制数据点
    plt.scatter(X[y==0, 0], X[y==0, 1], c='blue', label='类别 0',
                edgecolors='k', s=50)
    plt.scatter(X[y==1, 0], X[y==1, 1], c='red', label='类别 1',
                edgecolors='k', s=50)

    plt.xlabel('特征 1')
    plt.ylabel('特征 2')
    plt.title('逻辑回归决策边界')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()

# 绘制
plot_decision_boundary(model, X, y)
```

---

## 3.6 评估分类模型

### 📊 Confusion Matrix (混淆矩阵)

```
                预测
              0      1
真   0      TN     FP
实   1      FN     TP

TN (True Negative):  正确预测为负类
TP (True Positive):  正确预测为正类
FN (False Negative): 错误预测为负类（漏报）
FP (False Positive): 错误预测为正类（误报）
```

**例子**：癌症检测

```
                预测
           没病    有病
真 没病    90      10    (10个假阳性)
实 有病     5      95    (5个假阴性)
```

### 🔢 评估指标

#### **1. Accuracy (准确率)**

```
Accuracy = (TP + TN) / (TP + TN + FP + FN)

= 正确预测的数量 / 总数量

例：(90 + 95) / 200 = 0.925 = 92.5%
```

**问题**：类别不平衡时会误导

```
例：100个样本，95个负类，5个正类
如果全部预测为负类：
  Accuracy = 95/100 = 95%  (看起来很好！)
  但完全没有检测到正类！
```

#### **2. Precision (精确率)**

```
Precision = TP / (TP + FP)

= 预测为正类中，真正是正类的比例

例：95 / (95 + 10) = 0.905 = 90.5%

理解：在我说"有病"的人中，真的有病的比例
```

**使用场景**：当 **误报代价高** 时
- 垃圾邮件过滤：不要把正常邮件标记为垃圾邮件
- 信用卡欺诈：不要误报正常交易

#### **3. Recall (召回率 / 灵敏度)**

```
Recall = TP / (TP + FN)

= 真正的正类中，被正确预测的比例

例：95 / (95 + 5) = 0.95 = 95%

理解：所有真正有病的人中，被检测出来的比例
```

**使用场景**：当 **漏报代价高** 时
- 疾病检测：不能漏掉真正的病人
- 欺诈检测：不能漏掉真正的欺诈

#### **4. F1 Score**

```
F1 = 2 · (Precision · Recall) / (Precision + Recall)

= Precision 和 Recall 的调和平均数

例：2 · (0.905 · 0.95) / (0.905 + 0.95) = 0.927
```

**特点**：平衡 Precision 和 Recall

#### **5. ROC 曲线和 AUC**

**ROC (Receiver Operating Characteristic) 曲线**：

```
TPR (True Positive Rate) = Recall = TP/(TP+FN)
FPR (False Positive Rate) = FP/(FP+TN)

横轴：FPR (假阳性率)
纵轴：TPR (真阳性率)
```

**AUC (Area Under Curve)**：ROC 曲线下面积
- AUC = 1: 完美分类器
- AUC = 0.5: 随机猜测
- AUC > 0.8: 通常认为不错

### 💻 计算评估指标

```python
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)
import matplotlib.pyplot as plt

# 预测
y_pred = model.predict(X)
y_proba = model.predict_proba(X)

# 1. 混淆矩阵
cm = confusion_matrix(y, y_pred)
print("混淆矩阵：")
print(cm)

# 2. 基本指标
accuracy = accuracy_score(y, y_pred)
precision = precision_score(y, y_pred)
recall = recall_score(y, y_pred)
f1 = f1_score(y, y_pred)

print(f"\nAccuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")

# 3. 完整报告
print("\n分类报告：")
print(classification_report(y, y_pred,
                           target_names=['类别0', '类别1']))

# 4. ROC 曲线
fpr, tpr, thresholds = roc_curve(y, y_proba)
auc = roc_auc_score(y, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC (AUC = {auc:.3f})')
plt.plot([0, 1], [0, 1], 'r--', label='随机猜测')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 5. Precision-Recall 曲线
from sklearn.metrics import precision_recall_curve

precision_vals, recall_vals, _ = precision_recall_curve(y, y_proba)

plt.figure(figsize=(8, 6))
plt.plot(recall_vals, precision_vals, 'b-', linewidth=2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.grid(True, alpha=0.3)
plt.show()
```

---

## 3.7 多元分类 (Multi-class Classification)

### 🎯 从二元到多元

**问题**：手写数字识别 (0-9)，共 10 个类别

### 🔹 方法 1：One-vs-Rest (OvR)

**策略**：训练 K 个二元分类器

```
分类器 1: 类别 0 vs 其他
分类器 2: 类别 1 vs 其他
...
分类器 K: 类别 K-1 vs 其他

预测时：选择输出概率最大的分类器
```

**例子**：识别数字 3

```
分类器 1 (0 vs 其他): P = 0.05
分类器 2 (1 vs 其他): P = 0.10
分类器 3 (2 vs 其他): P = 0.08
分类器 4 (3 vs 其他): P = 0.95  ✓ 最大
...
分类器 10 (9 vs 其他): P = 0.03

→ 预测为类别 3
```

### 🔹 方法 2：Softmax Regression

**核心思想**：扩展 Sigmoid 到多个类别

#### Softmax 函数

```
给定 K 个类别，计算 K 个分数：

z₁ = b₁ + w₁ᵀx
z₂ = b₂ + w₂ᵀx
...
zₖ = bₖ + wₖᵀx

Softmax:
P(y=i|x) = e^(zᵢ) / Σⱼ e^(zⱼ)

性质：
1. 所有概率和为 1: Σᵢ P(y=i|x) = 1
2. 每个概率都在 (0, 1)
3. 如果 K=2，退化为 Sigmoid
```

#### 可视化理解

```
例：3个类别

z₁ = 2.0    →  e^2.0 = 7.39
z₂ = 1.0    →  e^1.0 = 2.72
z₃ = 0.1    →  e^0.1 = 1.11
                ________
                Sum = 11.22

P(y=1) = 7.39/11.22 = 0.659  (65.9%)
P(y=2) = 2.72/11.22 = 0.242  (24.2%)
P(y=3) = 1.11/11.22 = 0.099  (9.9%)

预测：类别 1 (概率最高)
```

### 📐 Cross Entropy for Multi-class

```
对于单个样本：
L = -Σᵢ yᵢ·log(ŷᵢ)

其中：
  yᵢ: one-hot 编码的真实标签
  ŷᵢ: softmax 输出的预测概率

例：真实类别是 2 (共3个类别)
y = [0, 1, 0]       (one-hot)
ŷ = [0.1, 0.7, 0.2]  (预测)

L = -(0·log(0.1) + 1·log(0.7) + 0·log(0.2))
  = -log(0.7)
  = 0.357
```

### 💻 实现 Softmax Regression

```python
import numpy as np

class SoftmaxRegression:
    def __init__(self, learning_rate=0.01, epochs=1000):
        self.lr = learning_rate
        self.epochs = epochs
        self.W = None
        self.b = None

    def softmax(self, z):
        """Softmax 函数"""
        # 减去最大值防止数值溢出
        exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
        return exp_z / np.sum(exp_z, axis=1, keepdims=True)

    def fit(self, X, y):
        """训练模型"""
        n_samples, n_features = X.shape
        n_classes = len(np.unique(y))

        # One-hot 编码标签
        y_onehot = np.eye(n_classes)[y]

        # 初始化参数
        self.W = np.random.randn(n_features, n_classes) * 0.01
        self.b = np.zeros(n_classes)

        # 梯度下降
        for epoch in range(self.epochs):
            # 前向传播
            z = np.dot(X, self.W) + self.b
            y_pred = self.softmax(z)

            # 计算损失
            loss = -np.mean(np.sum(y_onehot * np.log(y_pred + 1e-9), axis=1))

            # 计算梯度
            dz = y_pred - y_onehot
            dW = np.dot(X.T, dz) / n_samples
            db = np.mean(dz, axis=0)

            # 更新参数
            self.W -= self.lr * dW
            self.b -= self.lr * db

            if epoch % 100 == 0:
                accuracy = np.mean(np.argmax(y_pred, axis=1) == y)
                print(f"Epoch {epoch}: Loss = {loss:.4f}, Acc = {accuracy:.4f}")

    def predict_proba(self, X):
        """预测概率"""
        z = np.dot(X, self.W) + self.b
        return self.softmax(z)

    def predict(self, X):
        """预测类别"""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

# 使用示例
if __name__ == "__main__":
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split

    # 加载鸢尾花数据集（3个类别）
    iris = load_iris()
    X, y = iris.data, iris.target

    # 分割数据
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 训练
    model = SoftmaxRegression(learning_rate=0.1, epochs=1000)
    model.fit(X_train, y_train)

    # 测试
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"\n测试集准确率: {accuracy:.4f}")
```

---

## 3.8 实战 1：信用卡欺诈检测

### 📋 问题描述

**数据集**：Kaggle Credit Card Fraud Detection
- 284,807 笔交易记录
- 492 笔欺诈（0.172%）← 极度不平衡！
- 30 个特征（PCA 处理过，已脱敏）

### ⚠️ 类别不平衡问题

```
正常交易: 284,315  (99.83%)
欺诈交易:     492  (0.17%)

如果全部预测为"正常"：
  Accuracy = 99.83%  (看起来很高！)
  但完全没有检测到欺诈！
```

### 💡 处理方法

#### **方法 1：重采样**

**欠采样 (Under-sampling)**：
```
减少多数类样本
正常: 284,315 → 492
欺诈:     492 → 492

优点：平衡数据集
缺点：丢失大量信息
```

**过采样 (Over-sampling)**：
```
增加少数类样本（复制或生成）
正常: 284,315 → 284,315
欺诈:     492 → 284,315

优点：保留所有信息
缺点：可能过拟合
```

**SMOTE (Synthetic Minority Over-sampling)**：
```
合成新的少数类样本
不是简单复制，而是在特征空间中插值生成
```

#### **方法 2：调整类别权重**

```python
from sklearn.linear_model import LogisticRegression

# 自动计算权重
model = LogisticRegression(class_weight='balanced')

# 手动设置
model = LogisticRegression(class_weight={0: 1, 1: 100})
```

#### **方法 3：调整决策阈值**

```python
# 默认阈值 0.5
y_pred = (y_proba >= 0.5).astype(int)

# 降低阈值，提高召回率
y_pred = (y_proba >= 0.3).astype(int)

# 提高阈值，提高精确率
y_pred = (y_proba >= 0.7).astype(int)
```

#### **方法 4：使用适当的评估指标**

```
不要用 Accuracy！
应该用：
  - F1 Score
  - Precision-Recall AUC
  - ROC AUC
```

### 💻 完整代码

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve
)
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 1. 加载数据
df = pd.read_csv('creditcard.csv')

print("数据形状:", df.shape)
print("\n类别分布:")
print(df['Class'].value_counts())
print(f"\n欺诈比例: {df['Class'].mean():.4%}")

# 2. 准备数据
X = df.drop('Class', axis=1)
y = df['Class']

# 3. 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 4. 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 5. 处理不平衡：SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(
    X_train_scaled, y_train
)

print(f"\nSMOTE 前: {len(y_train)} 样本")
print(f"SMOTE 后: {len(y_train_resampled)} 样本")
print(f"新的类别分布:\n{pd.Series(y_train_resampled).value_counts()}")

# 6. 训练模型
model = LogisticRegression(max_iter=1000, random_state=42)
model.fit(X_train_resampled, y_train_resampled)

# 7. 预测
y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

# 8. 评估
print("\n混淆矩阵:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

print("\n分类报告:")
print(classification_report(y_test, y_pred,
                           target_names=['正常', '欺诈']))

print(f"\nROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# 9. 可视化混淆矩阵
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()

# 10. Precision-Recall 曲线
precision, recall, thresholds = precision_recall_curve(y_test, y_proba)

plt.figure(figsize=(10, 6))
plt.plot(thresholds, precision[:-1], 'b-', label='Precision')
plt.plot(thresholds, recall[:-1], 'r-', label='Recall')
plt.xlabel('阈值')
plt.ylabel('分数')
plt.title('Precision vs Recall vs 阈值')
plt.legend()
plt.grid(True, alpha=0.3)
plt.show()

# 11. 找最优阈值
f1_scores = 2 * (precision * recall) / (precision + recall + 1e-9)
optimal_idx = np.argmax(f1_scores[:-1])
optimal_threshold = thresholds[optimal_idx]

print(f"\n最优阈值: {optimal_threshold:.4f}")
print(f"对应 F1 Score: {f1_scores[optimal_idx]:.4f}")

# 12. 使用最优阈值重新预测
y_pred_optimal = (y_proba >= optimal_threshold).astype(int)

print("\n使用最优阈值的结果:")
print(classification_report(y_test, y_pred_optimal,
                           target_names=['正常', '欺诈']))
```

---

## 3.9 实战 2：手写数字识别 (MNIST)

### 📋 MNIST 数据集

```
70,000 张手写数字图片
  - 60,000 训练集
  - 10,000 测试集

每张图片：
  - 28×28 像素
  - 灰度值 0-255
  - 标签：0-9
```

### 💻 完整代码

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# 1. 加载数据
print("加载 MNIST 数据...")
mnist = fetch_openml('mnist_784', version=1, parser='auto')
X, y = mnist['data'], mnist['target'].astype(int)

print(f"数据形状: {X.shape}")
print(f"标签形状: {y.shape}")

# 2. 可视化一些样本
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for i, ax in enumerate(axes.flat):
    ax.imshow(X.iloc[i].values.reshape(28, 28), cmap='gray')
    ax.set_title(f'标签: {y.iloc[i]}')
    ax.axis('off')
plt.tight_layout()
plt.show()

# 3. 数据预处理
# 归一化到 [0, 1]
X = X / 255.0

# 使用部分数据（加快训练）
X_subset = X[:10000]
y_subset = y[:10000]

# 分割数据
X_train, X_test, y_train, y_test = train_test_split(
    X_subset, y_subset, test_size=0.2, random_state=42
)

# 4. 训练模型
print("\n训练逻辑回归模型...")
model = LogisticRegression(
    max_iter=100,
    multi_class='multinomial',  # Softmax
    solver='lbfgs',             # 优化算法
    random_state=42
)
model.fit(X_train, y_train)

# 5. 预测
y_pred = model.predict(X_test)

# 6. 评估
accuracy = accuracy_score(y_test, y_pred)
print(f"\n准确率: {accuracy:.4f}")

print("\n分类报告:")
print(classification_report(y_test, y_pred))

# 7. 混淆矩阵
cm = confusion_matrix(y_test, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel('预测标签')
plt.ylabel('真实标签')
plt.title('混淆矩阵')
plt.show()

# 8. 可视化一些预测结果
fig, axes = plt.subplots(3, 5, figsize=(12, 7))
indices = np.random.choice(len(X_test), 15, replace=False)

for i, (ax, idx) in enumerate(zip(axes.flat, indices)):
    image = X_test.iloc[idx].values.reshape(28, 28)
    true_label = y_test.iloc[idx]
    pred_label = y_pred[idx]

    ax.imshow(image, cmap='gray')
    color = 'green' if true_label == pred_label else 'red'
    ax.set_title(f'真实: {true_label}, 预测: {pred_label}', color=color)
    ax.axis('off')

plt.tight_layout()
plt.show()

# 9. 查看模型学到的权重
# 每个数字的权重可以看作一个"模板"
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for digit, ax in enumerate(axes.flat):
    weight = model.coef_[digit].reshape(28, 28)
    ax.imshow(weight, cmap='RdBu', vmin=-weight.max(), vmax=weight.max())
    ax.set_title(f'数字 {digit}的权重')
    ax.axis('off')
plt.tight_layout()
plt.show()
```

---

## 3.10 逻辑回归的优缺点

### ✅ 优点

1. **简单高效**
   - 易于实现和理解
   - 训练速度快
   - 预测速度快

2. **可解释性强**
   - 可以看权重了解特征重要性
   - 输出概率，方便决策

3. **不需要太多数据**
   - 相比深度学习，数据需求少

4. **不容易过拟合**
   - 模型简单，泛化能力好
   - 可以用正则化进一步控制

### ❌ 缺点

1. **线性模型**
   - 只能学习线性决策边界
   - 无法处理复杂的非线性关系

2. **特征工程依赖**
   - 需要手动设计好的特征
   - 特征质量决定模型上限

3. **多重共线性敏感**
   - 特征高度相关时，模型不稳定

4. **类别不平衡问题**
   - 需要特殊处理

---

## 📝 本章作业

### 作业 1：概念题

1. **为什么不能用线性回归做分类？**
   - 举例说明问题
   - 画图解释

2. **Sigmoid vs Softmax**
   - 什么时候用 Sigmoid？
   - 什么时候用 Softmax？
   - 它们的关系是什么？

3. **评估指标选择**
   - 癌症检测应该关注 Precision 还是 Recall？为什么？
   - 垃圾邮件过滤应该关注哪个？
   - 给出 3 个场景和对应的最重要指标

### 作业 2：编程实践

#### 任务 1：乳腺癌检测

```python
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target

# TODO:
# 1. 分割数据（80/20）
# 2. 特征缩放
# 3. 训练逻辑回归模型
# 4. 评估模型（混淆矩阵、ROC曲线）
# 5. 调整决策阈值，优化 Recall
# 6. 比较不同正则化强度（C参数）的效果
```

#### 任务 2：多分类实战

使用 Scikit-learn 的 `load_digits` 数据集（8×8 手写数字）

```python
from sklearn.datasets import load_digits

# TODO:
# 1. 加载数据并可视化
# 2. 训练 Softmax Regression
# 3. 分析哪些数字容易混淆（看混淆矩阵）
# 4. 可视化模型学到的权重
# 5. 实现 One-vs-Rest 方法并对比性能
```

### 作业 3：Kaggle 竞赛

参加 "Titanic - Machine Learning from Disaster"

**要求**：
1. 数据探索和可视化
2. 特征工程（处理缺失值、创建新特征）
3. 训练逻辑回归模型
4. 调整超参数
5. 提交预测结果
6. 写一份完整报告

---

## 🔑 本章关键概念总结

| 概念 | 说明 |
|------|------|
| 分类 | 预测离散类别 |
| Sigmoid | 将实数映射到 (0,1) |
| 逻辑回归 | 用于二元分类 |
| Cross Entropy | 分类问题的损失函数 |
| Softmax | 多元分类的激活函数 |
| Confusion Matrix | 评估分类性能 |
| Precision | 预测为正的准确率 |
| Recall | 找出所有正例的能力 |
| F1 Score | Precision 和 Recall 的调和平均 |
| ROC/AUC | 评估分类器性能 |
| 类别不平衡 | 类别样本数差异大 |
| SMOTE | 合成少数类样本 |

---

## 🎯 下一章预告

**第四章：深度神经网络基础 (Deep Neural Networks)**
- 从逻辑回归到神经网络
- 激活函数 (ReLU, Tanh, etc.)
- 反向传播算法
- 深度网络的训练技巧
- 实战：用神经网络改进 MNIST

---

-----
