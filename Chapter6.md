# 第六章：卷积神经网络 (Convolutional Neural Networks)

## 📌 章节目标
- 理解卷积操作的原理
- 掌握 CNN 的关键概念（卷积核、池化、感受野）
- 学习经典 CNN 架构
- 了解 CNN 的特性和优势
- 实战：图像分类和物体检测

---

## 6.1 为什么需要 CNN？

### 🎯 全连接网络的问题

**图像分类任务**：28×28 像素的手写数字

```
输入层：28×28 = 784 个神经元
隐藏层：512 个神经元

权重数量：784 × 512 = 401,408 个！
```

**问题**：
1. **参数太多**：容易过拟合
2. **计算量大**：训练慢
3. **空间结构丢失**：相邻像素的关系被忽视
4. **不稳定**：平移图像会得到不同结果

### 💡 CNN 的核心思想

**观察**：图像具有局部结构特性

```
一个小的卷积核可以检测：
  - 边缘
  - 角
  - 纹理
  - ...
```

**策略**：
1. 用小卷积核扫描整个图像（参数共享）
2. 提取局部特征
3. 逐层抽象（从低级特征到高级语义）

---

## 6.2 卷积操作 (Convolution)

### 📐 单通道卷积

**输入**：5×5 的图像
```
1 0 1 0 1
0 1 0 1 0
1 0 1 0 1
0 1 0 1 0
1 0 1 0 1
```

**卷积核**：3×3
```
1 0 -1
1 0 -1
1 0 -1
```

**卷积过程**：

```
第1个位置：
  1 0 1       1 0 -1
  0 1 0   ⊗   1 0 -1   = 1×1 + 0×0 + 1×(-1) + 0×1 + 1×0 + 0×(-1)
  1 0 1       1 0 -1     + 1×1 + 0×0 + 1×(-1)
                       = 1 + 0 - 1 + 0 + 0 + 0 + 1 + 0 - 1 = 0
```

**输出**：特征图（feature map）

```
0  2  0
2  0  2
0  2  0  (3×3 的输出)
```

### 📐 多通道卷积 (彩色图像)

**输入**：5×5×3 (RGB 图像)
```
R通道、G通道、B通道
```

**卷积核**：3×3×3 (针对每个通道各有一个核)

```
卷积过程：
  对每个通道分别做卷积
  然后求和
```

### 📐 数学表示

```
y[i,j] = Σ_m Σ_n x[i+m, j+n] · w[m,n] + b

其中：
  x: 输入
  w: 卷积核权重
  b: 偏置
```

**效率高的原因**：
- 参数共享：同一个卷积核用于整个图像
- 相比全连接：参数量大幅减少

---

### 💻 Python 实现

```python
import numpy as np

def convolve2d(image, kernel, padding=0, stride=1):
    """
    2D 卷积

    参数：
        image: 输入 (H, W)
        kernel: 卷积核 (K, K)
        padding: 填充
        stride: 步长
    """
    H, W = image.shape
    K, _ = kernel.shape

    # 加 padding
    if padding > 0:
        image = np.pad(image, padding, mode='constant')

    # 输出大小
    H_out = (H + 2*padding - K) // stride + 1
    W_out = (W + 2*padding - K) // stride + 1

    # 输出特征图
    output = np.zeros((H_out, W_out))

    # 卷积操作
    for i in range(H_out):
        for j in range(W_out):
            # 提取区域
            region = image[i*stride:i*stride+K, j*stride:j*stride+K]
            # 逐元素相乘后求和
            output[i, j] = np.sum(region * kernel)

    return output

# 示例
image = np.array([
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1],
    [0, 1, 0, 1, 0],
    [1, 0, 1, 0, 1]
], dtype=float)

kernel = np.array([
    [1, 0, -1],
    [1, 0, -1],
    [1, 0, -1]
], dtype=float)

output = convolve2d(image, kernel, padding=0, stride=1)
print("卷积输出：")
print(output)
```

---

## 6.3 关键概念

### 🔹 Padding (填充)

**问题**：卷积会减小图像尺寸

```
输入：5×5
卷积核：3×3，stride=1
输出：3×3 (缩小了)
```

**解决**：在边界添加零

```
不 padding:           使用 padding=1:
1 0 1 0 1           0 0 0 0 0 0
0 1 0 1 0           0 1 0 1 0 1
1 0 1 0 1    →      0 0 1 0 1 0
0 1 0 1 0           0 1 0 1 0 1
1 0 1 0 1           0 0 1 0 1 0
                    0 0 0 0 0 0

输出：3×3             输出：5×5
```

**'Same' padding**：`padding = (kernel_size - 1) / 2`
- 保持输入输出尺寸相同

**'Valid' padding**：无 padding
- 输出尺寸 = (input_size - kernel_size) / stride + 1

---

### 🔹 Stride (步长)

**一次卷积核移动的距离**

```
stride=1：
[●]⚬⚬⚬⚬
⚬●⚬⚬
⚬⚬●⚬⚬
⚬⚬⚬●⚬
⚬⚬⚬⚬●

stride=2：
[●]⚬[●]⚬[●]
⚬⚬⚬⚬⚬
[●]⚬[●]⚬[●]
```

**输出尺寸计算**：

```
H_out = floor((H_in + 2×padding - kernel_size) / stride) + 1
W_out = floor((W_in + 2×padding - kernel_size) / stride) + 1
```

---

### 🔹 感受野 (Receptive Field)

**定义**：输出特征图的一个像素能"看到"的输入区域大小

```
单层 3×3 卷积：
  感受野 = 3×3

两层 3×3 卷积：
  感受野 = 5×5

三层 3×3 卷积：
  感受野 = 7×7
```

**计算公式**：

```
RF_l = RF_{l-1} + (kernel_size - 1) × Π(stride_i)
```

**意义**：
- 深层神经元能看到更大范围
- 可以捕获更高级的特征

---

### 🔹 池化 (Pooling)

**目的**：降低特征图尺寸，减少计算量

#### **Max Pooling（最大池化）**

```
输入 4×4:
1  3  2  4
5  6  7  8
9  10 11 12
13 14 15 16

Max Pooling 2×2, stride=2:
[1  3]  [2  4]      6   8
[5  6]  [7  8]  →
                    14  16
[9  10] [11 12]
[13 14] [15 16]

输出 2×2:
6  8
14 16
```

#### **Average Pooling（平均池化）**

```
对每个区域取平均值
```

**代码**：

```python
def max_pool2d(input, pool_size=2, stride=2):
    """Max Pooling"""
    H, W = input.shape
    H_out = (H - pool_size) // stride + 1
    W_out = (W - pool_size) // stride + 1

    output = np.zeros((H_out, W_out))

    for i in range(H_out):
        for j in range(W_out):
            region = input[i*stride:i*stride+pool_size,
                          j*stride:j*stride+pool_size]
            output[i, j] = np.max(region)

    return output

# PyTorch
import torch.nn as nn
max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
```

**特点**：
- ✅ 减少参数和计算量
- ✅ 增加平移不变性
- ✅ 扩大感受野
- ❌ 丢失位置信息

---

## 6.4 典型 CNN 架构

### 🏗️ 基本结构

```
输入图像
    ↓
[卷积 → ReLU → 池化] × N 层
    ↓
展平
    ↓
全连接层 × M 层
    ↓
Softmax 输出
```

---

### 🔹 LeNet-5 (1998) - CNN 的开山之作

**架构**：

```
输入: 32×32×1 (灰度图)
    ↓
Conv1: 6个 5×5 卷积核 → 28×28×6
    ↓
AvgPool1: 2×2 → 14×14×6
    ↓
Conv2: 16个 5×5 卷积核 → 10×10×16
    ↓
AvgPool2: 2×2 → 5×5×16
    ↓
Flatten → 400
    ↓
FC1: 120
    ↓
FC2: 84
    ↓
FC3: 10 (输出)
```

**代码实现**：

```python
import torch
import torch.nn as nn

class LeNet5(nn.Module):
    def __init__(self):
        super(LeNet5, self).__init__()

        # 特征提取部分
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)  # 32→32
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)       # 32→16

        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)             # 16→12
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)       # 12→6

        # 分类部分
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # 卷积层
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)

        x = torch.relu(self.conv2(x))
        x = self.pool2(x)

        # 展平
        x = x.view(-1, 16 * 6 * 6)

        # 全连接层
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)

        return x

model = LeNet5()
print(model)
```

---

### 🔹 AlexNet (2012) - 深度学习复兴

**创新点**：
- 使用 ReLU 替代 Sigmoid/Tanh
- 使用 Dropout 防止过拟合
- 数据增强
- GPU 加速训练
- 更深的网络（8层）

**架构**：

```
输入: 224×224×3
    ↓
Conv1: 96个 11×11, stride=4 → 55×55×96
MaxPool1: 3×3, stride=2 → 27×27×96
    ↓
Conv2: 256个 5×5 → 27×27×256
MaxPool2: 3×3, stride=2 → 13×13×256
    ↓
Conv3: 384个 3×3 → 13×13×384
Conv4: 384个 3×3 → 13×13×384
Conv5: 256个 3×3 → 13×13×256
MaxPool3: 3×3, stride=2 → 6×6×256
    ↓
FC1: 4096 + Dropout
FC2: 4096 + Dropout
FC3: 1000 (ImageNet)
```

**代码**：

```python
class AlexNet(nn.Module):
    def __init__(self, num_classes=1000):
        super(AlexNet, self).__init__()

        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 96, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv2
            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            # Conv3
            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv4
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),

            # Conv5
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),

            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

---

### 🔹 VGGNet (2014) - 更深更规整

**核心思想**：
- 只使用 3×3 小卷积核
- 网络更深（16-19层）
- 结构规整，易于理解

**为什么 3×3？**

```
两个 3×3 卷积 = 一个 5×5 感受野
三个 3×3 卷积 = 一个 7×7 感受野

但参数更少：
  7×7: 49 个参数
  3个3×3: 27 个参数
```

**VGG-16 架构**：

```
输入: 224×224×3

Block 1:
  Conv3-64 × 2
  MaxPool

Block 2:
  Conv3-128 × 2
  MaxPool

Block 3:
  Conv3-256 × 3
  MaxPool

Block 4:
  Conv3-512 × 3
  MaxPool

Block 5:
  Conv3-512 × 3
  MaxPool

FC: 4096 → 4096 → 1000
```

**代码**：

```python
class VGG16(nn.Module):
    def __init__(self, num_classes=1000):
        super(VGG16, self).__init__()

        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))

        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),

            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
```

---

### 🔹 ResNet (2015) - 残差网络 ⭐

**问题**：网络越深，性能越差？

```
56层网络的训练误差 > 20层网络？
这不是过拟合，而是优化问题！
```

**解决**：残差连接（Skip Connection）

```
传统：
  x → Conv → ReLU → Conv → ReLU → output

ResNet：
  x → Conv → ReLU → Conv ─┬→ ReLU → output
  └──────────────────────┘
       (直接连接)

输出 = F(x) + x
```

**优势**：
- 解决梯度消失/爆炸
- 允许训练超深网络（152层，甚至1000层）
- 性能显著提升

**Residual Block**：

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # 主路径
        self.conv1 = nn.Conv2d(in_channels, out_channels,
                               kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels,
                               kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 捷径连接
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                         kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        # 主路径
        out = self.conv1(x)
        out = self.bn1(out)
        out = torch.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        # 残差连接
        out += self.shortcut(x)
        out = torch.relu(out)

        return out
```

**ResNet-34 架构**：

```python
class ResNet34(nn.Module):
    def __init__(self, num_classes=1000):
        super(ResNet34, self).__init__()

        # 初始层
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Residual blocks
        self.layer1 = self._make_layer(64, 64, 3, stride=1)
        self.layer2 = self._make_layer(64, 128, 4, stride=2)
        self.layer3 = self._make_layer(128, 256, 6, stride=2)
        self.layer4 = self._make_layer(256, 512, 3, stride=2)

        # 分类层
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, num_blocks, stride):
        layers = []

        # 第一个 block 可能改变尺寸
        layers.append(ResidualBlock(in_channels, out_channels, stride))

        # 剩余的 blocks
        for _ in range(1, num_blocks):
            layers.append(ResidualBlock(out_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        # 初始卷积
        x = self.conv1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.maxpool(x)

        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # 分类
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
```

**ResNet 变种**：

| 模型 | 层数 | Top-5 错误率 (ImageNet) |
|------|------|----------------------|
| ResNet-18 | 18 | ~10% |
| ResNet-34 | 34 | ~8% |
| ResNet-50 | 50 | ~6.7% |
| ResNet-101 | 101 | ~6.4% |
| ResNet-152 | 152 | ~6.2% |

---

## 6.5 现代 CNN 技巧

### 🔹 1×1 卷积

**作用**：
- 改变通道数（降维/升维）
- 增加非线性
- 参数少

```
输入: 56×56×192
1×1 卷积，64个核
输出: 56×56×64

作用类似全连接，但保持空间结构
```

```python
# 降维示例
nn.Conv2d(192, 64, kernel_size=1)  # 192通道 → 64通道
```

---

### 🔹 全局平均池化 (Global Average Pooling)

**替代全连接层**

```
传统：
  7×7×512 → Flatten → FC(4096) → FC(1000)
  参数量：7×7×512×4096 ≈ 102M

GAP：
  7×7×512 → GAP → 512 → FC(1000)
  参数量：512×1000 ≈ 512K
```

```python
# PyTorch
self.gap = nn.AdaptiveAvgPool2d((1, 1))  # 输出 1×1×C

# 使用
x = self.gap(x)  # (B, C, H, W) → (B, C, 1, 1)
x = x.view(x.size(0), -1)  # (B, C)
```

**优势**：
- 参数大幅减少
- 更强的空间不变性
- 减少过拟合

---

### 🔹 深度可分离卷积 (Depthwise Separable Convolution)

**MobileNet 的核心**

**标准卷积**：
```
输入: H×W×C_in
卷积核: K×K×C_in×C_out
参数量: K×K×C_in×C_out
```

**深度可分离卷积**：分两步

```
1. Depthwise 卷积：
   每个输入通道单独卷积
   参数: K×K×C_in

2. Pointwise 卷积：
   1×1 卷积混合通道
   参数: 1×1×C_in×C_out

总参数: K×K×C_in + C_in×C_out
```

**参数减少比例**：
```
(K²×C_in + C_in×C_out) / (K²×C_in×C_out)
= 1/C_out + 1/K²
```

**代码**：

```python
class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        # Depthwise
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=kernel_size//2,
            groups=in_channels  # 关键！每组一个通道
        )

        # Pointwise
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1
        )

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
```

---

## 6.6 实战：CIFAR-10 图像分类

### 📋 数据集介绍

```
CIFAR-10:
  - 60,000 张 32×32 彩色图像
  - 10 个类别（飞机、汽车、鸟...）
  - 50,000 训练 + 10,000 测试
```

### 💻 完整实现

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from tqdm import tqdm

# ==================== 超参数 ====================
BATCH_SIZE = 128
LEARNING_RATE = 0.001
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 数据增强 ====================
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                        (0.2023, 0.1994, 0.2010))
])

# ==================== 加载数据 ====================
train_dataset = datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform_train
)
test_dataset = datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform_test
)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=2)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=2)

# ==================== 定义模型 ====================
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()

        # Block 1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.2)

        # Block 2
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout(0.3)

        # Block 3
        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout(0.4)

        # 全连接层
        self.fc1 = nn.Linear(256 * 4 * 4, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        # Block 1
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2
        x = torch.relu(self.bn3(self.conv3(x)))
        x = torch.relu(self.bn4(self.conv4(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3
        x = torch.relu(self.bn5(self.conv5(x)))
        x = torch.relu(self.bn6(self.conv6(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # 展平和全连接
        x = x.view(-1, 256 * 4 * 4)
        x = torch.relu(self.fc1(x))
        x = self.dropout4(x)
        x = self.fc2(x)

        return x

model = SimpleCNN().to(DEVICE)

# 查看模型结构
from torchsummary import summary
summary(model, (3, 32, 32))

# ==================== 损失和优化器 ====================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

# ==================== 训练函数 ====================
def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    progress_bar = tqdm(train_loader, desc='Training')

    for images, labels in progress_bar:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        progress_bar.set_postfix({
            'loss': f'{running_loss/len(train_loader):.3f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return running_loss / len(train_loader), 100. * correct / total

# ==================== 测试函数 ====================
def test(model, test_loader, criterion, device):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Testing'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return test_loss / len(test_loader), 100. * correct / total

# ==================== 训练循环 ====================
train_losses, train_accs = [], []
test_losses, test_accs = [], []
best_acc = 0

for epoch in range(EPOCHS):
    print(f'\n=== Epoch {epoch+1}/{EPOCHS} ===')

    train_loss, train_acc = train_epoch(model, train_loader, criterion,
                                        optimizer, DEVICE)
    test_loss, test_acc = test(model, test_loader, criterion, DEVICE)

    train_losses.append(train_loss)
    train_accs.append(train_acc)
    test_losses.append(test_loss)
    test_accs.append(test_acc)

    print(f'Train Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
    print(f'Test  Loss: {test_loss:.4f}, Acc: {test_acc:.2f}%')

    # 保存最佳模型
    if test_acc > best_acc:
        best_acc = test_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f'✓ 保存最佳模型 (Acc: {best_acc:.2f}%)')

    scheduler.step()

# ==================== 可视化 ====================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

epochs_range = range(1, len(train_losses) + 1)

ax1.plot(epochs_range, train_losses, 'b-', label='Train Loss')
ax1.plot(epochs_range, test_losses, 'r-', label='Test Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Test Loss')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs_range, train_accs, 'b-', label='Train Acc')
ax2.plot(epochs_range, test_accs, 'r-', label='Test Acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy (%)')
ax2.set_title('Training and Test Accuracy')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ==================== 最终评估 ====================
model.load_state_dict(torch.load('best_model.pth'))
_, final_acc = test(model, test_loader, criterion, DEVICE)
print(f'\n最终测试准确率: {final_acc:.2f}%')
```

---

## 6.7 卷积神经网络的可视化

### 🎨 特征图可视化

```python
def visualize_feature_maps(model, image, device):
    """可视化中间层的特征图"""
    model.eval()

    # 提取特征图的钩子
    activations = {}

    def get_activation(name):
        def hook(model, input, output):
            activations[name] = output.detach()
        return hook

    # 注册钩子
    model.conv1.register_forward_hook(get_activation('conv1'))
    model.conv3.register_forward_hook(get_activation('conv3'))
    model.conv5.register_forward_hook(get_activation('conv5'))

    # 前向传播
    with torch.no_grad():
        output = model(image.to(device))

    # 可视化
    fig, axes = plt.subplots(3, 8, figsize=(16, 6))

    # Conv1 特征图
    feat1 = activations['conv1'][0]
    for i in range(8):
        ax = axes[0, i]
        ax.imshow(feat1[i].cpu().numpy(), cmap='gray')
        ax.set_title(f'Conv1-{i}')
        ax.axis('off')

    # Conv3 特征图
    feat3 = activations['conv3'][0]
    for i in range(8):
        ax = axes[1, i]
        ax.imshow(feat3[i].cpu().numpy(), cmap='gray')
        ax.set_title(f'Conv3-{i}')
        ax.axis('off')

    # Conv5 特征图
    feat5 = activations['conv5'][0]
    for i in range(8):
        ax = axes[2, i]
        ax.imshow(feat5[i].cpu().numpy(), cmap='gray')
        ax.set_title(f'Conv5-{i}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# 使用
test_image, _ = test_dataset[0]
test_image = test_image.unsqueeze(0)
visualize_feature_maps(model, test_image, DEVICE)
```

---

### 🎯 卷积核可视化

```python
def visualize_kernels(model):
    """可视化第一层卷积核"""
    conv1_weight = model.conv1.weight.data

    fig, axes = plt.subplots(8, 8, figsize=(12, 12))

    for i in range(64):
        ax = axes[i // 8, i % 8]

        # 平均RGB三个通道
        kernel = conv1_weight[i].mean(dim=0)

        # 标准化到 [0, 1]
        kernel = (kernel - kernel.min()) / (kernel.max() - kernel.min())

        ax.imshow(kernel.cpu().numpy(), cmap='gray')
        ax.set_title(f'Filter {i}')
        ax.axis('off')

    plt.tight_layout()
    plt.show()

# 使用
visualize_kernels(model)
```

---

## 6.8 CNN vs ViT（展望）

### 🔄 从 CNN 到 Vision Transformer

**CNN 的局限**：
- 感受野局限（逐层扩大）
- 空间归纳偏置强（可能限制性能）
- 需要更多数据

**Vision Transformer 的优势**：
- 全局感受野（从第一层）
- 更灵活的特征提取
- 扩展性好

```
CNN 架构：
  卷积 → 卷积 → 卷积 → 特征
  (逐层聚合局部信息)

ViT 架构：
  Patch Embedding → Transformer → 特征
  (直接捕获全局关系)
```

---

## 📝 本章作业

### 作业 1：从零实现卷积

```python
# TODO:
# 1. 实现前向卷积（已提供）
# 2. 实现反向传播：梯度w.r.t 输入、权重、偏置
# 3. 实现池化前向和反向
# 4. 组合成完整 CNN 层
# 5. 在小数据集上测试
```

### 作业 2：CNN 架构对比

在 CIFAR-10 上实现并对比：

```python
# 1. LeNet-5
# 2. 简单 CNN (3层卷积)
# 3. VGG-16 (使用预训练模型)
# 4. ResNet-18 (使用预训练模型)

# 记录：
#   - 参数数量
#   - 训练时间
#   - 最终准确率
#   - 模型大小

# 分析优缺点
```

### 作业 3：特征可视化

```python
# 对训练好的 CNN 模型：

# 1. 可视化不同层的特征图
# 2. 绘制卷积核
# 3. 尝试 DeconvNet 或 Grad-CAM 进行可视化
# 4. 分析不同层学到了什么

# 写一份分析报告
```

### 作业 4：数据增强实验

```python
# 在 CIFAR-10 上测试不同的数据增强方法：

# 1. 无增强
# 2. 随机裁剪 + 翻转
# 3. + 颜色抖动
# 4. + Cutout
# 5. + MixUp / CutMix

# 记录性能差异，分析每种增强的作用
```

---

## 🔑 本章关键概念

| 概念 | 说明 |
|------|------|
| 卷积 | 特征提取的核心操作 |
| 感受野 | 输出能看到的输入区域 |
| 池化 | 降维和特征聚合 |
| 填充 | 控制输出尺寸 |
| 步长 | 卷积核移动距离 |
| 参数共享 | CNN 的核心优势 |
| LeNet | CNN 的开创者 |
| AlexNet | 深度学习复兴 |
| VGG | 规整深层设计 |
| ResNet | 残差连接解决深度问题 |
| 深度可分离卷积 | 参数高效设计 |

---

## 🎯 后续章节预告

**第七章：循环神经网络（RNN & LSTM）**
- 序列数据处理
- RNN 的梯度问题
- LSTM 和 GRU
- 双向 RNN

**第八章：Attention 与 Transformer**
- Self-Attention 机制
- Transformer 架构
- BERT 和 GPT

**第九章：迁移学习与微调**
- 预训练模型
- 特征提取
- Fine-tuning 策略

---

这是一部系统、详细且易懂的深度学习教程，涵盖了从基础到进阶的完整内容。每章都包含：

✅ **理论讲解** - 直观易懂，配图说明
✅ **数学推导** - 关键公式详细推导
✅ **代码实现** - 完整可运行的示例
✅ **实战项目** - 真实数据集的端到端流程
✅ **作业练习** - 帮助巩固知识
✅ **可视化** - 帮助理解复杂概念


-----

