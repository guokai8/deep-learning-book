# 第十二章：可解释性与对抗攻击

## 📌 章节目标
- 理解深度学习的黑盒问题
- 掌握模型可解释性技术
- 学习对抗攻击和防御方法
- 了解鲁棒性和安全性
- 实战：可视化模型决策、生成对抗样本

---

## 12.1 为什么需要可解释性？

### 🎯 黑盒问题

```
深度学习模型 = 黑盒？

输入 → [神经网络] → 输出
        ？？？

问题：
  - 为什么做出这个预测？
  - 模型学到了什么特征？
  - 如何调试错误？
  - 如何建立信任？
```

### 📊 应用场景需求

```
医疗诊断：
  "为什么诊断为癌症？"
  → 需要指出关键区域

金融风控：
  "为什么拒绝贷款？"
  → 法律要求可解释

自动驾驶：
  "为什么做出这个决策？"
  → 安全性要求
```

---

## 12.2 可解释性方法分类

### 📐 分类维度

#### **1. 全局 vs 局部**

```
全局解释 (Global Interpretation):
  - 模型整体如何工作
  - 哪些特征最重要
  - 例：特征重要性

局部解释 (Local Interpretation):
  - 单个预测如何产生
  - 为什么这个样本被分类为X
  - 例：LIME, SHAP
```

#### **2. 模型特定 vs 模型无关**

```
模型特定 (Model-Specific):
  - 针对特定模型架构
  - 例：神经网络的梯度可视化

模型无关 (Model-Agnostic):
  - 适用于任何模型
  - 例：LIME, SHAP
```

---

## 12.3 特征重要性

### 🔹 排列重要性 (Permutation Importance)

**原理**：打乱某个特征，看性能下降多少

```python
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

def permutation_importance(model, X, y, metric=accuracy_score, n_repeats=10):
    """
    计算排列重要性

    参数:
        model: 训练好的模型
        X: 特征矩阵
        y: 标签
        metric: 评估指标
        n_repeats: 重复次数
    """
    # 基线性能
    baseline_score = metric(y, model.predict(X))

    importances = []
    n_features = X.shape[1]

    for feature_idx in range(n_features):
        scores = []

        for _ in range(n_repeats):
            X_permuted = X.copy()

            # 打乱该特征
            np.random.shuffle(X_permuted[:, feature_idx])

            # 计算性能下降
            permuted_score = metric(y, model.predict(X_permuted))
            score_decrease = baseline_score - permuted_score
            scores.append(score_decrease)

        # 平均重要性
        importances.append({
            'feature': feature_idx,
            'importance': np.mean(scores),
            'std': np.std(scores)
        })

    return sorted(importances, key=lambda x: x['importance'], reverse=True)

# ==================== 可视化 ====================

def plot_feature_importance(importances, feature_names=None):
    """绘制特征重要性"""

    indices = [imp['feature'] for imp in importances]
    values = [imp['importance'] for imp in importances]
    stds = [imp['std'] for imp in importances]

    if feature_names is None:
        feature_names = [f'Feature {i}' for i in indices]
    else:
        feature_names = [feature_names[i] for i in indices]

    plt.figure(figsize=(10, 6))
    plt.barh(range(len(values)), values, xerr=stds,
            color='steelblue', alpha=0.7)
    plt.yticks(range(len(values)), feature_names)
    plt.xlabel('Importance')
    plt.title('Feature Importance (Permutation)')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.show()

# ==================== 示例 ====================

if __name__ == '__main__':
    from sklearn.datasets import load_iris
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    # 加载数据
    iris = load_iris()
    X, y = iris.data, iris.target
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # 训练模型
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # 计算重要性
    importances = permutation_importance(model, X_test, y_test)

    # 可视化
    plot_feature_importance(importances, iris.feature_names)
```

---

## 12.4 梯度可视化技术

### 🔹 Saliency Maps（显著图）

**原理**：计算输出对输入的梯度

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class SaliencyMap:
    """显著图生成器"""

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def generate(self, image, target_class=None):
        """
        生成显著图

        参数:
            image: 输入图像 (C, H, W)
            target_class: 目标类别（None 则使用预测类别）
        """
        # 确保需要梯度
        image = image.unsqueeze(0).requires_grad_(True)

        # 前向传播
        output = self.model(image)

        # 选择目标类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 反向传播
        self.model.zero_grad()
        output[0, target_class].backward()

        # 获取梯度
        saliency = image.grad.data.abs()

        # 取最大值作为显著性
        saliency = saliency.max(dim=1)[0]  # (1, H, W)

        return saliency.squeeze().cpu().numpy()

# ==================== 可视化 ====================

def visualize_saliency(image, saliency, title='Saliency Map'):
    """可视化显著图"""

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原图
    if image.shape[0] == 3:  # RGB
        img_display = image.permute(1, 2, 0).cpu().numpy()
    else:  # 灰度
        img_display = image.squeeze().cpu().numpy()

    axes[0].imshow(img_display, cmap='gray' if len(img_display.shape)==2 else None)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # 显著图
    axes[1].imshow(saliency, cmap='hot')
    axes[1].set_title('Saliency Map')
    axes[1].axis('off')

    # 叠加
    axes[2].imshow(img_display, cmap='gray' if len(img_display.shape)==2 else None)
    axes[2].imshow(saliency, cmap='hot', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()
```

---

### 🔹 Grad-CAM (Gradient-weighted Class Activation Mapping)

**原理**：结合梯度和特征图

```python
class GradCAM:
    """Grad-CAM 类激活映射"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer

        self.gradients = None
        self.activations = None

        # 注册钩子
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        """保存激活"""
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        """保存梯度"""
        self.gradients = grad_output[0].detach()

    def generate(self, image, target_class=None):
        """
        生成 Grad-CAM

        返回: CAM 热力图
        """
        # 前向传播
        output = self.model(image)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 反向传播
        self.model.zero_grad()
        output[0, target_class].backward()

        # 计算权重（全局平均池化梯度）
        weights = self.gradients.mean(dim=(2, 3), keepdim=True)

        # 加权求和
        cam = (weights * self.activations).sum(dim=1, keepdim=True)

        # ReLU
        cam = F.relu(cam)

        # 归一化
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.squeeze().cpu().numpy()

# ==================== 可视化 Grad-CAM ====================

def visualize_gradcam(image, cam, title='Grad-CAM'):
    """可视化 Grad-CAM"""

    import cv2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # 原图
    if image.shape[0] == 3:
        img_display = image.permute(1, 2, 0).cpu().numpy()
    else:
        img_display = image.squeeze().cpu().numpy()

    # 归一化到 [0, 1]
    img_display = (img_display - img_display.min()) / (img_display.max() - img_display.min())

    axes[0].imshow(img_display)
    axes[0].set_title('Original Image')
    axes[0].axis('off')

    # CAM 热力图
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('Grad-CAM')
    axes[1].axis('off')

    # 叠加
    # 调整 CAM 大小到图像大小
    cam_resized = cv2.resize(cam, (img_display.shape[1], img_display.shape[0]))

    axes[2].imshow(img_display)
    axes[2].imshow(cam_resized, cmap='jet', alpha=0.5)
    axes[2].set_title('Overlay')
    axes[2].axis('off')

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()

# ==================== 使用示例 ====================

if __name__ == '__main__':
    from torchvision import models, transforms
    from PIL import Image

    # 加载预训练模型
    model = models.resnet50(pretrained=True)
    model.eval()

    # 目标层（ResNet 的最后一个卷积层）
    target_layer = model.layer4[-1]

    # 创建 Grad-CAM
    gradcam = GradCAM(model, target_layer)

    # 加载图像
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open('example.jpg')
    image_tensor = transform(image).unsqueeze(0)

    # 生成 Grad-CAM
    cam = gradcam.generate(image_tensor)

    # 可视化
    visualize_gradcam(image_tensor[0], cam)
```

---

## 12.5 LIME (Local Interpretable Model-agnostic Explanations)

### 🎯 核心思想

```
为单个预测提供局部线性解释：

1. 在预测点附近采样
2. 用黑盒模型预测这些样本
3. 训练简单模型（如线性模型）拟合
4. 用简单模型解释
```

### 💻 实现

```python
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics.pairwise import euclidean_distances

class LIME:
    """LIME 解释器"""

    def __init__(self, kernel_width=0.25, n_samples=5000):
        self.kernel_width = kernel_width
        self.n_samples = n_samples

    def explain_instance(self, instance, predict_fn, num_features=10):
        """
        解释单个实例

        参数:
            instance: 要解释的实例
            predict_fn: 预测函数
            num_features: 返回最重要的特征数
        """
        # 在实例附近采样
        samples = self._generate_samples(instance)

        # 用黑盒模型预测
        predictions = predict_fn(samples)

        # 计算权重（距离越近权重越大）
        distances = euclidean_distances(samples, instance.reshape(1, -1)).ravel()
        weights = self._kernel(distances)

        # 训练线性模型
        linear_model = Ridge(alpha=1.0)
        linear_model.fit(samples, predictions, sample_weight=weights)

        # 获取特征重要性
        feature_importance = linear_model.coef_

        # 返回最重要的特征
        top_features = np.argsort(np.abs(feature_importance))[-num_features:][::-1]

        explanation = [
            (feature_idx, feature_importance[feature_idx])
            for feature_idx in top_features
        ]

        return explanation

    def _generate_samples(self, instance):
        """在实例附近生成样本"""
        n_features = len(instance)

        # 高斯扰动
        samples = np.random.normal(
            loc=instance,
            scale=1.0,
            size=(self.n_samples, n_features)
        )

        return samples

    def _kernel(self, distances):
        """核函数（距离 → 权重）"""
        return np.exp(-(distances ** 2) / (self.kernel_width ** 2))

# ==================== 图像 LIME ====================

class ImageLIME:
    """图像 LIME 解释器"""

    def __init__(self, n_samples=1000, n_segments=50):
        self.n_samples = n_samples
        self.n_segments = n_segments

    def explain_instance(self, image, predict_fn, top_labels=1):
        """
        解释图像分类

        参数:
            image: 输入图像 (H, W, C)
            predict_fn: 预测函数
            top_labels: 解释的类别数
        """
        from skimage.segmentation import quickshift

        # 超像素分割
        segments = quickshift(image, kernel_size=4, max_dist=200, ratio=0.2)
        n_segments = len(np.unique(segments))

        # 生成扰动样本
        samples = np.zeros((self.n_samples, n_segments))
        perturbed_images = []

        for i in range(self.n_samples):
            # 随机mask一些超像素
            active_segments = np.random.choice(
                [0, 1], size=n_segments, p=[0.5, 0.5]
            )
            samples[i] = active_segments

            # 生成扰动图像
            perturbed_image = image.copy()
            for seg_id in range(n_segments):
                if active_segments[seg_id] == 0:
                    perturbed_image[segments == seg_id] = 0

            perturbed_images.append(perturbed_image)

        # 预测
        perturbed_images = np.array(perturbed_images)
        predictions = predict_fn(perturbed_images)

        # 训练线性模型
        from sklearn.linear_model import Ridge

        explanations = []
        for label in range(top_labels):
            linear_model = Ridge(alpha=1.0)
            linear_model.fit(samples, predictions[:, label])

            # 获取超像素重要性
            segment_importance = linear_model.coef_

            explanations.append({
                'label': label,
                'segments': segments,
                'importance': segment_importance
            })

        return explanations

    def visualize_explanation(self, image, explanation, threshold=0.1):
        """可视化解释"""
        segments = explanation['segments']
        importance = explanation['importance']

        # 归一化重要性
        importance = (importance - importance.min()) / (importance.max() - importance.min())

        # 创建掩码
        mask = np.zeros(image.shape[:2])
        for seg_id in range(len(importance)):
            if importance[seg_id] > threshold:
                mask[segments == seg_id] = importance[seg_id]

        # 可视化
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))

        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')

        axes[1].imshow(mask, cmap='hot')
        axes[1].set_title('Importance Map')
        axes[1].axis('off')

        axes[2].imshow(image)
        axes[2].imshow(mask, cmap='hot', alpha=0.5)
        axes[2].set_title('Overlay')
        axes[2].axis('off')

        plt.tight_layout()
        plt.show()
```

---

## 12.6 SHAP (SHapley Additive exPlanations)

### 🎯 核心思想

```
基于博弈论的 Shapley 值：

每个特征对预测的边际贡献

SHAP 值性质：
  1. 局部准确性
  2. 缺失性
  3. 一致性
```

### 💻 使用 SHAP 库

```python
import shap
import numpy as np
import matplotlib.pyplot as plt

# ==================== 树模型 SHAP ====================

def explain_tree_model(model, X, feature_names=None):
    """解释树模型"""

    # 创建 SHAP 解释器
    explainer = shap.TreeExplainer(model)

    # 计算 SHAP 值
    shap_values = explainer.shap_values(X)

    # 可视化
    # 1. 摘要图
    shap.summary_plot(shap_values, X, feature_names=feature_names)

    # 2. 单个样本解释
    shap.force_plot(
        explainer.expected_value,
        shap_values[0],
        X[0],
        feature_names=feature_names,
        matplotlib=True
    )
    plt.show()

    # 3. 依赖图
    if feature_names:
        shap.dependence_plot(
            feature_names[0],
            shap_values,
            X,
            feature_names=feature_names
        )

    return shap_values

# ==================== 深度学习 SHAP ====================

def explain_deep_model(model, X, background_data):
    """解释深度学习模型"""

    # 创建 DeepExplainer
    explainer = shap.DeepExplainer(model, background_data)

    # 计算 SHAP 值
    shap_values = explainer.shap_values(X)

    # 图像可视化
    if len(X.shape) == 4:  # 图像数据
        shap.image_plot(shap_values, X)

    return shap_values

# ==================== 示例 ====================

if __name__ == '__main__':
    from sklearn.datasets import load_boston
    from sklearn.ensemble import RandomForestRegressor

    # 加载数据
    boston = load_boston()
    X, y = boston.data, boston.target

    # 训练模型
    model = RandomForestRegressor(random_state=42)
    model.fit(X, y)

    # SHAP 解释
    shap_values = explain_tree_model(
        model, X[:100], feature_names=boston.feature_names
    )
```

---

## 12.7 对抗攻击 (Adversarial Attacks)

### 🎯 什么是对抗样本？

```
对抗样本：故意设计的输入，使模型产生错误预测

x_adv = x + δ

其中 δ 是精心设计的微小扰动，人眼几乎无法察觉
```

**示例**：

```
原图：熊猫 → 预测：熊猫 (99% 置信度)
     ↓ + 微小噪声
对抗样本：熊猫? → 预测：长臂猿 (99% 置信度)
```

---

### 🔹 FGSM (Fast Gradient Sign Method)

**原理**：沿着梯度方向添加扰动

```
x_adv = x + ε · sign(∇_x L(θ, x, y))

ε: 扰动幅度
L: 损失函数
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class FGSM:
    """Fast Gradient Sign Method 攻击"""

    def __init__(self, model, epsilon=0.03):
        self.model = model
        self.epsilon = epsilon
        self.model.eval()

    def attack(self, images, labels):
        """
        生成对抗样本

        参数:
            images: 原始图像 (batch, C, H, W)
            labels: 真实标签
        """
        images = images.clone().detach().requires_grad_(True)

        # 前向传播
        outputs = self.model(images)
        loss = F.cross_entropy(outputs, labels)

        # 反向传播
        self.model.zero_grad()
        loss.backward()

        # 获取梯度符号
        data_grad = images.grad.data
        sign_data_grad = data_grad.sign()

        # 生成对抗样本
        perturbed_images = images + self.epsilon * sign_data_grad

        # 裁剪到合法范围 [0, 1]
        perturbed_images = torch.clamp(perturbed_images, 0, 1)

        return perturbed_images.detach()

# ==================== 可视化对抗攻击 ====================

def visualize_adversarial_attack(model, image, label, epsilon=0.03):
    """可视化对抗攻击效果"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    image = image.to(device)
    label = label.to(device)

    # 原始预测
    with torch.no_grad():
        output_orig = model(image.unsqueeze(0))
        pred_orig = output_orig.argmax(dim=1).item()
        conf_orig = F.softmax(output_orig, dim=1)[0, pred_orig].item()

    # 生成对抗样本
    fgsm = FGSM(model, epsilon=epsilon)
    adv_image = fgsm.attack(image.unsqueeze(0), label.unsqueeze(0))

    # 对抗样本预测
    with torch.no_grad():
        output_adv = model(adv_image)
        pred_adv = output_adv.argmax(dim=1).item()
        conf_adv = F.softmax(output_adv, dim=1)[0, pred_adv].item()

    # 扰动
    perturbation = (adv_image - image.unsqueeze(0)).squeeze()

    # 可视化
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    # 原图
    img_orig = image.permute(1, 2, 0).cpu().numpy()
    axes[0].imshow(img_orig)
    axes[0].set_title(f'Original\nPred: {pred_orig} ({conf_orig:.2%})')
    axes[0].axis('off')

    # 扰动（放大显示）
    pert_display = perturbation.permute(1, 2, 0).cpu().numpy()
    pert_display = (pert_display - pert_display.min()) / (pert_display.max() - pert_display.min())
    axes[1].imshow(pert_display)
    axes[1].set_title(f'Perturbation (×10)')
    axes[1].axis('off')

    # 对抗样本
    img_adv = adv_image.squeeze().permute(1, 2, 0).cpu().numpy()
    axes[2].imshow(img_adv)
    axes[2].set_title(f'Adversarial\nPred: {pred_adv} ({conf_adv:.2%})')
    axes[2].axis('off')

    # 差异
    diff = np.abs(img_adv - img_orig)
    axes[3].imshow(diff)
    axes[3].set_title('Absolute Difference')
    axes[3].axis('off')

    plt.tight_layout()
    plt.show()

    return pred_orig == pred_adv  # 攻击是否成功
```

---

### 🔹 PGD (Projected Gradient Descent)

**更强的攻击**：迭代版 FGSM

```python
class PGD:
    """Projected Gradient Descent 攻击"""

    def __init__(self, model, epsilon=0.03, alpha=0.01, num_iter=40):
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.num_iter = num_iter
        self.model.eval()

    def attack(self, images, labels):
        """生成对抗样本"""

        # 随机初始化
        delta = torch.zeros_like(images).uniform_(-self.epsilon, self.epsilon)
        delta.requires_grad = True

        for _ in range(self.num_iter):
            # 前向传播
            outputs = self.model(images + delta)
            loss = F.cross_entropy(outputs, labels)

            # 反向传播
            loss.backward()

            # 更新扰动
            delta.data = delta.data + self.alpha * delta.grad.sign()

            # 投影到 ε-球内
            delta.data = torch.clamp(delta.data, -self.epsilon, self.epsilon)

            # 确保在合法范围内
            delta.data = torch.clamp(images.data + delta.data, 0, 1) - images.data

            # 清空梯度
            delta.grad.zero_()

        return (images + delta).detach()
```

---

### 🔹 C&W Attack (Carlini & Wagner)

**最优化攻击**：

```
最小化: ||δ||_2 + c·loss(x+δ, t)

其中 t 是目标类别（定向攻击）
```

```python
class CWAttack:
    """Carlini & Wagner L2 攻击"""

    def __init__(self, model, c=1.0, kappa=0, learning_rate=0.01, num_iter=1000):
        self.model = model
        self.c = c
        self.kappa = kappa
        self.learning_rate = learning_rate
        self.num_iter = num_iter
        self.model.eval()

    def attack(self, images, labels, targeted=False, target_labels=None):
        """
        C&W 攻击

        参数:
            targeted: 是否为定向攻击
            target_labels: 目标类别（定向攻击时使用）
        """
        batch_size = images.size(0)

        # 使用 tanh 空间
        w = torch.zeros_like(images, requires_grad=True)
        optimizer = torch.optim.Adam([w], lr=self.learning_rate)

        best_adv = images.clone()
        best_l2 = float('inf') * torch.ones(batch_size)

        for iteration in range(self.num_iter):
            # 转换回图像空间
            adv_images = 0.5 * (torch.tanh(w) + 1)

            # 预测
            outputs = self.model(adv_images)

            # C&W 损失
            if targeted:
                # 定向攻击：最大化目标类别
                loss_adv = self._cw_loss(outputs, target_labels, targeted=True)
            else:
                # 非定向攻击：最小化真实类别
                loss_adv = self._cw_loss(outputs, labels, targeted=False)

            # L2 距离
            l2_dist = torch.norm((adv_images - images).view(batch_size, -1), p=2, dim=1)

            # 总损失
            loss = l2_dist.sum() + self.c * loss_adv.sum()

            # 优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 更新最佳样本
            for i in range(batch_size):
                if l2_dist[i] < best_l2[i]:
                    pred = outputs[i].argmax().item()
                    if targeted:
                        if pred == target_labels[i].item():
                            best_l2[i] = l2_dist[i]
                            best_adv[i] = adv_images[i]
                    else:
                        if pred != labels[i].item():
                            best_l2[i] = l2_dist[i]
                            best_adv[i] = adv_images[i]

        return best_adv.detach()

    def _cw_loss(self, outputs, labels, targeted):
        """C&W 损失函数"""
        real = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)

        # 获取除真实类别外的最大 logit
        other, _ = torch.max(outputs - 1e9 * F.one_hot(labels, outputs.size(1)), dim=1)

        if targeted:
            # 定向：max(other - real, -kappa)
            loss = torch.clamp(other - real, min=-self.kappa)
        else:
            # 非定向：max(real - other, -kappa)
            loss = torch.clamp(real - other, min=-self.kappa)

        return loss
```

---

## 12.8 对抗防御

### 🔹 对抗训练 (Adversarial Training)

**最有效的防御方法**

```python
def adversarial_training(model, train_loader, num_epochs=10, epsilon=0.03):
    """对抗训练"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    # 创建攻击器
    fgsm = FGSM(model, epsilon=epsilon)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_clean = 0
        correct_adv = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # 生成对抗样本
            adv_images = fgsm.attack(images, labels)

            # 合并干净样本和对抗样本
            all_images = torch.cat([images, adv_images])
            all_labels = torch.cat([labels, labels])

            # 前向传播
            outputs = model(all_images)
            loss = criterion(outputs, all_labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()

            # 准确率（分别统计）
            outputs_clean = outputs[:len(images)]
            outputs_adv = outputs[len(images):]

            _, pred_clean = outputs_clean.max(1)
            _, pred_adv = outputs_adv.max(1)

            correct_clean += pred_clean.eq(labels).sum().item()
            correct_adv += pred_adv.eq(labels).sum().item()
            total += labels.size(0)

        print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, '
              f'Clean Acc={100.*correct_clean/total:.2f}%, '
              f'Adv Acc={100.*correct_adv/total:.2f}%')

    return model
```

---

### 🔹 输入变换防御

```python
class InputTransformDefense:
    """输入变换防御"""

    def __init__(self, model):
        self.model = model
        self.model.eval()

    def predict_with_defense(self, images):
        """带防御的预测"""

        # 1. JPEG 压缩
        images_jpeg = self._jpeg_compression(images)

        # 2. 随机调整大小和填充
        images_resized = self._random_resize_pad(images_jpeg)

        # 3. 位深度降低
        images_quantized = self._bit_depth_reduction(images_resized)

        # 预测
        with torch.no_grad():
            outputs = self.model(images_quantized)

        return outputs

    def _jpeg_compression(self, images, quality=75):
        """JPEG 压缩"""
        from PIL import Image
        from io import BytesIO

        compressed = []
        for img in images:
            img_pil = transforms.ToPILImage()(img.cpu())
            buffer = BytesIO()
            img_pil.save(buffer, format='JPEG', quality=quality)
            buffer.seek(0)
            img_compressed = Image.open(buffer)
            compressed.append(transforms.ToTensor()(img_compressed))

        return torch.stack(compressed).to(images.device)

    def _random_resize_pad(self, images, resize_factor=0.9):
        """随机调整大小和填充"""
        B, C, H, W = images.shape
        new_size = int(H * resize_factor)

        resized = F.interpolate(images, size=new_size, mode='bilinear')

        # 随机位置填充
        pad_size = H - new_size
        pad_top = torch.randint(0, pad_size + 1, (1,)).item()
        pad_left = torch.randint(0, pad_size + 1, (1,)).item()

        padded = F.pad(resized,
                      (pad_left, pad_size - pad_left,
                       pad_top, pad_size - pad_top))

        return padded

    def _bit_depth_reduction(self, images, bits=4):
        """位深度降低"""
        levels = 2 ** bits
        images_quantized = torch.round(images * (levels - 1)) / (levels - 1)
        return images_quantized
```

---

### 🔹 集成防御

```python
class EnsembleDefense:
    """集成防御"""

    def __init__(self, models):
        self.models = models
        for model in self.models:
            model.eval()

    def predict(self, images):
        """集成预测"""

        all_outputs = []

        with torch.no_grad():
            for model in self.models:
                outputs = model(images)
                all_outputs.append(F.softmax(outputs, dim=1))

        # 平均概率
        ensemble_output = torch.stack(all_outputs).mean(dim=0)

        return ensemble_output
```

---

## 12.9 鲁棒性评估

```python
class RobustnessEvaluator:
    """鲁棒性评估器"""

    def __init__(self, model, test_loader, device):
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.model.eval()

    def evaluate_clean_accuracy(self):
        """评估干净样本准确率"""
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        accuracy = 100. * correct / total
        print(f"Clean Accuracy: {accuracy:.2f}%")
        return accuracy

    def evaluate_adversarial_robustness(self, attack, epsilons=[0.01, 0.03, 0.1]):
        """评估对抗鲁棒性"""

        results = {}

        for epsilon in epsilons:
            attack.epsilon = epsilon

            correct = 0
            total = 0

            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                # 生成对抗样本
                adv_images = attack.attack(images, labels)

                # 预测
                with torch.no_grad():
                    outputs = self.model(adv_images)
                    _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            accuracy = 100. * correct / total
            results[epsilon] = accuracy
            print(f"Adversarial Accuracy (ε={epsilon}): {accuracy:.2f}%")

        return results

    def plot_robustness_curve(self, results):
        """绘制鲁棒性曲线"""
        epsilons = sorted(results.keys())
        accuracies = [results[eps] for eps in epsilons]

        plt.figure(figsize=(10, 6))
        plt.plot(epsilons, accuracies, 'bo-', linewidth=2, markersize=8)
        plt.xlabel('Perturbation Magnitude (ε)')
        plt.ylabel('Accuracy (%)')
        plt.title('Model Robustness to Adversarial Attacks')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

# ==================== 使用示例 ====================

if __name__ == '__main__':
    # 评估模型鲁棒性
    evaluator = RobustnessEvaluator(model, test_loader, device)

    # 干净准确率
    evaluator.evaluate_clean_accuracy()

    # 对抗鲁棒性
    fgsm = FGSM(model)
    results = evaluator.evaluate_adversarial_robustness(
        fgsm, epsilons=[0.0, 0.01, 0.03, 0.05, 0.1, 0.2]
    )

    # 绘制曲线
    evaluator.plot_robustness_curve(results)
```

---

## 📝 本章作业

### 作业 1：模型可解释性

```python
# 在图像分类任务上：
# 1. 实现 Saliency Map
# 2. 实现 Grad-CAM
# 3. 使用 LIME 解释预测
# 4. 对比三种方法的结果
# 5. 分析模型关注的区域是否合理
```

### 作业 2：对抗攻击实验

```python
# 实现并对比：
# 1. FGSM
# 2. PGD
# 3. C&W
#
# 评估：
#   - 攻击成功率
#   - 扰动大小（L2, L∞）
#   - 可感知性
#   - 计算时间
```

### 作业 3：对抗防御

```python
# 实现并评估防御方法：
# 1. 对抗训练
# 2. 输入变换
# 3. 集成防御
#
# 对比：
#   - 干净样本准确率
#   - 对抗样本准确率
#   - 鲁棒性 vs 准确性 trade-off
```

### 作业 4：可信 AI 系统

```python
# 设计一个可信 AI 系统：
# 1. 提供预测解释
# 2. 评估预测置信度
# 3. 检测对抗样本
# 4. 提供不确定性估计
#
# 在医疗或金融场景中测试
```

---

## 🔑 本章关键概念

| 概念 | 说明 |
|------|------|
| 可解释性 | 理解模型决策过程 |
| Saliency Map | 基于梯度的显著性 |
| Grad-CAM | 类激活映射 |
| LIME | 局部线性解释 |
| SHAP | Shapley 值解释 |
| 对抗样本 | 精心设计的扰动输入 |
| FGSM | 快速梯度符号攻击 |
| PGD | 投影梯度下降攻击 |
| C&W | 优化攻击 |
| 对抗训练 | 最有效的防御 |
| 鲁棒性 | 对扰动的抵抗能力 |

---

需要我继续写**第十三章：大语言模型时代**吗？这将是最后一章，涵盖 LLM、Prompt Engineering、In-Context Learning 等前沿话题。

-----
