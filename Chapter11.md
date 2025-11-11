# 第十一章：无监督学习 (Unsupervised Learning)

## 📌 章节目标
- 理解无监督学习的核心思想
- 掌握聚类算法（K-means, DBSCAN, 层次聚类）
- 学习降维技术（PCA, t-SNE, UMAP）
- 了解自监督学习方法
- 实战：数据探索、特征学习、异常检测

---

## 11.1 无监督学习概述

### 🎯 什么是无监督学习？

**定义**：从无标签数据中学习数据的内在结构和模式

**与监督学习的区别**：

```
监督学习：
  输入: (X, y)  有标签
  目标: 学习 f: X → y

无监督学习：
  输入: X only  无标签
  目标: 发现数据的隐藏结构
```

### 📊 主要任务

#### **1. 聚类 (Clustering)**

```
目标：将相似样本分组

应用：
  - 客户分群
  - 图像分割
  - 文档组织
  - 基因分析
```

#### **2. 降维 (Dimensionality Reduction)**

```
目标：在低维空间保留数据特性

应用：
  - 可视化
  - 数据压缩
  - 噪声消除
  - 特征提取
```

#### **3. 密度估计 (Density Estimation)**

```
目标：估计数据的概率分布

应用：
  - 异常检测
  - 生成模型
```

#### **4. 表示学习 (Representation Learning)**

```
目标：学习有用的数据表示

应用：
  - 自监督学习
  - 预训练模型
```

---

## 11.2 聚类算法

### 🔹 K-Means 聚类

#### **算法原理**

```
目标：最小化类内距离平方和

J = ∑_{k=1}^K ∑_{x∈C_k} ||x - μ_k||²

算法步骤：
1. 随机初始化 K 个中心 μ_k
2. 分配：每个点分配到最近的中心
3. 更新：重新计算每个簇的中心
4. 重复 2-3 直到收敛
```

#### **实现**

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

class KMeans:
    """K-Means 聚类"""

    def __init__(self, n_clusters=3, max_iters=100, random_state=None):
        self.n_clusters = n_clusters
        self.max_iters = max_iters
        self.random_state = random_state
        self.centers = None
        self.labels = None

    def fit(self, X):
        """
        训练 K-Means

        参数:
            X: (n_samples, n_features)
        """
        np.random.seed(self.random_state)
        n_samples = X.shape[0]

        # 随机初始化中心
        random_indices = np.random.choice(n_samples, self.n_clusters, replace=False)
        self.centers = X[random_indices]

        for iteration in range(self.max_iters):
            # 分配样本到最近的中心
            labels = self._assign_clusters(X)

            # 更新中心
            new_centers = self._update_centers(X, labels)

            # 检查收敛
            if np.allclose(self.centers, new_centers):
                print(f"收敛于第 {iteration+1} 次迭代")
                break

            self.centers = new_centers

        self.labels = labels
        return self

    def _assign_clusters(self, X):
        """分配样本到最近的簇"""
        distances = np.sqrt(((X[:, np.newaxis] - self.centers) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)

    def _update_centers(self, X, labels):
        """更新簇中心"""
        new_centers = np.zeros((self.n_clusters, X.shape[1]))

        for k in range(self.n_clusters):
            cluster_points = X[labels == k]
            if len(cluster_points) > 0:
                new_centers[k] = cluster_points.mean(axis=0)
            else:
                # 如果簇为空，重新随机初始化
                new_centers[k] = X[np.random.randint(X.shape[0])]

        return new_centers

    def predict(self, X):
        """预测新样本的簇"""
        return self._assign_clusters(X)

    def fit_predict(self, X):
        """训练并预测"""
        self.fit(X)
        return self.labels

# ==================== 可视化 K-Means ====================

def visualize_kmeans(X, kmeans, title='K-Means Clustering'):
    """可视化 K-Means 结果"""

    fig, ax = plt.subplots(figsize=(10, 8))

    # 绘制数据点
    scatter = ax.scatter(X[:, 0], X[:, 1], c=kmeans.labels,
                        cmap='viridis', alpha=0.6, s=50)

    # 绘制中心
    ax.scatter(kmeans.centers[:, 0], kmeans.centers[:, 1],
              c='red', marker='X', s=200, edgecolor='black',
              linewidth=2, label='Centroids')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(title)
    ax.legend()
    plt.colorbar(scatter, label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==================== 肘部法则 (Elbow Method) ====================

def elbow_method(X, max_k=10):
    """使用肘部法则确定最佳 K 值"""

    inertias = []
    K_range = range(1, max_k + 1)

    for k in K_range:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)

        # 计算簇内平方和 (inertia)
        labels = kmeans.labels
        inertia = 0
        for i in range(k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                inertia += ((cluster_points - kmeans.centers[i]) ** 2).sum()

        inertias.append(inertia)

    # 绘制肘部曲线
    plt.figure(figsize=(10, 6))
    plt.plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (K)')
    plt.ylabel('Inertia (Within-cluster Sum of Squares)')
    plt.title('Elbow Method for Optimal K')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return inertias

# ==================== 示例 ====================

if __name__ == '__main__':
    # 生成合成数据
    X, y_true = make_blobs(n_samples=300, centers=4,
                           n_features=2, random_state=42)

    # 肘部法则
    print("使用肘部法则确定最佳 K...")
    elbow_method(X, max_k=10)

    # K-Means 聚类
    kmeans = KMeans(n_clusters=4, random_state=42)
    kmeans.fit(X)

    print(f"\n簇中心:\n{kmeans.centers}")
    print(f"簇分配: {kmeans.labels}")

    # 可视化
    visualize_kmeans(X, kmeans)
```

---

### 🔹 DBSCAN (Density-Based Spatial Clustering)

#### **算法原理**

```
基于密度的聚类：
  - 核心点：ε 邻域内至少有 MinPts 个点
  - 边界点：在核心点的 ε 邻域内，但自己不是核心点
  - 噪声点：既不是核心点也不是边界点

优势：
  ✓ 可以发现任意形状的簇
  ✓ 不需要预先指定簇数
  ✓ 能识别噪声点

参数：
  - ε (epsilon): 邻域半径
  - MinPts: 最小点数
```

#### **实现**

```python
from sklearn.neighbors import NearestNeighbors

class DBSCAN:
    """DBSCAN 聚类"""

    def __init__(self, eps=0.5, min_samples=5):
        self.eps = eps
        self.min_samples = min_samples
        self.labels = None

    def fit(self, X):
        """训练 DBSCAN"""
        n_samples = X.shape[0]

        # 计算所有点的邻域
        neighbors_model = NearestNeighbors(radius=self.eps)
        neighbors_model.fit(X)
        neighborhoods = neighbors_model.radius_neighbors(X, return_distance=False)

        # 初始化标签（-1 表示未分类）
        labels = np.full(n_samples, -1)

        # 当前簇 ID
        cluster_id = 0

        for i in range(n_samples):
            # 如果已分类，跳过
            if labels[i] != -1:
                continue

            # 获取邻域
            neighbors = neighborhoods[i]

            # 如果不是核心点，标记为噪声（暂时）
            if len(neighbors) < self.min_samples:
                labels[i] = -1
                continue

            # 开始新簇
            labels[i] = cluster_id

            # 种子集合（待扩展的点）
            seeds = set(neighbors) - {i}

            while seeds:
                q = seeds.pop()

                # 如果是噪声点，改为边界点
                if labels[q] == -1:
                    labels[q] = cluster_id

                # 如果已分类到其他簇，跳过
                if labels[q] != -1:
                    continue

                labels[q] = cluster_id

                # 如果 q 也是核心点，扩展种子集
                q_neighbors = neighborhoods[q]
                if len(q_neighbors) >= self.min_samples:
                    seeds.update(q_neighbors)

            cluster_id += 1

        self.labels = labels
        return self

    def fit_predict(self, X):
        """训练并返回标签"""
        self.fit(X)
        return self.labels

# ==================== 可视化 DBSCAN ====================

def visualize_dbscan(X, dbscan):
    """可视化 DBSCAN 结果"""

    fig, ax = plt.subplots(figsize=(10, 8))

    # 核心样本 mask
    unique_labels = set(dbscan.labels)
    colors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        if k == -1:
            # 噪声点用黑色表示
            col = [0, 0, 0, 1]

        class_member_mask = (dbscan.labels == k)

        xy = X[class_member_mask]
        ax.scatter(xy[:, 0], xy[:, 1], c=[col],
                  s=50, alpha=0.6,
                  label=f'Cluster {k}' if k != -1 else 'Noise')

    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title('DBSCAN Clustering')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==================== 示例 ====================

if __name__ == '__main__':
    # 生成月牙形数据
    from sklearn.datasets import make_moons
    X, _ = make_moons(n_samples=300, noise=0.05, random_state=42)

    # DBSCAN 聚类
    dbscan = DBSCAN(eps=0.2, min_samples=5)
    dbscan.fit(X)

    print(f"发现 {len(set(dbscan.labels)) - (1 if -1 in dbscan.labels else 0)} 个簇")
    print(f"噪声点数: {list(dbscan.labels).count(-1)}")

    # 可视化
    visualize_dbscan(X, dbscan)
```

---

### 🔹 层次聚类 (Hierarchical Clustering)

#### **算法原理**

```
两种策略：

1. 凝聚 (Agglomerative)：自底向上
   - 每个点初始化为一个簇
   - 逐步合并最相似的簇
   - 直到形成一个大簇或达到停止条件

2. 分裂 (Divisive)：自顶向下
   - 所有点初始化为一个簇
   - 逐步分裂簇

链接方法 (Linkage):
  - Single: 最近点距离
  - Complete: 最远点距离
  - Average: 平均距离
  - Ward: 最小化方差
```

#### **实现**

```python
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import pdist, squareform

class AgglomerativeClustering:
    """凝聚层次聚类"""

    def __init__(self, n_clusters=2, linkage='ward'):
        self.n_clusters = n_clusters
        self.linkage_method = linkage
        self.labels = None
        self.linkage_matrix = None

    def fit(self, X):
        """训练层次聚类"""
        # 计算链接矩阵
        self.linkage_matrix = linkage(X, method=self.linkage_method)

        # 根据 n_clusters 切割树状图
        from scipy.cluster.hierarchy import fcluster
        self

-----

> continue

.labels = fcluster(self.linkage_matrix,
                                     self.n_clusters,
                                     criterion='maxclust') - 1  # 转为 0-based

        return self

    def fit_predict(self, X):
        """训练并返回标签"""
        self.fit(X)
        return self.labels

# ==================== 树状图可视化 ====================

def plot_dendrogram(X, method='ward', title='Hierarchical Clustering Dendrogram'):
    """绘制树状图"""

    plt.figure(figsize=(12, 6))

    # 计算链接矩阵
    Z = linkage(X, method=method)

    # 绘制树状图
    dendrogram(Z)

    plt.xlabel('Sample Index')
    plt.ylabel('Distance')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==================== 示例 ====================

if __name__ == '__main__':
    # 生成数据
    X, _ = make_blobs(n_samples=100, centers=3,
                      n_features=2, random_state=42)

    # 绘制树状图
    plot_dendrogram(X, method='ward')

    # 层次聚类
    hc = AgglomerativeClustering(n_clusters=3, linkage='ward')
    labels = hc.fit_predict(X)

    # 可视化结果
    plt.figure(figsize=(10, 8))
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', s=50)
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Hierarchical Clustering Result')
    plt.colorbar(label='Cluster')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

---

## 11.3 降维技术

### 🔹 主成分分析 (PCA)

#### **算法原理**

```
目标：找到方差最大的方向

数学表达：
1. 中心化数据：X_centered = X - mean(X)
2. 计算协方差矩阵：C = (1/n)·X^T·X
3. 特征值分解：C = V·Λ·V^T
4. 选择前 k 个主成分

投影：
  Z = X·V_k  (降维后的数据)

重构：
  X_reconstructed = Z·V_k^T
```

#### **实现**

```python
class PCA:
    """主成分分析"""

    def __init__(self, n_components=2):
        self.n_components = n_components
        self.components = None
        self.mean = None
        self.explained_variance = None
        self.explained_variance_ratio = None

    def fit(self, X):
        """训练 PCA"""
        # 中心化
        self.mean = np.mean(X, axis=0)
        X_centered = X - self.mean

        # 计算协方差矩阵
        cov_matrix = np.cov(X_centered.T)

        # 特征值分解
        eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

        # 按特征值降序排序
        idx = eigenvalues.argsort()[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # 选择前 n_components 个主成分
        self.components = eigenvectors[:, :self.n_components]
        self.explained_variance = eigenvalues[:self.n_components]
        self.explained_variance_ratio = (
            self.explained_variance / eigenvalues.sum()
        )

        return self

    def transform(self, X):
        """降维"""
        X_centered = X - self.mean
        return np.dot(X_centered, self.components)

    def fit_transform(self, X):
        """训练并降维"""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, Z):
        """重构"""
        return np.dot(Z, self.components.T) + self.mean

# ==================== 可视化 PCA ====================

def visualize_pca(X, y=None, title='PCA Visualization'):
    """可视化 PCA 降维结果"""

    # PCA 降到 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure(figsize=(12, 5))

    # 降维后的数据
    plt.subplot(1, 2, 1)
    if y is not None:
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1],
                            c=y, cmap='viridis', alpha=0.6)
        plt.colorbar(scatter, label='Class')
    else:
        plt.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)

    plt.xlabel(f'PC1 ({pca.explained_variance_ratio[0]:.2%})')
    plt.ylabel(f'PC2 ({pca.explained_variance_ratio[1]:.2%})')
    plt.title('PCA Projection')
    plt.grid(True, alpha=0.3)

    # 方差解释比例
    plt.subplot(1, 2, 2)
    n_components = min(10, X.shape[1])
    pca_full = PCA(n_components=n_components)
    pca_full.fit(X)

    cumsum = np.cumsum(pca_full.explained_variance_ratio)

    plt.plot(range(1, n_components+1),
            pca_full.explained_variance_ratio,
            'bo-', label='Individual')
    plt.plot(range(1, n_components+1),
            cumsum,
            'rs-', label='Cumulative')

    plt.xlabel('Principal Component')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Variance Explained')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# ==================== 示例：手写数字 ====================

if __name__ == '__main__':
    from sklearn.datasets import load_digits

    # 加载 MNIST 数字数据集
    digits = load_digits()
    X, y = digits.data, digits.target

    print(f"原始维度: {X.shape}")

    # PCA 降维
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    print(f"降维后: {X_pca.shape}")
    print(f"方差解释比例: {pca.explained_variance_ratio}")

    # 可视化
    visualize_pca(X, y, title='PCA on MNIST Digits')
```

---

### 🔹 t-SNE (t-Distributed Stochastic Neighbor Embedding)

#### **算法原理**

```
目标：保持局部结构

步骤：
1. 计算高维空间中点对的相似度 p_ij
2. 在低维空间随机初始化
3. 计算低维空间中的相似度 q_ij
4. 最小化 KL 散度：KL(P||Q)

特点：
  ✓ 擅长可视化聚类结构
  ✓ 保留局部邻域
  ✗ 计算复杂度高 O(n²)
  ✗ 全局结构不保证
  ✗ 每次运行结果不同
```

#### **使用 sklearn**

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def visualize_tsne(X, y, perplexity=30, title='t-SNE Visualization'):
    """t-SNE 可视化"""

    print("运行 t-SNE...")
    tsne = TSNE(n_components=2, perplexity=perplexity,
                random_state=42, n_iter=1000)
    X_tsne = tsne.fit_transform(X)

    plt.figure(figsize=(10, 8))

    if y is not None:
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1],
                            c=y, cmap='tab10', alpha=0.6, s=20)
        plt.colorbar(scatter, label='Class')
    else:
        plt.scatter(X_tsne[:, 0], X_tsne[:, 1], alpha=0.6, s=20)

    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

# ==================== 对比 PCA 和 t-SNE ====================

def compare_pca_tsne(X, y):
    """对比 PCA 和 t-SNE"""

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    scatter1 = axes[0].scatter(X_pca[:, 0], X_pca[:, 1],
                              c=y, cmap='tab10', alpha=0.6, s=20)
    axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio[0]:.2%})')
    axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio[1]:.2%})')
    axes[0].set_title('PCA')
    axes[0].grid(True, alpha=0.3)
    plt.colorbar(scatter1, ax=axes[0], label='Class')

    # t-SNE
    print("运行 t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    X_tsne = tsne.fit_transform(X)

    scatter2 = axes[1].scatter(X_tsne[:, 0], X_tsne[:, 1],
                              c=y, cmap='tab10', alpha=0.6, s=20)
    axes[1].set_xlabel('t-SNE 1')
    axes[1].set_ylabel('t-SNE 2')
    axes[1].set_title('t-SNE')
    axes[1].grid(True, alpha=0.3)
    plt.colorbar(scatter2, ax=axes[1], label='Class')

    plt.tight_layout()
    plt.show()

# ==================== 示例 ====================

if __name__ == '__main__':
    from sklearn.datasets import load_digits

    digits = load_digits()
    X, y = digits.data, digits.target

    # 降采样（t-SNE 很慢）
    from sklearn.model_selection import train_test_split
    X_sample, _, y_sample, _ = train_test_split(
        X, y, train_size=500, stratify=y, random_state=42
    )

    # 对比
    compare_pca_tsne(X_sample, y_sample)
```

---

### 🔹 UMAP (Uniform Manifold Approximation and Projection)

```
优势：
  ✓ 比 t-SNE 快
  ✓ 更好地保留全局结构
  ✓ 支持新数据的 transform

使用：
```

```python
import umap

def visualize_umap(X, y, title='UMAP Visualization'):
    """UMAP 可视化"""

    print("运行 UMAP...")
    reducer = umap.UMAP(n_components=2, random_state=42)
    X_umap = reducer.fit_transform(X)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(X_umap[:, 0], X_umap[:, 1],
                         c=y, cmap='tab10', alpha=0.6, s=20)
    plt.colorbar(scatter, label='Class')
    plt.xlabel('UMAP 1')
    plt.ylabel('UMAP 2')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
```

---

## 11.4 自监督学习 (Self-Supervised Learning)

### 🎯 核心思想

```
从无标签数据中自动生成监督信号

常见预训练任务：

图像：
  - 旋转预测
  - 拼图求解
  - 图像修复
  - 对比学习

文本：
  - 掩码语言模型 (MLM)
  - 下一句预测 (NSP)
  - 自回归语言模型
```

### 🔹 图像对比学习：SimCLR

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimCLR(nn.Module):
    """SimCLR 对比学习框架"""

    def __init__(self, base_encoder, projection_dim=128):
        super(SimCLR, self).__init__()

        # 编码器（如 ResNet）
        self.encoder = base_encoder

        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(base_encoder.output_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim)
        )

    def forward(self, x):
        # 提取特征
        h = self.encoder(x)

        # 投影
        z = self.projector(h)

        # L2 归一化
        z = F.normalize(z, dim=1)

        return z

class NTXentLoss(nn.Module):
    """归一化温度交叉熵损失"""

    def __init__(self, temperature=0.5):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        参数:
            z_i, z_j: 两个增强视图的表示
        """
        batch_size = z_i.size(0)

        # 拼接
        z = torch.cat([z_i, z_j], dim=0)  # (2B, D)

        # 计算相似度矩阵
        sim = torch.mm(z, z.T) / self.temperature  # (2B, 2B)

        # 掩码：去掉对角线
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim = sim.masked_fill(mask, -1e9)

        # 正样本：对角块外的对应位置
        positive_pairs = torch.arange(batch_size).to(z.device)
        positive_pairs = torch.cat([
            positive_pairs + batch_size,  # z_i 的正样本是 z_j
            positive_pairs                 # z_j 的正样本是 z_i
        ])

        # 交叉熵损失
        loss = F.cross_entropy(sim, positive_pairs)

        return loss

# ==================== 数据增强 ====================

from torchvision import transforms

def get_simclr_augmentation():
    """SimCLR 数据增强"""

    color_jitter = transforms.ColorJitter(
        brightness=0.8, contrast=0.8,
        saturation=0.8, hue=0.2
    )

    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([color_jitter], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

# ==================== 训练 SimCLR ====================

def train_simclr(model, dataloader, num_epochs=100):
    """训练 SimCLR"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = NTXentLoss(temperature=0.5)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for (x_i, x_j), _ in dataloader:
            x_i, x_j = x_i.to(device), x_j.to(device)

            # 前向传播
            z_i = model(x_i)
            z_j = model(x_j)

            # 计算损失
            loss = criterion(z_i, z_j)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}')

    return model
```

---

### 🔹 自编码器 (Autoencoder)

```python
class Autoencoder(nn.Module):
    """自编码器"""

    def __init__(self, input_dim, encoding_dim):
        super(Autoencoder, self).__init__()

        # 编码器
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, encoding_dim)
        )

        # 解码器
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
            nn.Sigmoid()  # 输出 [0, 1]
        )

    def forward(self, x):
        # 编码
        encoded = self.encoder(x)

        # 解码
        decoded = self.decoder(encoded)

        return decoded

    def encode(self, x):
        """仅编码"""
        return self.encoder(x)

# ==================== 训练自编码器 ====================

def train_autoencoder(model, dataloader, num_epochs=50):
    """训练自编码器"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for data, _ in dataloader:
            data = data.to(device)
            data = data.view(data.size(0), -1)  # Flatten

            # 前向传播
            reconstructed = model(data)

            # 重构损失
            loss = criterion(reconstructed, data)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}: Loss = {total_loss/len(dataloader):.4f}')

    return model

# ==================== 可视化重构 ====================

def visualize_reconstruction(model, dataloader, num_images=10):
    """可视化重构结果"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()

    # 获取一批数据
    data, _ = next(iter(dataloader))
    data = data[:num_images].to(device)

    # 重构
    with torch.no_grad():
        data_flat = data.view(data.size(0), -1)
        reconstructed = model(data_flat)
        reconstructed = reconstructed.view_as(data)

    # 绘图
    fig, axes = plt.subplots(2, num_images, figsize=(num_images*2, 4))

    for i in range(num_images):
        # 原图
        axes[0, i].imshow(data[i].cpu().squeeze(), cmap='gray')
        axes[0, i].axis('off')
        if i == 0:
            axes[0, i].set_title('Original', fontsize=12)

        # 重构
        axes[1, i].imshow(reconstructed[i].cpu().squeeze(), cmap='gray')
        axes[1, i].axis('off')
        if i == 0:
            axes[1, i].set_title('Reconstructed', fontsize=12)

    plt.tight_layout()
    plt.show()

# ==================== 示例 ====================

if __name__ == '__main__':
    from torchvision.datasets import MNIST
    from torch.utils.data import DataLoader

    # 加载 MNIST
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_dataset = MNIST(root='./data', train=True,
                         download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)

    # 创建自编码器
    autoencoder = Autoencoder(input_dim=784, encoding_dim=32)

    # 训练
    print("训练自编码器...")
    train_autoencoder(autoencoder, train_loader, num_epochs=20)

    # 可视化
    visualize_reconstruction(autoencoder, train_loader)
```

---

### 🔹 变分自编码器 (VAE)

```python
class VAE(nn.Module):
    """变分自编码器"""

    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()

        # 编码器
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc21 = nn.Linear(512, latent_dim)  # 均值
        self.fc22 = nn.Linear(512, latent_dim)  # 对数方差

        # 解码器
        self.fc3 = nn.Linear(latent_dim, 512)
        self.fc4 = nn.Linear(512, input_dim)

    def encode(self, x):
        """编码为均值和方差"""
        h = F.relu(self.fc1(x))
        mu = self.fc21(h)
        logvar = self.fc22(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """解码"""
        h = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

def vae_loss(recon_x, x, mu, logvar):
    """VAE 损失函数"""
    # 重构损失
    recon_loss = F.binary_cross_entropy(recon_x, x, reduction='sum')

    # KL 散度
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return recon_loss + kl_loss

# 训练类似自编码器，但使用 vae_loss
```

---

## 11.5 实战：客户分群

```python
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

# ==================== 生成模拟客户数据 ====================

def generate_customer_data(n_samples=1000):
    """生成模拟客户数据"""

    np.random.seed(42)

    data = {
        'customer_id': range(n_samples),
        'age': np.random.randint(18, 70, n_samples),
        'income': np.random.lognormal(10, 1, n_samples),
        'spending_score': np.random.randint(1, 100, n_samples),
        'num_purchases': np.random.poisson(5, n_samples),
        'avg_purchase_value': np.random.gamma(50, 2, n_samples),
        'days_since_last_purchase': np.random.exponential(30, n_samples)
    }

    df = pd.DataFrame(data)
    return df

# ==================== 客户分群流程 ====================

class CustomerSegmentation:
    """客户分群分析"""

    def __init__(self, n_clusters=4):
        self.n_clusters = n_clusters
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=2)
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    def preprocess(self, df):
        """数据预处理"""
        # 选择数值特征
        features = ['age', 'income', 'spending_score',
                   'num_purchases', 'avg_purchase_value',
                   'days_since_last_purchase']

        X = df[features].values

        # 标准化
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled, features

    def fit(self, df):
        """训练分群模型"""
        X_scaled, features = self.preprocess(df)

        # K-Means 聚类
        labels = self.kmeans.fit_predict(X_scaled)

        # PCA 降维用于可视化
        X_pca = self.pca.fit_transform(X_scaled)

        # 添加到 DataFrame
        df['cluster'] = labels
        df['pca1'] = X_pca[:, 0]
        df['pca2'] = X_pca[:, 1]

        return df

    def analyze_clusters(self, df):
        """分析簇特征"""
        features = ['age', 'income', 'spending_score',
                   'num_purchases', 'avg_purchase_value',
                   'days_since_last_purchase']

        print("\n各簇统计信息:")
        print("="*80)

        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['cluster'] == cluster_id]

            print(f"\n簇 {cluster_id} (n={len(cluster_data)}):")
            print(cluster_data[features].describe().T[['mean', 'std']])

        return df.groupby('cluster')[features].mean()

    def visualize(self, df):
        """可视化分群结果"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. PCA 空间中的簇
        axes[0, 0].scatter(df['pca1'], df['pca2'],
                          c=df['cluster'], cmap='viridis',
                          alpha=0.6, s=50)
        axes[0, 0].set_xlabel('PC1')
        axes[0, 0].set_ylabel('PC2')
        axes[0, 0].set_title('Customer Segments in PCA Space')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. 年龄 vs 收入
        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['cluster'] == cluster_id]
            axes[0, 1].scatter(cluster_data['age'],
                             cluster_data['income'],
                             label=f'Cluster {cluster_id}',
                             alpha=0.6, s=30)

        axes[0, 1].set_xlabel('Age')
        axes[0, 1].set_ylabel('Income')
        axes[0, 1].set_title('Age vs Income by Cluster')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 消费分数分布
        for cluster_id in range(self.n_clusters):
            cluster_data = df[df['cluster'] == cluster_id]
            axes[1, 0].hist(cluster_data['spending_score'],
                          alpha=0.5, bins=20,
                          label=f'Cluster {cluster_id}')

        axes[1, 0].set_xlabel('Spending Score')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Spending Score Distribution')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)

        # 4. 簇大小
        cluster_sizes = df['cluster'].value_counts().sort_index()
        axes[1, 1].bar(cluster_sizes.index, cluster_sizes.values,
                      color='steelblue', alpha=0.7)
        axes[1, 1].set_xlabel('Cluster')
        axes[1, 1].set_ylabel('Number of Customers')
        axes[1, 1].set_title('Cluster Sizes')
        axes[1, 1].grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.show()

    def recommend_actions(self, cluster_stats):
        """为每个簇推荐营销策略"""
        print("\n营销策略推荐:")
        print("="*80)

        for cluster_id in range(self.n_clusters):
            stats = cluster_stats.loc[cluster_id]

            print(f"\n簇 {cluster_id}:")

            if stats['income'] > cluster_stats['income'].median():
                if stats['spending_score'] > cluster_stats['spending_score'].median():
                    print("  类型: 高价值客户 💎")
                    print("  策略: VIP 服务、高端产品推荐")
                else:
                    print("  类型: 潜力客户 📈")
                    print("  策略: 个性化推荐、促销活动")
            else:
                if stats['spending_score'] > cluster_stats['spending_score'].median():
                    print("  类型: 活跃客户 ⭐")
                    print("  策略: 忠诚度计划、会员优惠")
                else:
                    print("  类型: 低活跃客户 💤")
                    print("  策略: 激活campaign、折扣优惠")

# ==================== 主程序 ====================

if __name__ == '__main__':
    # 生成数据
    df = generate_customer_data(n_samples=1000)

    print("客户数据预览:")
    print(df.head())

    # 客户分群
    segmentation = CustomerSegmentation(n_clusters=4)
    df = segmentation.fit(df)

    # 分析簇
    cluster_stats = segmentation.analyze_clusters(df)

    # 可视化
    segmentation.visualize(df)

    # 推荐策略
    segmentation.recommend_actions(cluster_stats)
```

---

## 📝 本章作业

### 作业 1：聚类对比

```python
# 在同一数据集上对比：
# 1. K-Means
# 2. DBSCAN
# 3. 层次聚类

# 评估指标：
#   - Silhouette Score
#   - Davies-Bouldin Index
#   - Calinski-Harabasz Index

# 分析：
#   - 哪种算法最适合你的数据？
#   - 不同参数的影响
```

### 作业 2：降维技术对比

```python
# 在 MNIST 或 Fashion-MNIST 上对比：
# 1. PCA
# 2. t-SNE
# 3. UMAP

# 评估：
#   - 可视化效果
#   - 运行时间
#   - 保留的信息量
#   - 在降维后数据上训练分类器的性能
```

### 作业 3：异常检测

```python
# 实现异常检测系统：
# 1. 使用 Autoencoder
# 2. 使用 Isolation Forest
# 3. 使用 One-Class SVM

# 数据集：信用卡欺诈检测
# 评估：ROC-AUC, Precision-Recall
```

### 作业 4：自监督学习

```python
# 实现一个自监督学习pipeline：
# 1. 选择预训练任务（如 SimCLR）
# 2. 在无标签数据上预训练
# 3. 在小量标注数据上微调
# 4. 对比：从零训练 vs 自监督预训练
```

---

## 🔑 本章关键概念

| 概念 | 说明 |
|------|------|
| 无监督学习 | 从无标签数据学习结构 |
| 聚类 | 将相似样本分组 |
| K-Means | 基于中心的聚类 |
| DBSCAN | 基于密度的聚类 |
| 层次聚类 | 构建聚类树 |
| PCA | 线性降维 |
| t-SNE | 非线性降维（可视化） |
| UMAP | 快速非线性降维 |
| 自监督学习 | 自动生成监督信号 |
| 对比学习 | 学习相似和不相似 |
| 自编码器 | 重构学习表示 |

---

