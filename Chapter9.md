# 第九章：迁移学习与微调 (Transfer Learning & Fine-tuning)

## 📌 章节目标
- 理解迁移学习的动机和原理
- 掌握预训练模型的使用方法
- 学习不同的微调策略
- 了解领域自适应技术
- 实战：图像分类、文本分类的迁移学习

---

## 9.1 为什么需要迁移学习？

### 🚨 从零训练的问题

**传统深度学习**：

```
收集大量数据 → 设计网络 → 从零训练 → 部署

问题：
  ❌ 需要海量标注数据
  ❌ 训练时间长（几天到几周）
  ❌ 计算资源昂贵
  ❌ 容易过拟合（小数据集）
```

**例子**：训练 ResNet-50 on ImageNet

```
数据：120万张标注图片
时间：8个 GPU，几天到一周
成本：数千美元
```

### ✅ 迁移学习的优势

**核心思想**：利用已有知识加速新任务学习

```
预训练（大数据集）→ 微调（小数据集）→ 部署

优势：
  ✓ 需要更少的数据
  ✓ 训练更快（小时级别）
  ✓ 性能更好（特别是小数据集）
  ✓ 降低成本
```

### 🧠 直觉理解

**人类学习的类比**：

```
学习识别猫：
  不需要从零学习"什么是边缘"、"什么是纹理"
  已经有视觉系统的基础知识
  只需要学习"猫的特征"

迁移学习：
  预训练模型 = 已有的视觉/语言知识
  微调 = 针对特定任务调整
```

---

## 9.2 迁移学习的分类

### 📊 按任务关系分类

#### **1. 归纳迁移 (Inductive Transfer)**

```
源任务 ≠ 目标任务，但相关

例：
  源：ImageNet 分类 (1000类)
  目标：医学图像分类 (5类)
```

#### **2. 转导迁移 (Transductive Transfer)**

```
源任务 = 目标任务，但数据分布不同

例：
  源：电影评论情感分析
  目标：产品评论情感分析
```

#### **3. 无监督迁移 (Unsupervised Transfer)**

```
源任务和目标任务都无标签

例：
  聚类、降维任务
```

---

### 📊 按迁移内容分类

#### **1. 特征迁移 (Feature Transfer)**

```
迁移学到的特征表示

方法：固定预训练模型，只训练分类器
```

#### **2. 参数迁移 (Parameter Transfer)**

```
迁移模型参数作为初始化

方法：用预训练权重初始化，然后微调
```

#### **3. 关系迁移 (Relation Transfer)**

```
迁移样本间的关系

例：知识图谱、结构化预测
```

---

## 9.3 计算机视觉中的迁移学习

### 🖼️ 预训练模型

**常用预训练模型**（在 ImageNet 上训练）：

```
轻量级：
  - MobileNet (4M 参数)
  - EfficientNet-B0 (5M)

中等：
  - ResNet-50 (25M)
  - VGG-16 (138M)

大型：
  - ResNet-152 (60M)
  - EfficientNet-B7 (66M)
  - Vision Transformer (ViT) (86M)
```

### 📐 特征提取 vs 微调

#### **特征提取 (Feature Extraction)**

```
冻结预训练模型 → 只训练新的分类器

适用场景：
  ✓ 数据集很小 (< 1000 样本)
  ✓ 目标任务与源任务相似
  ✓ 计算资源有限
```

**实现**：

```python
import torch
import torch.nn as nn
from torchvision import models

# 加载预训练模型
model = models.resnet50(pretrained=True)

# 冻结所有参数
for param in model.parameters():
    param.requires_grad = False

# 替换最后的全连接层
num_features = model.fc.in_features
model.fc = nn.Linear(num_features, num_classes)  # 只有这层会训练

# 只优化新添加的层
optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
```

---

#### **微调 (Fine-tuning)**

```
解冻部分或全部层 → 在新数据上训练

适用场景：
  ✓ 数据集中等大小 (1k - 100k)
  ✓ 目标任务与源任务有差异
  ✓ 追求更好性能
```

**策略**：

```
1. 全局微调：
   解冻所有层，用小学习率训练

2. 逐层微调：
   先训练顶层，逐渐解冻底层

3. 判别式微调：
   不同层用不同学习率
```

**实现**：

```python
# 方法1：全局微调（小学习率）
model = models.resnet50(pretrained=True)

# 替换分类器
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 所有参数都训练，但用小学习率
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

```python
# 方法2：逐层微调
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 第一阶段：只训练分类器
for param in model.parameters():
    param.requires_grad = False
for param in model.fc.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam(model.fc.parameters(), lr=0.001)
# 训练几个 epoch...

# 第二阶段：解冻 layer4
for param in model.layer4.parameters():
    param.requires_grad = True

optimizer = torch.optim.Adam([
    {'params': model.fc.parameters(), 'lr': 0.001},
    {'params': model.layer4.parameters(), 'lr': 0.0001}
])
# 继续训练...
```

```python
# 方法3：判别式学习率
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, num_classes)

# 不同层组使用不同学习率
optimizer = torch.optim.Adam([
    {'params': model.conv1.parameters(), 'lr': 1e-5},
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-4},
    {'params': model.layer4.parameters(), 'lr': 1e-3},
    {'params': model.fc.parameters(), 'lr': 1e-2}
])
```

---

### 🎯 实战：猫狗分类（迁移学习）

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm
import os

# ==================== 超参数 ====================
DATA_DIR = './data/cats_and_dogs'
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 2
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ==================== 数据增强 ====================
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

# ==================== 数据加载 ====================
image_datasets = {
    x: datasets.ImageFolder(os.path.join(DATA_DIR, x), data_transforms[x])
    for x in ['train', 'val']
}

dataloaders = {
    x: DataLoader(image_datasets[x], batch_size=BATCH_SIZE,
                  shuffle=(x=='train'), num_workers=4)
    for x in ['train', 'val']
}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

print(f"训练集: {dataset_sizes['train']} 张")
print(f"验证集: {dataset_sizes['val']} 张")
print(f"类别: {class_names}")

# ==================== 模型定义 ====================

def create_model(model_name='resnet50', num_classes=2, feature_extract=False):
    """
    创建迁移学习模型

    参数:
        model_name: 预训练模型名称
        num_classes: 输出类别数
        feature_extract: True=特征提取, False=微调
    """
    model = None

    if model_name == 'resnet50':
        model = models.resnet50(pretrained=True)

        # 特征提取模式：冻结参数
        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

        # 替换分类器
        num_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=True)

        if feature_extract:
            for param in model.parameters():
                param.requires_grad = False

        num_features = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    elif model_name == 'vgg16':
        model = models.vgg16(pretrained=True)

        if feature_extract:
            for param in model.features.parameters():
                param.requires_grad = False

        num_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(num_features, num_classes)

    return model

# ==================== 训练函数 ====================

def train_model(model, criterion, optimizer, scheduler, num_epochs):
    """训练模型"""
    best_acc = 0.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        print('-' * 60)

        # 训练和验证
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            # 进度条
            pbar = tqdm(dataloaders[phase], desc=f'{phase.capitalize()}')

            for inputs, labels in pbar:
                inputs = inputs.to(DEVICE)
                labels = labels.to(DEVICE)

                optimizer.zero_grad()

                # 前向传播
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # 反向传播
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # 统计
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                # 更新进度条
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{torch.sum(preds == labels.data).item() / len(labels):.4f}'
                })

            # Epoch 统计
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f'{phase.capitalize()} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            # 记录历史
            history[f'{phase}_loss'].append(epoch_loss)
            history[f'{phase}_acc'].append(epoch_acc.item())

            # 保存最佳模型
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), 'best_model.pth')
                print(f'✓ 保存最佳模型 (Acc: {best_acc:.4f})')

        # 学习率调度
        scheduler.step()

    print(f'\n最佳验证准确率: {best_acc:.4f}')

    # 加载最佳模型
    model.load_state_dict(torch.load('best_model.pth'))

    return model, history

# ==================== 对比实验 ====================

def compare_strategies():
    """对比不同迁移学习策略"""
    strategies = {
        'Feature Extraction': {
            'model': create_model('resnet50', NUM_CLASSES, feature_extract=True),
            'lr': 0.001,
            'color': 'blue'
        },
        'Fine-tuning (全局)': {
            'model': create_model('resnet50', NUM_CLASSES, feature_extract=False),
            'lr': 0.0001,
            'color': 'red'
        }
    }

    results = {}

    for strategy_name, config in strategies.items():
        print(f'\n{"="*70}')
        print(f'策略: {strategy_name}')
        print(f'{"="*70}')

        model = config['model'].to(DEVICE)

        # 损失函数和优化器
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config['lr']
        )
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

        # 训练
        model, history = train_model(model, criterion, optimizer, scheduler, EPOCHS)

        results[strategy_name] = {
            'model': model,
            'history': history,
            'color': config['color']
        }

    # ==================== 可视化对比 ====================

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Loss 对比
    for strategy_name, data in results.items():
        epochs_range = range(1, len(data['history']['train_loss']) + 1)
        axes[0].plot(epochs_range, data['history']['train_loss'],
                    label=f'{strategy_name} (Train)',
                    linestyle='--', color=data['color'])
        axes[0].plot(epochs_range, data['history']['val_loss'],
                    label=f'{strategy_name} (Val)',
                    linestyle='-', color=data['color'])

    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Accuracy 对比
    for strategy_name, data in results.items():
        epochs_range = range(1, len(data['history']['train_acc']) + 1)
        axes[1].plot(epochs_range, data['history']['train_acc'],
                    label=f'{strategy_name} (Train)',
                    linestyle='--', color=data['color'])
        axes[1].plot(epochs_range, data['history']['val_acc'],
                    label=f'{strategy_name} (Val)',
                    linestyle='-', color=data['color'])

    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('transfer_learning_comparison.png', dpi=300)
    plt.show()

    return results

# ==================== 可视化预测 ====================

def visualize_predictions(model, num_images=16):
    """可视化模型预测"""
    model.eval()

    images_so_far = 0
    fig = plt.figure(figsize=(16, 12))

    with torch.no_grad():
        for inputs, labels in dataloaders['val']:
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(4, 4, images_so_far)
                ax.axis('off')

                # 反归一化显示
                img = inputs.cpu().data[j]
                img = img.numpy().transpose((1, 2, 0))
                mean = np.array([0.485, 0.456, 0.406])
                std = np.array([0.229, 0.224, 0.225])
                img = std * img + mean
                img = np.clip(img, 0, 1)

                ax.imshow(img)

                # 标题：预测 vs 真实
                color = 'green' if preds[j] == labels[j] else 'red'
                ax.set_title(f'Pred: {class_names[preds[j]]}\nTrue: {class_names[labels[j]]}',
                           color=color, fontsize=10)

                if images_so_far == num_images:
                    plt.tight_layout()
                    plt.savefig('predictions.png', dpi=300)
                    plt.show()
                    return

# ==================== 主程序 ====================

if __name__ == '__main__':
    # 对比不同策略
    results = compare_strategies()

    # 可视化最佳模型的预测
    best_strategy = max(results.items(),
                       key=lambda x: max(x[1]['history']['val_acc']))
    print(f'\n最佳策略: {best_strategy[0]}')

    visualize_predictions(best_strategy[1]['model'])
```

---

## 9.4 自然语言处理中的迁移学习

### 📝 预训练语言模型

**发展历程**：

```
2013: Word2Vec, GloVe
      ↓
2018: ELMo (动态词向量)
      ↓
2018: BERT (双向预训练)
      ↓
2019: GPT-2 (大规模生成)
      ↓
2020: GPT-3 (超大规模)
      ↓
2023: ChatGPT, GPT-4
```

### 🔹 使用 Hugging Face Transformers

```python
from transformers import (
    BertTokenizer, BertForSequenceClassification,
    GPT2Tokenizer, GPT2LMHeadModel,
    AutoTokenizer, AutoModelForSequenceClassification,
    Trainer, TrainingArguments
)
import torch
from torch.utils.data import Dataset
import numpy as np

# ==================== 数据集定义 ====================

class TextClassificationDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# ==================== BERT 微调 ====================

def fine_tune_bert(train_texts, train_labels, val_texts, val_labels,
                   num_labels=2, epochs=3):
    """
    BERT 微调用于文本分类
    """
    # 加载预训练模型和分词器
    model_name = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels
    )

    # 创建数据集
    train_dataset = TextClassificationDataset(
        train_texts, train_labels, tokenizer
    )
    val_dataset = TextClassificationDataset(
        val_texts, val_labels, tokenizer
    )

    # 训练参数
    training_args = TrainingArguments(
        output_dir='./results',
        num_train_epochs=epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        warmup_steps=500,
        weight_decay=0.01,
        logging_dir='./logs',
        logging_steps=100,
        evaluation_strategy='epoch',
        save_strategy='epoch',
        load_best_model_at_end=True,
        learning_rate=2e-5,
    )

    # 定义评估指标
    def compute_metrics(eval_pred):
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        accuracy = (predictions == labels).mean()
        return {'accuracy': accuracy}

    # 训练器
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_metrics
    )

    # 训练
    trainer.train()

    # 评估
    eval_results = trainer.evaluate()
    print(f"\n评估结果: {eval_results}")

    return model, tokenizer

# ==================== 实战示例：情感分析 ====================

def sentiment_analysis_example():
    """情感分析完整示例"""

    # 示例数据（实际应使用 IMDB 等数据集）
    train_texts = [
        "This movie is fantastic! I loved it.",
        "Great film, highly recommended.",
        "Amazing performance by the actors.",
        "Terrible waste of time.",
        "Boring and predictable plot.",
        "I hated every minute of it."
    ] * 100

    train_labels = [1, 1, 1, 0, 0, 0] * 100  # 1=正面, 0=负面

    val_texts = [
        "Excellent movie!",
        "Not worth watching.",
        "Pretty good film.",
        "Absolutely awful."
    ]
    val_labels = [1, 0, 1, 0]

    # 微调 BERT
    model, tokenizer = fine_tune_bert(
        train_texts, train_labels,
        val_texts, val_labels,
        num_labels=2,
        epochs=3
    )

    # 预测函数
    def predict(text):
        encoding = tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        model.eval()
        with torch.no_grad():
            outputs = model(**encoding)
            predictions = torch.softmax(outputs.logits, dim=1)
            label = torch.argmax(predictions, dim=1).item()
            confidence = predictions[0][label].item()

        sentiment = "正面" if label == 1 else "负面"
        return sentiment, confidence

    # 测试
    test_texts = [
        "This is an amazing movie!",
        "Worst film I've ever seen.",
        "It was okay, nothing special."
    ]

    print("\n预测结果:")
    for text in test_texts:
        sentiment, confidence = predict(text)
        print(f"\n文本: {text}")
        print(f"情感: {sentiment} (置信度: {confidence:.4f})")

    return model, tokenizer

# 运行示例
if __name__ == '__main__':
    model, tokenizer = sentiment_analysis_example()
```

---

## 9.5 领域自适应 (Domain Adaptation)

### 🎯 问题设定

```
源领域：有大量标注数据
目标领域：数据分布不同，标注少或无

例子：
  源：新闻文本分类
  目标：社交媒体文本分类
```

### 📐 方法分类

#### **1. 基于实例的方法**

**重要性加权**：

```python
class ImportanceWeightedLoss(nn.Module):
    """根据样本相似度加权损失"""

    def __init__(self, base_criterion):
        super().__init__()
        self.base_criterion = base_criterion

    def compute_weights(self, source_features, target_features):
        """
        计算源域样本的重要性权重
        使得源域分布接近目标域
        """
        # 简化版：基于特征距离
        distances = torch.cdist(source_features, target_features)
        min_distances = distances.min(dim=1)[0]
        weights = torch.exp(-min_distances)
        weights = weights / weights.sum()
        return weights

    def forward(self, outputs, targets, weights):
        loss = self.base_criterion(outputs, targets)
        weighted_loss = (loss * weights).mean()
        return weighted_loss
```

---

#### **2. 基于特征的方法**

**领域对抗训练 (Domain Adversarial Training)**：

```python
class GradientReversalLayer(torch.autograd.Function):
    """梯度反转层"""

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

class DomainAdversarialNetwork(nn.Module):
    """领域对抗网络"""

    def __init__(self, input_size, hidden_size, num_classes):
        super().__init__()

        # 特征提取器（共享）
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # 标签预测器
        self.label_predictor = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(hidden_size, num_classes)
        )

        # 域分类器
        self.domain_classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_size, 2)  # 2个域
        )

    def forward(self, x, alpha=1.0):
        # 特征提取
        features = self.feature_extractor(x)

        # 标签预测
        class_output = self.label_predictor(features)

        # 域分类（梯度反转）
        reverse_features = GradientReversalLayer.apply(features, alpha)
        domain_output = self.domain_classifier(reverse_features)

        return class_output, domain_output

# 训练
def train_domain_adaptation(model, source_loader, target_loader,
                            num_epochs=50):
    """领域自适应训练"""

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    class_criterion = nn.CrossEntropyLoss()
    domain_criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        model.train()

        for (source_data, source_labels), (target_data, _) in zip(source_loader, target_loader):

            # Alpha 随着训练增加（从 0 到 1）
            p = float(epoch) / num_epochs
            alpha = 2. / (1. + np.exp(-10 * p)) - 1

            # 源域数据
            class_output_s, domain_output_s = model(source_data, alpha)

            # 目标域数据
            _, domain_output_t = model(target_data, alpha)

            # 域标签：源=0，目标=1
            domain_label_s = torch.zeros(len(source_data)).long()
            domain_label_t = torch.ones(len(target_data)).long()

            # 计算损失
            class_loss = class_criterion(class_output_s, source_labels)
            domain_loss_s = domain_criterion(domain_output_s, domain_label_s)
            domain_loss_t = domain_criterion(domain_output_t, domain_label_t)
            domain_loss = domain_loss_s + domain_loss_t

            total_loss = class_loss + domain_loss

            # 反向传播
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            if epoch % 10 == 0:
                print(f'Epoch {epoch}: Class Loss={class_loss.item():.4f}, '
                      f'Domain Loss={domain_loss.item():.4f}')

    return model
```

---

#### **3. 自训练 (Self-Training)**

```python
class SelfTraining:
    """自训练/伪标签方法"""

    def __init__(self, model, confidence_threshold=0.9):
        self.model = model
        self.threshold = confidence_threshold

    def generate_pseudo_labels(self, unlabeled_loader):
        """生成伪标签"""
        self.model.eval()

        pseudo_data = []
        pseudo_labels = []

        with torch.no_grad():
            for data in unlabeled_loader:
                outputs = self.model(data)
                probs = torch.softmax(outputs, dim=1)

                # 只选择高置信度样本
                max_probs, predictions = torch.max(probs, dim=1)

                mask = max_probs > self.threshold

                pseudo_data.append(data[mask])
                pseudo_labels.append(predictions[mask])

        return torch.cat(pseudo_data), torch.cat(pseudo_labels)

    def train(self, labeled_loader, unlabeled_loader, num_iterations=5):
        """迭代训练"""

        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        for iteration in range(num_iterations):
            print(f'\n=== Iteration {iteration+1}/{num_iterations} ===')

            # 在标注数据上训练
            self.model.train()
            for data, labels in labeled_loader:
                outputs = self.model(data)
                loss = criterion(outputs, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # 生成伪标签
            pseudo_data, pseudo_labels = self.generate_pseudo_labels(unlabeled_loader)

            if len(pseudo_data) > 0:
                print(f'生成了 {len(pseudo_data)} 个伪标签')

                # 在伪标签数据上训练
                pseudo_dataset = torch.utils.data.TensorDataset(
                    pseudo_data, pseudo_labels
                )
                pseudo_loader = torch.utils.data.DataLoader(
                    pseudo_dataset, batch_size=32, shuffle=True
                )

                self.model.train()
                for data, labels in pseudo_loader:
                    outputs = self.model(data)
                    loss = criterion(outputs, labels)

                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

        return self.model
```

---

## 9.6 少样本学习 (Few-Shot Learning)

### 🎯 问题定义

```
N-way K-shot 分类：
  N: 类别数
  K: 每类的样本数

例：5-way 1-shot
  5个类别，每类只有1个样本
```

### 📐 元学习 (Meta-Learning)

**MAML (Model-Agnostic Meta-Learning)**：

```python
class MAML:
    """模型无关的元学习"""

    def __init__(self, model, inner_lr=0.01, outer_lr=0.001):
        self.model = model
        self.inner_lr = inner_lr  # 任务内学习率
        self.outer_lr = outer_lr  # 跨任务学习率
        self.meta_optimizer = torch.optim.Adam(
            model.parameters(), lr=outer_lr
        )

    def inner_loop(self, support_x, support_y, num_steps=5):
        """
        任务内适应（快速学习）

        参数：
            support_x: 支持集输入
            support_y: 支持集标签
            num_steps: 内循环步数
        """
        # 复制模型参数
        fast_weights = [p.clone() for p in self.model.parameters()]

        for step in range(num_steps):
            # 前向传播
            outputs = self.model(support_x)
            loss = nn.CrossEntropyLoss()(outputs, support_y)

            # 计算梯度
            grads = torch.autograd.grad(
                loss, fast_weights, create_graph=True
            )

            # 更新参数（一步梯度下降）
            fast_weights = [
                w - self.inner_lr * g
                for w, g in zip(fast_weights, grads)
            ]

        return fast_weights

    def outer_loop(self, tasks, num_epochs=1000):
        """
        跨任务学习（元学习）

        参数：
            tasks: 任务列表，每个任务包含 (support_set, query_set)
        """
        for epoch in range(num_epochs):
            meta_loss = 0

            for task in tasks:
                support_x, support_y = task['support']
                query_x, query_y = task['query']

                # 内循环：在支持集上快速适应
                fast_weights = self.inner_loop(support_x, support_y)

                # 在查询集上评估
                # 使用更新后的参数
                outputs = self.model(query_x, weights=fast_weights)
                loss = nn.CrossEntropyLoss()(outputs, query_y)

                meta_loss += loss

            # 外循环：更新元参数
            meta_loss = meta_loss / len(tasks)

            self.meta_optimizer.zero_grad()
            meta_loss.backward()
            self.meta_optimizer.step()

            if epoch % 100 == 0:
                print(f'Epoch {epoch}: Meta Loss = {meta_loss.item():.4f}')
```

---

### 📐 原型网络 (Prototypical Networks)

```python
class PrototypicalNetwork(nn.Module):
    """原型网络"""

    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def compute_prototypes(self, support_x, support_y, n_way):
        """
        计算每个类的原型（均值）

        参数：
            support_x: (n_way * n_support, *)
            support_y: (n_way * n_support,)
            n_way: 类别数
        """
        # 编码
        embeddings = self.encoder(support_x)  # (n_way*n_support, d)

        # 计算原型
        prototypes = []
        for c in range(n_way):
            mask = (support_y == c)
            class_embeddings = embeddings[mask]
            prototype = class_embeddings.mean(dim=0)
            prototypes.append(prototype)

        prototypes = torch.stack(prototypes)  # (n_way, d)
        return prototypes

    def forward(self, support_x, support_y, query_x, n_way):
        """
        参数：
            support_x: 支持集输入
            support_y: 支持集标签
            query_x: 查询集输入
            n_way: 类别数
        """
        # 计算原型
        prototypes = self.compute_prototypes(support_x, support_y, n_way)

        # 编码查询集
        query_embeddings = self.encoder(query_x)  # (n_query, d)

        # 计算距离（欧氏距离）
        distances = torch.cdist(query_embeddings, prototypes)  # (n_query, n_way)

        # 转换为概率（负距离的 softmax）
        logits = -distances

        return logits

# 训练
def train_prototypical_network(model, tasks, num_epochs=1000):
    """训练原型网络"""

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        total_loss = 0
        total_acc = 0

        for task in tasks:
            support_x, support_y = task['support']
            query_x, query_y = task['query']
            n_way = len(torch.unique(support_y))

            # 前向传播
            logits = model(support_x, support_y, query_x, n_way)

            # 计算损失
            loss = criterion(logits, query_y)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            total_acc += (preds == query_y).float().mean().item()

        if epoch % 100 == 0:
            avg_loss = total_loss / len(tasks)
            avg_acc = total_acc / len(tasks)
            print(f'Epoch {epoch}: Loss={avg_loss:.4f}, Acc={avg_acc:.4f}')

    return model
```

---

## 9.7 知识蒸馏 (Knowledge Distillation)

### 🎯 核心思想

```
教师模型（大）→ 知识 → 学生模型（小）

目标：
  用大模型的"软标签"训练小模型
  小模型获得大模型的泛化能力
```

### 📐 温度 Softmax

```
标准 Softmax：
  p_i = exp(z_i) / Σ_j exp(z_j)

温度 Softmax：
  p_i = exp(z_i/T) / Σ_j exp(z_j/T)

T > 1: 输出更平滑（"软"标签）
T = 1: 标准 softmax
T → ∞: 均匀分布
```

### 💻 实现

```python
class DistillationLoss(nn.Module):
    """知识蒸馏损失"""

    def __init__(self, temperature=3.0, alpha=0.5):
        super().__init__()
        self.temperature = temperature
        self.alpha = alpha
        self.criterion = nn.CrossEntropyLoss()
        self.kl_div = nn.KLDivLoss(reduction='batchmean')

    def forward(self, student_logits, teacher_logits, targets):
        """
        参数：
            student_logits: 学生模型输出
            teacher_logits: 教师模型输出
            targets: 真实标签
        """
        # 硬标签损失
        hard_loss = self.criterion(student_logits, targets)

        # 软标签损失（KL 散度）
        student_soft = F.log_softmax(student_logits / self.temperature, dim=1)
        teacher_soft = F.softmax(teacher_logits / self.temperature, dim=1)
        soft_loss = self.kl_div(student_soft, teacher_soft) * (self.temperature ** 2)

        # 组合损失
        total_loss = self.alpha * hard_loss + (1 - self.alpha) * soft_loss

        return total_loss

def knowledge_distillation(teacher_model, student_model, train_loader,
                           num_epochs=50, temperature=3.0):
    """
    知识蒸馏训练

    参数：
        teacher_model: 预训练的大模型
        student_model: 待训练的小模型
        train_loader: 数据加载器
        num_epochs: 训练轮数
        temperature: 温度参数
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    teacher_model.to(device)
    student_model.to(device)

    teacher_model.eval()  # 教师模型不训练

    optimizer = torch.optim.Adam(student_model.parameters(), lr=0.001)
    distillation_loss = DistillationLoss(temperature=temperature, alpha=0.7)

    for epoch in range(num_epochs):
        student_model.train()

        total_loss = 0
        correct = 0
        total = 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            # 教师模型输出（不需要梯度）
            with torch.no_grad():
                teacher_logits = teacher_model(inputs)

            # 学生模型输出
            student_logits = student_model(inputs)

            # 计算蒸馏损失
            loss = distillation_loss(student_logits, teacher_logits, targets)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # 统计
            total_loss += loss.item()
            _, predicted = student_logits.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        accuracy = 100. * correct / total
        print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, '
              f'Acc={accuracy:.2f}%')

    return student_model

# 使用示例
if __name__ == '__main__':
    # 教师模型（大）
    teacher = models.resnet50(pretrained=True)
    teacher.fc = nn.Linear(teacher.fc.in_features, 10)

    # 学生模型（小）
    student = models.resnet18(pretrained=False)
    student.fc = nn.Linear(student.fc.in_features, 10)

    # 知识蒸馏
    student = knowledge_distillation(
        teacher, student, train_loader,
        num_epochs=50, temperature=3.0
    )
```

---

## 9.8 实战：完整迁移学习项目

### 📋 任务：医学图像分类

```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models
from PIL import Image
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns

# ==================== 自定义数据集 ====================

class MedicalImageDataset(Dataset):
    """医学图像数据集"""

    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # 收集所有图像路径
        self.images = []
        self.labels = []

        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                self.images.append(img_path)
                self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

# ==================== 数据增强策略 ====================

def get_transforms(phase='train'):
    """获取数据变换"""

    if phase == 'train':
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                 saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])

# ==================== 模型构建 ====================

class TransferLearningModel(nn.Module):
    """迁移学习模型"""

    def __init__(self, model_name='resnet50', num_classes=5,
                 pretrained=True, freeze_backbone=False):
        super().__init__()

        # 加载预训练模型
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = self.backbone.fc.in_features

            # 冻结骨干网络
            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            # 替换分类器
            self.backbone.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(num_features, 512),
                nn.ReLU(),
                nn.BatchNorm1d(512),
                nn.Dropout(0.3),
                nn.Linear(512, num_classes)
            )

        elif model_name == 'efficientnet_b3':
            self.backbone = models.efficientnet_b3(pretrained=pretrained)

            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            num_features = self.backbone.classifier[1].in_features
            self.backbone.classifier = nn.Sequential(
                nn.Dropout(0.3),
                nn.Linear(num_features, num_classes)
            )

        elif model_name == 'vit_b_16':
            self.backbone = models.vit_b_16(pretrained=pretrained)

            if freeze_backbone:
                for param in self.backbone.parameters():
                    param.requires_grad = False

            num_features = self.backbone.heads.head.in_features
            self.backbone.heads.head = nn.Linear(num_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# ==================== 训练器类 ====================

class Trainer:
    """训练器"""

    def __init__(self, model, train_loader, val_loader,
                 criterion, optimizer, scheduler, device,
                 num_epochs=50, early_stopping_patience=10):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience

        self.best_val_acc = 0.0
        self.patience_counter = 0
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': []
        }

    def train_epoch(self):
        """训练一个 epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in self.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            self.optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc

    def validate(self):
        """验证"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for inputs, labels in self.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                running_loss += loss.item() * inputs.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        epoch_loss = running_loss / total
        epoch_acc = 100. * correct / total

        return epoch_loss, epoch_acc, all_preds, all_labels

    def train(self):
        """完整训练流程"""
        print(f"开始训练，设备: {self.device}")
        print("="*70)

        for epoch in range(self.num_epochs):
            print(f'\nEpoch {epoch+1}/{self.num_epochs}')
            print('-' * 70)

            # 训练
            train_loss, train_acc = self.train_epoch()

            # 验证
            val_loss, val_acc, _, _ = self.validate()

            # 记录历史
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)

            # 打印结果
            print(f'训练 - Loss: {train_loss:.4f}, Acc: {train_acc:.2f}%')
            print(f'验证 - Loss: {val_loss:.4f}, Acc: {val_acc:.2f}%')

            # 学习率调度
            if self.scheduler:
                self.scheduler.step(val_loss)
                print(f'学习率: {self.optimizer.param_groups[0]["lr"]:.6f}')

            # 保存最佳模型
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                print(f'✓ 保存最佳模型 (Val Acc: {val_acc:.2f}%)')
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early Stopping
            if self.patience_counter >= self.early_stopping_patience:
                print(f'\nEarly stopping triggered at epoch {epoch+1}')
                break

        print(f'\n训练完成！最佳验证准确率: {self.best_val_acc:.2f}%')

        # 加载最佳模型
        self.model.load_state_dict(torch.load('best_model.pth'))

        return self.history

# ==================== 评估和可视化 ====================

class Evaluator:
    """评估器"""

    def __init__(self, model, test_loader, class_names, device):
        self.model = model
        self.test_loader = test_loader
        self.class_names = class_names
        self.device = device

    def evaluate(self):
        """完整评估"""
        self.model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for inputs, labels in self.test_loader:
                inputs = inputs.to(self.device)

                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                _, predicted = outputs.max(1)

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # 准确率
        accuracy = (all_preds == all_labels).mean()
        print(f'\n测试集准确率: {accuracy*100:.2f}%\n')

        # 分类报告
        print("分类报告:")
        print(classification_report(all_labels, all_preds,
                                   target_names=self.class_names))

        # 混淆矩阵
        self.plot_confusion_matrix(all_labels, all_preds)

        # ROC 曲线（多分类）
        if len(self.class_names) <= 10:
            self.plot_roc_curves(all_labels, all_probs)

        return all_preds, all_labels, all_probs

    def plot_confusion_matrix(self, labels, preds):
        """绘制混淆矩阵"""
        cm = confusion_matrix(labels, preds)

        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=self.class_names,
                   yticklabels=self.class_names)
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300)
        plt.show()

    def plot_roc_curves(self, labels, probs):
        """绘制 ROC 曲线"""
        from sklearn.metrics import roc_curve, auc
        from sklearn.preprocessing import label_binarize

        # 二值化标签
        labels_bin = label_binarize(labels, classes=range(len(self.class_names)))

        plt.figure(figsize=(10, 8))

        for i, class_name in enumerate(self.class_names):
            fpr, tpr, _ = roc_curve(labels_bin[:, i], probs[:, i])
            roc_auc = auc(fpr, tpr)

            plt.plot(fpr, tpr, label=f'{class_name} (AUC = {roc_auc:.2f})')

        plt.plot([0, 1], [0, 1], 'k--', label='随机猜测')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC 曲线')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('roc_curves.png', dpi=300)
        plt.show()

# ==================== 主程序 ====================

def main():
    """主函数"""

    # 超参数
    DATA_DIR = './data/medical_images'
    BATCH_SIZE = 32
    NUM_EPOCHS = 50
    LEARNING_RATE = 0.001
    NUM_CLASSES = 5
    MODEL_NAME = 'resnet50'  # 'resnet50', 'efficientnet_b3', 'vit_b_16'

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 数据加载
    train_dataset = MedicalImageDataset(
        os.path.join(DATA_DIR, 'train'),
        transform=get_transforms('train')
    )
    val_dataset = MedicalImageDataset(
        os.path.join(DATA_DIR, 'val'),
        transform=get_transforms('val')
    )
    test_dataset = MedicalImageDataset(
        os.path.join(DATA_DIR, 'test'),
        transform=get_transforms('test')
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                             shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE,
                           shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                            shuffle=False, num_workers=4)

    print(f"训练集: {len(train_dataset)} 张")
    print(f"验证集: {len(val_dataset)} 张")
    print(f"测试集: {len(test_dataset)} 张")
    print(f"类别: {train_dataset.classes}")

    # 创建模型
    model = TransferLearningModel(
        model_name=MODEL_NAME,
        num_classes=NUM_CLASSES,
        pretrained=True,
        freeze_backbone=False  # 先全局微调
    ).to(device)

    # 损失和优化器
    criterion = nn.CrossEntropyLoss()

    # 分层学习率
    backbone_params = []
    classifier_params = []

    for name, param in model.named_parameters():
        if 'fc' in name or 'classifier' in name or 'head' in name:
            classifier_params.append(param)
        else:
            backbone_params.append(param)

    optimizer = optim.Adam([
        {'params': backbone_params, 'lr': LEARNING_RATE * 0.1},
        {'params': classifier_params, 'lr': LEARNING_RATE}
    ], weight_decay=1e-4)

    # 学习率调度
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5, verbose=True
    )

    # 训练
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        num_epochs=NUM_EPOCHS,
        early_stopping_patience=10
    )

    history = trainer.train()

    # 可视化训练过程
    plot_training_history(history)

    # 评估
    evaluator = Evaluator(
        model=model,
        test_loader=test_loader,
        class_names=train_dataset.classes,
        device=device
    )

    preds, labels, probs = evaluator.evaluate()

    # Grad-CAM 可视化
    visualize_gradcam(model, test_loader, device, num_images=8)

    return model, history

# ==================== 可视化函数 ====================

def plot_training_history(history):
    """绘制训练历史"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    epochs = range(1, len(history['train_loss']) + 1)

    # Loss
    axes[0].plot(epochs, history['train_loss'], 'b-', label='训练 Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], 'r-', label='验证 Loss', linewidth=2)
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('训练和验证 Loss', fontsize=14)
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Accuracy
    axes[1].plot(epochs, history['train_acc'], 'b-', label='训练准确率', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], 'r-', label='验证准确率', linewidth=2)
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('准确率 (%)', fontsize=12)
    axes[1].set_title('训练和验证准确率', fontsize=14)
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=300)
    plt.show()

# ==================== Grad-CAM 可视化 ====================

class GradCAM:
    """Grad-CAM 类激活映射"""

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        # 注册钩子
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate_cam(self, input_image, target_class=None):
        """生成 CAM"""
        # 前向传播
        output = self.model(input_image)

        if target_class is None:
            target_class = output.argmax(dim=1)

        # 反向传播
        self.model.zero_grad()
        class_score = output[:, target_class]
        class_score.backward()

        # 计算权重
        pooled_gradients = torch.mean(self.gradients, dim=[2, 3])

        # 加权求和
        for i in range(self.activations.size(1)):
            self.activations[:, i, :, :] *= pooled_gradients[:, i].view(-1, 1, 1)

        # CAM
        cam = torch.mean(self.activations, dim=1).squeeze()
        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / cam.max()

        return cam.cpu().numpy()

def visualize_gradcam(model, test_loader, device, num_images=8):
    """可视化 Grad-CAM"""

    # 获取目标层
    if hasattr(model.backbone, 'layer4'):
        target_layer = model.backbone.layer4[-1]
    elif hasattr(model.backbone, 'features'):
        target_layer = model.backbone.features[-1]
    else:
        print("无法找到目标层，跳过 Grad-CAM")
        return

    gradcam = GradCAM(model, target_layer)

    model.eval()

    fig, axes = plt.subplots(4, 4, figsize=(16, 16))
    axes = axes.flatten()

    images_shown = 0

    for images, labels in test_loader:
        if images_shown >= num_images:
            break

        for i in range(min(len(images), num_images - images_shown)):
            image = images[i:i+1].to(device)
            label = labels[i].item()

            # 生成 CAM
            cam = gradcam.generate_cam(image)

            # 反归一化图像
            img = images[i].cpu().numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            # 调整 CAM 大小
            import cv2
            cam_resized = cv2.resize(cam, (224, 224))

            # 叠加显示
            ax = axes[images_shown * 2]
            ax.imshow(img)
            ax.set_title(f'原图 (标签: {label})')
            ax.axis('off')

            ax = axes[images_shown * 2 + 1]
            ax.imshow(img)
            ax.imshow(cam_resized, cmap='jet', alpha=0.5)
            ax.set_title('Grad-CAM')
            ax.axis('off')

            images_shown += 1

            if images_shown >= num_images:
                break

    plt.tight_layout()
    plt.savefig('gradcam_visualization.png', dpi=300)
    plt.show()

# ==================== 运行 ====================

if __name__ == '__main__':
    model, history = main()
```

---

## 9.9 迁移学习最佳实践

### ✅ 数据集大小策略

```python
def choose_strategy(dataset_size, similarity_to_pretrain):
    """
    根据数据集大小和相似度选择策略

    参数：
        dataset_size: 数据集大小
        similarity_to_pretrain: 与预训练数据的相似度
    """

    if dataset_size < 1000:
        if similarity_to_pretrain == 'high':
            return "特征提取（冻结骨干网络）"
        else:
            return "数据增强 + 轻微微调（小学习率）"

    elif 1000 <= dataset_size < 10000:
        if similarity_to_pretrain == 'high':
            return "微调顶层（解冻后几层）"
        else:
            return "全局微调（小学习率 + 数据增强）"

    else:  # > 10000
        if similarity_to_pretrain == 'high':
            return "全局微调"
        else:
            return "全局微调 或 从头训练"
```

### 📊 学习率策略

```python
# 策略1：判别式学习率
optimizer = optim.Adam([
    {'params': model.layer1.parameters(), 'lr': 1e-5},
    {'params': model.layer2.parameters(), 'lr': 1e-4},
    {'params': model.layer3.parameters(), 'lr': 1e-3},
    {'params': model.fc.parameters(), 'lr': 1e-2}
])

# 策略2：渐进式解冻
def progressive_unfreezing(model, epoch, unfreeze_schedule):
    """
    渐进式解冻

    unfreeze_schedule: {epoch: [layer_names]}
    """
    if epoch in unfreeze_schedule:
        for layer_name in unfreeze_schedule[epoch]:
            layer = getattr(model, layer_name)
            for param in layer.parameters():
                param.requires_grad = True
        print(f"Epoch {epoch}: 解冻 {unfreeze_schedule[epoch]}")

# 使用
unfreeze_schedule = {
    0: ['fc'],           # epoch 0: 只训练分类器
    5: ['layer4'],       # epoch 5: 解冻 layer4
    10: ['layer3'],      # epoch 10: 解冻 layer3
    15: ['layer2'],      # epoch 15: 解冻 layer2
}
```

### 🎯 数据增强技巧

```python
# 高级数据增强
from torchvision.transforms import autoaugment, v2

advanced_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),

    # AutoAugment
    autoaugment.AutoAugment(
        autoaugment.AutoAugmentPolicy.IMAGENET
    ),

    # RandAugment
    # autoaugment.RandAugment(),

    # Mixup / CutMix (需要特殊处理)
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
])

# Mixup
class MixupDataset(Dataset):
    def __init__(self, dataset, alpha=0.2):
        self.dataset = dataset
        self.alpha = alpha

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img1, label1 = self.dataset[idx]

        # 随机选择另一个样本
        idx2 = np.random.randint(0, len(self.dataset))
        img2, label2 = self.dataset[idx2]

        # Mixup
        lam = np.random.beta(self.alpha, self.alpha)
        mixed_img = lam * img1 + (1 - lam) * img2

        return mixed_img, (label1, label2, lam)
```

---

## 📝 本章作业

### 作业 1：对比实验

```python
# 在同一个数据集上对比：
# 1. 从零训练
# 2. 特征提取
# 3. 微调（冻结部分层）
# 4. 微调（全局）
# 5. 知识蒸馏

# 记录：
#   - 训练时间
#   - 最终准确率
#   - 参数量
#   - 收敛曲线

# 分析：哪种策略最适合你的数据集？
```

### 作业 2：医学图像分类

```python
# 使用真实医学图像数据集（如 Chest X-Ray）
# 要求：
# 1. EDA 和数据预处理
# 2. 尝试至少 3 种预训练模型
# 3. 实现数据增强
# 4. 使用 Grad-CAM 可视化
# 5. 评估模型性能（准确率、F1、AUC）
# 6. 分析错误案例
# 7. 编写完整报告
```

### 作业 3：少样本学习

```python
# 实现少样本学习
# 任务：5-way 1-shot / 5-shot 分类
# 方法：
# 1. 原型网络
# 2. MAML
# 3. 对比学习（SimCLR）

# 在 Omniglot 或 Mini-ImageNet 上测试
```

### 作业 4：领域自适应

```python
# 实现领域自适应
# 源域：MNIST
# 目标域：SVHN 或 USPS

# 方法：
# 1. 领域对抗训练
# 2. 自训练
# 3. 对比两种方法的效果
```

---

## 🔑 本章关键概念

| 概念 | 说明 |
|------|------|
| 迁移学习 | 利用已有知识加速新任务学习 |
| 预训练模型 | 在大数据集上训练的模型 |
| 特征提取 | 冻结骨干网络，只训练分类器 |
| 微调 | 解冻部分或全部层进行训练 |
| 判别式学习率 | 不同层使用不同学习率 |
| 领域自适应 | 处理源域和目标域分布不同 |
| 少样本学习 |用极少样本学习新任务 |
| 元学习 | 学习如何学习 |
| 知识蒸馏 | 用大模型训练小模型 |
| MAML | 模型无关的元学习 |
| 原型网络 | 基于距离的少样本学习 |
| Grad-CAM | 类激活映射可视化 |

---

## 9.10 进阶话题

### 🔹 多任务学习 (Multi-Task Learning)

```python
class MultiTaskModel(nn.Module):
    """多任务学习模型"""

    def __init__(self, backbone, num_classes_list):
        """
        参数：
            backbone: 共享的特征提取器
            num_classes_list: 每个任务的类别数列表
        """
        super().__init__()

        self.backbone = backbone

        # 为每个任务创建独立的分类器
        self.task_heads = nn.ModuleList([
            nn.Linear(backbone.output_dim, num_classes)
            for num_classes in num_classes_list
        ])

    def forward(self, x):
        # 共享特征提取
        features = self.backbone(x)

        # 多个任务的输出
        outputs = [head(features) for head in self.task_heads]

        return outputs

# 多任务损失
class MultiTaskLoss(nn.Module):
    """多任务损失（不确定性加权）"""

    def __init__(self, num_tasks):
        super().__init__()
        # 可学习的任务权重（log 方差）
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))

    def forward(self, losses):
        """
        参数：
            losses: 每个任务的损失列表
        """
        weighted_losses = []

        for i, loss in enumerate(losses):
            precision = torch.exp(-self.log_vars[i])
            weighted_loss = precision * loss + self.log_vars[i]
            weighted_losses.append(weighted_loss)

        return sum(weighted_losses)

# 使用示例
def train_multitask():
    # 创建模型
    backbone = models.resnet50(pretrained=True)
    backbone.fc = nn.Identity()  # 移除最后的 FC 层
    backbone.output_dim = 2048

    model = MultiTaskModel(
        backbone=backbone,
        num_classes_list=[10, 5, 2]  # 3个任务
    )

    # 多任务损失
    criterion = MultiTaskLoss(num_tasks=3)

    # 训练循环
    for images, (labels1, labels2, labels3) in dataloader:
        outputs = model(images)

        # 计算每个任务的损失
        losses = [
            nn.CrossEntropyLoss()(outputs[0], labels1),
            nn.CrossEntropyLoss()(outputs[1], labels2),
            nn.CrossEntropyLoss()(outputs[2], labels3)
        ]

        # 组合损失
        total_loss = criterion(losses)

        # 反向传播
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
```

---

### 🔹 持续学习 (Continual Learning)

```python
class ElasticWeightConsolidation:
    """弹性权重巩固 (EWC)"""

    def __init__(self, model, dataloader, lambda_ewc=1000):
        self.model = model
        self.lambda_ewc = lambda_ewc

        # 计算 Fisher 信息矩阵
        self.fisher_matrix = self._compute_fisher(dataloader)

        # 保存当前参数
        self.optimal_params = {
            name: param.clone().detach()
            for name, param in model.named_parameters()
        }

    def _compute_fisher(self, dataloader):
        """计算 Fisher 信息矩阵"""
        fisher = {}

        self.model.eval()

        for name, param in self.model.named_parameters():
            fisher[name] = torch.zeros_like(param)

        for inputs, labels in dataloader:
            self.model.zero_grad()

            outputs = self.model(inputs)
            loss = nn.CrossEntropyLoss()(outputs, labels)
            loss.backward()

            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    fisher[name] += param.grad.pow(2)

        # 归一化
        num_samples = len(dataloader.dataset)
        for name in fisher:
            fisher[name] /= num_samples

        return fisher

    def penalty(self):
        """EWC 惩罚项"""
        loss = 0

        for name, param in self.model.named_parameters():
            if name in self.fisher_matrix:
                loss += (self.fisher_matrix[name] *
                        (param - self.optimal_params[name]).pow(2)).sum()

        return self.lambda_ewc * loss

# 使用
def train_with_ewc(model, old_task_loader, new_task_loader):
    """使用 EWC 训练新任务"""

    # 在旧任务上计算 Fisher 矩阵
    ewc = ElasticWeightConsolidation(model, old_task_loader)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    # 在新任务上训练
    for epoch in range(num_epochs):
        for inputs, labels in new_task_loader:
            outputs = model(inputs)

            # 新任务损失 + EWC 惩罚
            loss = criterion(outputs, labels) + ewc.penalty()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

### 🔹 对比学习 (Contrastive Learning)

```python
class SimCLR(nn.Module):
    """SimCLR 对比学习"""

    def __init__(self, base_encoder, projection_dim=128):
        super().__init__()

        # 编码器
        self.encoder = base_encoder

        # 投影头
        self.projector = nn.Sequential(
            nn.Linear(base_encoder.output_dim, 2048),
            nn.ReLU(),
            nn.Linear(2048, projection_dim)
        )

    def forward(self, x):
        # 编码
        features = self.encoder(x)

        # 投影
        z = self.projector(features)

        # L2 归一化
        z = F.normalize(z, dim=1)

        return z

class NTXentLoss(nn.Module):
    """归一化温度交叉熵损失 (NT-Xent)"""

    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z_i, z_j):
        """
        参数：
            z_i, z_j: 两个增强视图的表示 (batch_size, projection_dim)
        """
        batch_size = z_i.size(0)

        # 拼接
        z = torch.cat([z_i, z_j], dim=0)  # (2*batch_size, dim)

        # 计算相似度矩阵
        sim_matrix = torch.mm(z, z.T) / self.temperature

        # 创建标签：对角线外的对应位置为正样本
        labels = torch.arange(batch_size).to(z.device)
        labels = torch.cat([labels + batch_size, labels])

        # 掩码：去掉自己和自己的相似度
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -1e9)

        # 计算损失
        loss = nn.CrossEntropyLoss()(sim_matrix, labels)

        return loss

# 训练 SimCLR
def train_simclr(model, dataloader, num_epochs=100):
    """训练 SimCLR"""

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = NTXentLoss(temperature=0.5)

    for epoch in range(num_epochs):
        for (x_i, x_j), _ in dataloader:  # x_i, x_j 是两个增强视图
            # 前向传播
            z_i = model(x_i)
            z_j = model(x_j)

            # 计算损失
            loss = criterion(z_i, z_j)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch {epoch}: Loss = {loss.item():.4f}')

    return model

# 使用预训练的 SimCLR 进行下游任务
def finetune_simclr(pretrained_encoder, train_loader, num_classes):
    """微调 SimCLR 编码器"""

    # 冻结编码器
    for param in pretrained_encoder.parameters():
        param.requires_grad = False

    # 添加线性分类器
    classifier = nn.Linear(pretrained_encoder.output_dim, num_classes)

    # 训练分类器
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for inputs, labels in train_loader:
            # 提取特征（冻结）
            with torch.no_grad():
                features = pretrained_encoder(inputs)

            # 分类
            outputs = classifier(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## 9.11 实用工具和技巧

### 🛠️ 模型转换和部署

```python
# 1. ONNX 导出
def export_to_onnx(model, dummy_input, output_path):
    """导出为 ONNX 格式"""
    model.eval()

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )

    print(f"模型已导出到 {output_path}")

# 使用
dummy_input = torch.randn(1, 3, 224, 224)
export_to_onnx(model, dummy_input, 'model.onnx')

# 2. TorchScript 转换
scripted_model = torch.jit.script(model)
scripted_model.save('model_scripted.pt')

# 3. 量化（减小模型大小）
def quantize_model(model):
    """动态量化"""
    quantized_model = torch.quantization.quantize_dynamic(
        model,
        {nn.Linear, nn.Conv2d},
        dtype=torch.qint8
    )
    return quantized_model

quantized = quantize_model(model)
```

---

### 📊 模型分析工具

```python
from torchinfo import summary
from fvcore.nn import FlopCountAnalysis, parameter_count

def analyze_model(model, input_size=(1, 3, 224, 224)):
    """分析模型"""

    # 模型摘要
    print("="*70)
    print("模型结构:")
    print("="*70)
    summary(model, input_size=input_size)

    # 参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n总参数: {total_params:,}")
    print(f"可训练参数: {trainable_params:,}")
    print(f"不可训练参数: {total_params - trainable_params:,}")

    # FLOPs
    dummy_input = torch.randn(input_size)
    flops = FlopCountAnalysis(model, dummy_input)
    print(f"\nFLOPs: {flops.total():,}")

    # 内存占用
    print(f"模型大小: {total_params * 4 / 1024 / 1024:.2f} MB (FP32)")

    # 推理速度测试
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    dummy_input = dummy_input.to(device)

    import time

    # 预热
    for _ in range(10):
        _ = model(dummy_input)

    # 测试
    num_iterations = 100
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = time.time()

    with torch.no_grad():
        for _ in range(num_iterations):
            _ = model(dummy_input)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()

    avg_time = (end - start) / num_iterations
    fps = 1 / avg_time

    print(f"\n推理时间: {avg_time*1000:.2f} ms")
    print(f"FPS: {fps:.2f}")

# 使用
analyze_model(model)
```

---

### 🔍 错误分析工具

```python
class ErrorAnalyzer:
    """错误分析工具"""

    def __init__(self, model, dataloader, class_names, device):
        self.model = model
        self.dataloader = dataloader
        self.class_names = class_names
        self.device = device

    def analyze(self):
        """完整错误分析"""
        self.model.eval()

        errors = []

        with torch.no_grad():
            for inputs, labels in self.dataloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                # 找出错误样本
                wrong_mask = preds != labels

                if wrong_mask.any():
                    for i in torch.where(wrong_mask)[0]:
                        errors.append({
                            'image': inputs[i].cpu(),
                            'true_label': labels[i].item(),
                            'pred_label': preds[i].item(),
                            'confidence': probs[i, preds[i]].item(),
                            'true_prob': probs[i, labels[i]].item()
                        })

        print(f"总错误数: {len(errors)}")

        # 按类别统计错误
        self._error_by_class(errors)

        # 按置信度分析
        self._error_by_confidence(errors)

        # 可视化最难样本
        self._visualize_hard_examples(errors)

        return errors

    def _error_by_class(self, errors):
        """按类别统计错误"""
        from collections import defaultdict

        error_count = defaultdict(int)
        confusion = defaultdict(lambda: defaultdict(int))

        for error in errors:
            true_label = error['true_label']
            pred_label = error['pred_label']

            error_count[true_label] += 1
            confusion[true_label][pred_label] += 1

        print("\n每类错误数:")
        for class_id, count in sorted(error_count.items()):
            print(f"  {self.class_names[class_id]}: {count}")

        print("\n最常见的混淆:")
        for true_id, pred_dict in confusion.items():
            for pred_id, count in sorted(pred_dict.items(),
                                        key=lambda x: x[1],
                                        reverse=True)[:3]:
                print(f"  {self.class_names[true_id]} → "
                      f"{self.class_names[pred_id]}: {count}")

    def _error_by_confidence(self, errors):
        """按置信度分析"""
        confidences = [e['confidence'] for e in errors]

        print(f"\n错误预测的置信度:")
        print(f"  平均: {np.mean(confidences):.4f}")
        print(f"  中位数: {np.median(confidences):.4f}")
        print(f"  最大: {np.max(confidences):.4f}")
        print(f"  最小: {np.min(confidences):.4f}")

    def _visualize_hard_examples(self, errors, num_examples=16):
        """可视化最难的样本"""
        # 按置信度排序（高置信度但错误）
        sorted_errors = sorted(errors,
                              key=lambda x: x['confidence'],
                              reverse=True)[:num_examples]

        fig, axes = plt.subplots(4, 4, figsize=(16, 16))
        axes = axes.flatten()

        for i, error in enumerate(sorted_errors):
            if i >= num_examples:
                break

            # 反归一化
            img = error['image'].numpy().transpose(1, 2, 0)
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = std * img + mean
            img = np.clip(img, 0, 1)

            ax = axes[i]
            ax.imshow(img)
            ax.set_title(
                f"真实: {self.class_names[error['true_label']]}\n"
                f"预测: {self.class_names[error['pred_label']]}\n"
                f"置信度: {error['confidence']:.3f}",
                fontsize=10,
                color='red'
            )
            ax.axis('off')

        plt.tight_layout()
        plt.savefig('hard_examples.png', dpi=300)
        plt.show()

# 使用
analyzer = ErrorAnalyzer(model, test_loader, class_names, device)
errors = analyzer.analyze()
```

---

## 📚 推荐资源

### 📖 论文

**迁移学习基础**：
- "A Survey on Transfer Learning" (Pan & Yang, 2010)
- "How transferable are features in deep neural networks?" (Yosinski et al., 2014)

**领域自适应**：
- "Domain-Adversarial Training of Neural Networks" (Ganin et al., 2016)
- "Unsupervised Domain Adaptation by Backpropagation" (Ganin & Lempitsky, 2015)

**少样本学习**：
- "Model-Agnostic Meta-Learning (MAML)" (Finn et al., 2017)
- "Prototypical Networks for Few-shot Learning" (Snell et al., 2017)
- "Matching Networks for One Shot Learning" (Vinyals et al., 2016)

**知识蒸馏**：
- "Distilling the Knowledge in a Neural Network" (Hinton et al., 2015)

**对比学习**：
- "A Simple Framework for Contrastive Learning (SimCLR)" (Chen et al., 2020)
- "Momentum Contrast (MoCo)" (He et al., 2020)

### 🔧 工具和库

```python
# Hugging Face Transformers
from transformers import AutoModel, AutoTokenizer

# Timm (PyTorch Image Models)
import timm
model = timm.create_model('resnet50', pretrained=True)

# PyTorch Lightning (简化训练)
import pytorch_lightning as pl

# Weights & Biases (实验跟踪)
import wandb

# TensorBoard
from torch.utils.tensorboard import SummaryWriter
```

---

## 🎓 总结

### ✅ 迁移学习何时有效？

```
✓ 源任务和目标任务相关
✓ 目标任务数据较少
✓ 源任务数据量大且质量高
✓ 有合适的预训练模型可用
```

### ❌ 迁移学习何时无效？

```
✗ 源任务和目标任务完全不相关
✗ 目标任务数据充足
✗ 预训练模型不匹配目标任务
✗ 计算资源充足，可从头训练
```

### 🎯 实践建议

1. **优先尝试预训练模型**
2. **根据数据量选择策略**（特征提取 vs 微调）
3. **使用判别式学习率**
4. **充分利用数据增强**
5. **监控验证集避免过拟合**
6. **可视化理解模型行为**（Grad-CAM等）
7. **记录实验结果**（Weights & Biases）
8. **错误分析指导改进**

---
