## 📋 课程整体框架

### **第一部分：机器学习基础 (ML Basics)**

#### 1. **引入：机器学习是什么？**
- 🎯 **从生活例子出发**
  - "机器学习就像是教电脑'举一反三'"
  - 实例：语音识别、图像识别、AlphaGo
- **三个步骤框架**
  1. Define a function set (Model)
  2. Goodness of function (Loss)
  3. Pick the best function (Optimization)

#### 2. **回归 (Regression)**
- Linear Regression 实例
- 用宝可梦CP值预测演示
- Gradient Descent 直观解释
- 可视化：Loss Function 地形图

#### 3. **分类 (Classification)**
- Logistic Regression
- 为什么不能用 Regression 做 Classification？
- Softmax & Cross-Entropy
- 实例：手写数字识别 (MNIST)

---

### **第二部分：深度学习 (Deep Learning)**

#### 4. **神经网络基础**
- 🧠 **直观理解**："神经网络就是很多 Logistic Regression 叠起来"
- Activation Functions (Sigmoid, ReLU)
- Backpropagation（用计算图解释）
- 实战：建立你的第一个神经网络

#### 5. **训练技巧 (Tips for Training)**
  ```
  模型表现不好？
  ├─ 训练数据表现差？
  │  ├─ Optimization 问题 → Adaptive Learning Rate, Batch Normalization
  │  └─ Overfitting → Regularization, Dropout, Early Stopping
  └─ 测试数据表现差？
     └─ Overfitting → 更多数据, Data Augmentation
  ```

#### 6. **CNN (卷积神经网络)**
- 为什么需要 CNN？（参数太多的问题）
- Convolution & Pooling 直观解释
- 经典架构：LeNet, AlexNet, VGG, ResNet
- 应用：图像分类、物体检测

#### 7. **RNN & Sequence Models**
- 序列数据的特性
- RNN, LSTM, GRU
- Seq2Seq 架构
- 应用：语音识别、机器翻译

---

### **第三部分：进阶主题**

#### 8. **Self-Attention & Transformer**
- 🔥 **从 RNN 的限制谈起**
- Attention Mechanism 图解
- Multi-Head Attention
- Transformer 架构完整解析
- BERT, GPT 简介

#### 9. **生成模型 (Generative Models)**
- Auto-encoder
- VAE (Variational Auto-encoder)
- GAN (对抗生成网络)
  - Generator vs. Discriminator 的对抗游戏
  - 训练难点与技巧
- Diffusion Models 简介

#### 10. **强化学习 (Reinforcement Learning)**
- Agent, Environment, Reward
- Q-Learning
- Policy Gradient
- 实例：玩 Atari 游戏、AlphaGo 原理

#### 11. **无监督学习 (Unsupervised Learning)**
- Clustering (K-means, HAC)
- Dimension Reduction (PCA, t-SNE)
- Self-Supervised Learning

---

### **第四部分：实践与应用**

#### 12. **迁移学习 (Transfer Learning)**
- Pre-training & Fine-tuning
- Domain Adaptation

#### 13. **可解释性与对抗攻击**
- Explainable AI
- Adversarial Attack & Defense

#### 14. **大语言模型时代**
- LLM 原理
- Prompt Engineering
- In-Context Learning
- 未来展望

---

---

## 📚 推荐作业设置

1. **HW1**: Linear Regression (PM2.5 预测)
2. **HW2**: Classification (收入预测)
3. **HW3**: CNN (图像分类)
4. **HW4**: RNN (文本情感分析)
5. **HW5**: Transformer (机器翻译)
6. **HW6**: GAN (动漫人物生成)
7. **Final Project**: 开放式竞赛

---

## 🛠️ 工具与资源

- **编程环境**: Python + PyTorch/TensorFlow
- **平台**: Google Colab (免费 GPU)
- **数据集**: MNIST, CIFAR-10, ImageNet, Common Voice

---

-----
