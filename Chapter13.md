# 第十三章：大语言模型时代

## 📌 章节目标
- 理解大语言模型 (LLM) 的核心原理
- 掌握 Prompt Engineering 技巧
- 学习 In-Context Learning 和 Few-Shot Learning
- 了解 LLM 的微调方法（LoRA, PEFT）
- 探索 LLM 的应用和未来方向

---

## 13.1 从 GPT 到 ChatGPT：大语言模型的演进

### 🌟 关键里程碑

```
2017: Transformer (Attention Is All You Need)
      ↓
2018: GPT-1 (117M 参数)
      BERT (340M 参数)
      ↓
2019: GPT-2 (1.5B 参数)
      T5, BART, XLNet
      ↓
2020: GPT-3 (175B 参数) 👑
      - Few-shot learning
      - In-context learning
      ↓
2022: ChatGPT (GPT-3.5 + RLHF)
      InstructGPT
      ↓
2023: GPT-4 (多模态)
      Claude, LLaMA, PaLM
      ↓
2024: Gemini, Claude 3
      开源模型爆发
```

---

### 📐 核心能力的涌现

**规模定律 (Scaling Laws)**：

```
性能 ∝ log(模型大小 × 数据量 × 计算量)

涌现能力 (Emergent Abilities):
  - 少样本学习
  - 指令遵循
  - 思维链推理
  - 代码生成
  - 多步推理
```

---

## 13.2 LLM 的架构原理

### 🏗️ Transformer 回顾

```
输入 Token
    ↓
[Embedding + Positional Encoding]
    ↓
[Transformer Block] ×N
  - Multi-Head Attention
  - Feed-Forward Network
  - Layer Normalization
  - Residual Connection
    ↓
[Language Model Head]
    ↓
下一个 Token 的概率分布
```

### 🔹 GPT 架构特点

**只用 Decoder（自回归）**：

```python
class GPTBlock(nn.Module):
    """GPT Transformer 块"""

    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()

        # Causal Self-Attention（只看前文）
        self.attn = MultiHeadAttention(d_model, n_heads, dropout, causal=True)

        # Feed-Forward
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout)
        )

        # Layer Norm
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Pre-LN 架构
        x = x + self.attn(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x

class GPTModel(nn.Module):
    """简化的 GPT 模型"""

    def __init__(self, vocab_size, d_model=768, n_heads=12,
                 n_layers=12, d_ff=3072, max_len=1024, dropout=0.1):
        super().__init__()

        # Token Embedding
        self.token_emb = nn.Embedding(vocab_size, d_model)

        # Positional Embedding（可学习）
        self.pos_emb = nn.Embedding(max_len, d_model)

        # Transformer Blocks
        self.blocks = nn.ModuleList([
            GPTBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ])

        # Final Layer Norm
        self.ln_f = nn.LayerNorm(d_model)

        # Language Model Head
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

        # 权重共享（embedding 和 lm_head）
        self.lm_head.weight = self.token_emb.weight

        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, labels=None):
        """
        参数:
            input_ids: (batch, seq_len)
            labels: (batch, seq_len) 可选，用于训练
        """
        batch_size, seq_len = input_ids.shape

        # Embedding
        token_embeddings = self.token_emb(input_ids)  # (B, T, D)

        # Positional Embedding
        positions = torch.arange(0, seq_len, device=input_ids.device)
        position_embeddings = self.pos_emb(positions)  # (T, D)

        x = self.dropout(token_embeddings + position_embeddings)

        # Causal Mask（下三角矩阵）
        mask = torch.tril(torch.ones(seq_len, seq_len, device=input_ids.device))
        mask = mask.view(1, 1, seq_len, seq_len)

        # Transformer Blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.ln_f(x)

        # Logits
        logits = self.lm_head(x)  # (B, T, vocab_size)

        # 计算损失（如果提供了标签）
        loss = None
        if labels is not None:
            # 移位：预测下一个 token
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
                ignore_index=-100
            )

        return logits, loss

    def generate(self, input_ids, max_new_tokens=100,
                temperature=1.0, top_k=None, top_p=None):
        """
        自回归生成

        参数:
            input_ids: (batch, seq_len) 输入序列
            max_new_tokens: 生成的最大 token 数
            temperature: 温度参数（控制随机性）
            top_k: Top-K 采样
            top_p: Nucleus (Top-P) 采样
        """
        for _ in range(max_new_tokens):
            # 截断到最大长度（避免超过位置编码）
            input_ids_cond = input_ids if input_ids.size(1) <= 1024 else input_ids[:, -1024:]

            # 前向传播
            logits, _ = self.forward(input_ids_cond)

            # 获取最后一个位置的 logits
            logits = logits[:, -1, :] / temperature

            # Top-K 采样
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')

            # Top-P (Nucleus) 采样
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

                # 移除累积概率超过 top_p 的 tokens
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0

                for i in range(logits.size(0)):
                    indices_to_remove = sorted_indices[i, sorted_indices_to_remove[i]]
                    logits[i, indices_to_remove] = -float('Inf')

            # 采样
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)

            # 拼接
            input_ids = torch.cat([input_ids, next_token], dim=1)

        return input_ids
```

---

## 13.3 预训练：自监督学习

### 🎯 预训练任务

```
语言模型目标：
  给定上文 x₁, x₂, ..., x_{t-1}，预测 x_t

  P(x_t | x₁, ..., x_{t-1})

损失函数：
  L = -∑_t log P(x_t | x₁, ..., x_{t-1})
```

### 💻 预训练流程

```python
def train_gpt(model, dataloader, num_epochs=10):
    """预训练 GPT"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4,
                                  betas=(0.9, 0.95), weight_decay=0.1)

    # 学习率调度（warmup + cosine decay）
    def get_lr(step, warmup_steps=2000, max_steps=100000):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (max_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: get_lr(step)
    )

    step = 0
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = input_ids.clone()

            # 前向传播
            logits, loss = model(input_ids, labels)

            # 反向传播
            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            total_loss += loss.item()
            step += 1

            if step % 100 == 0:
                print(f'Step {step}: Loss = {loss.item():.4f}, '
                      f'LR = {scheduler.get_last_lr()[0]:.6f}')

        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}: Avg Loss = {avg_loss:.4f}')

    return model
```

---

## 13.4 Prompt Engineering

### 🎨 什么是 Prompt？

```
Prompt = 给模型的指令/示例

例：
  输入："将下面的英文翻译成中文：\nHello, world!"
  模型：根据 prompt 理解任务，生成翻译
```

### 📊 Prompt 设计原则

#### **1. 清晰明确**

```
❌ 差的 Prompt:
  "关于 AI"

✅ 好的 Prompt:
  "请用 200 字介绍人工智能的定义、发展历程和主要应用领域。"
```

#### **2. 提供上下文**

```
❌ 无上下文:
  "这个怎么样？"

✅ 有上下文:
  "我正在写一篇关于气候变化的文章。以下是草稿的第一段：
  [段落内容]
  请评价这段内容的逻辑性和说服力。"
```

#### **3. 使用示例（Few-shot）**

```
Zero-shot（无示例）:
  "情感分类：这部电影很好看"

Few-shot（有示例）:
  """
  情感分类任务：

  示例：
  文本: "这家餐厅太差了" → 负面
  文本: "服务态度很好" → 正面
  文本: "还行吧" → 中性

  现在分类：
  文本: "这部电影很好看" →
  """
```

#### **4. 思维链（Chain-of-Thought）**

```
普通 Prompt:
  "Roger 有 5 个网球。他又买了 2 罐网球，每罐 3 个球。
   他现在有多少个网球？"

CoT Prompt:
  "Roger 有 5 个网球。他又买了 2 罐网球，每罐 3 个球。
   他现在有多少个网球？

   让我们一步步思考：
   1. Roger 最初有 5 个网球
   2. 他买了 2 罐，每罐 3 个，所以买了 2×3=6 个
   3. 总共：5+6=11 个

   答案：11 个网球"
```

---

### 💻 Prompt 模板示例

```python
class PromptTemplate:
    """Prompt 模板管理"""

    def __init__(self):
        self.templates = {
            'translation': """
Translate the following {source_lang} text to {target_lang}:

Text: {text}

Translation:""",

            'summarization': """
Summarize the following text in {num_sentences} sentences:

{text}

Summary:""",

            'classification': """
Classify the sentiment of the following text as Positive, Negative, or Neutral.

Examples:
{examples}

Text: {text}
Sentiment:""",

            'qa': """
Answer the following question based on the context.

Context: {context}

Question: {question}

Answer:""",

            'cot_reasoning': """
Question: {question}

Let's think step by step:
""",
        }

    def format(self, template_name, **kwargs):
        """格式化模板"""
        template = self.templates[template_name]
        return template.format(**kwargs)

# ==================== 使用示例 ====================

prompt_template = PromptTemplate()

# 翻译
translation_prompt = prompt_template.format(
    'translation',
    source_lang='English',
    target_lang='Chinese',
    text='Hello, how are you?'
)

# Few-shot 分类
examples = """
Text: "This movie is amazing!" → Positive
Text: "Waste of time." → Negative
Text: "It's okay." → Neutral"""

classification_prompt = prompt_template.format(
    'classification',
    examples=examples,
    text='I love this product!'
)

print(classification_prompt)
```

---

### 🔧 高级 Prompt 技巧

#### **1. 角色扮演 (Role-Playing)**

```
"你是一位资深的 Python 程序员。请帮我优化以下代码：
[代码]"
```

#### **2. 约束条件**

```
"用简单易懂的语言（适合 10 岁儿童理解）解释什么是神经网络。
要求：
- 不超过 3 段
- 使用日常生活的比喻
- 避免专业术语"
```

#### **3. 输出格式**

```
"分析以下产品评论，以 JSON 格式输出：
{
  "sentiment": "positive/negative/neutral",
  "key_points": ["point1", "point2"],
  "rating": 1-5
}

评论：[评论内容]"
```

#### **4. 自我一致性 (Self-Consistency)**

```
多次生成答案，选择最一致的结果
```

```python
def self_consistency_generate(model, tokenizer, prompt, n=5):
    """自我一致性生成"""

    answers = []

    for _ in range(n):
        # 生成答案（带随机性）
        inputs = tokenizer(prompt, return_tensors='pt')
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=100,
            temperature=0.7,
            do_sample=True
        )

        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        answers.append(answer)

    # 选择最常见的答案
    from collections import Counter
    most_common = Counter(answers).most_common(1)[0][0]

    return most_common, answers
```

---

## 13.5 In-Context Learning

### 🎯 核心概念

```
In-Context Learning:
  在输入中提供示例，无需更新参数

关键特性：
  ✓ 无需梯度更新
  ✓ 即时适应新任务
  ✓ 灵活性高
```

### 📊 Few-Shot Learning 示例

```python
class FewShotLearner:
    """Few-Shot Learning 包装器"""

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def create_few_shot_prompt(self, task, examples, query):
        """
        创建 Few-Shot Prompt

        参数:
            task: 任务描述
            examples: [(input, output), ...]
            query: 查询输入
        """
        prompt = f"{task}\n\n"

        # 添加示例
        for i, (inp, out) in enumerate(examples, 1):
            prompt += f"Example {i}:\n"
            prompt += f"Input: {inp}\n"
            prompt += f"Output: {out}\n\n"

        # 添加查询
        prompt += f"Now solve:\n"
        prompt += f"Input: {query}\n"
        prompt += f"Output:"

        return prompt

    def predict(self, task, examples, query, max_tokens=100):
        """Few-Shot 预测"""

        # 创建 prompt
        prompt = self.create_few_shot_prompt(task, examples, query)

        # 生成
        inputs = self.tokenizer(prompt, return_tensors='pt')
        outputs = self.model.generate(
            inputs['input_ids'],
            max_new_tokens=max_tokens,
            temperature=0.7,
            top_p=0.9
        )

        # 解码
        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # 提取输出部分
        result = result.split("Output:")[-1].strip()

        return result

# ==================== 使用示例 ====================

# 情感分类 Few-Shot
task = "Classify the sentiment of the text as Positive or Negative."

examples = [
    ("I love this product!", "Positive"),
    ("Terrible experience.", "Negative"),
    ("Best purchase ever!", "Positive"),
    ("Complete waste of money.", "Negative"),
]

query = "This exceeded my expectations."

learner = FewShotLearner(model, tokenizer)
result = learner.predict(task, examples, query)

print(f"Query: {query}")
print(f"Prediction: {result}")
```

---

### 🔹 示例选择策略

```python
def select_diverse_examples(example_pool, query, n=5, method='semantic'):
    """
    选择多样化的示例

    方法:
        - random: 随机选择
        - semantic: 基于语义相似度
        - diverse: 最大化多样性
    """

    if method == 'random':
        return random.sample(example_pool, n)

    elif method == 'semantic':
        from sentence_transformers import SentenceTransformer

        # 加载句子编码器
        encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # 编码查询和示例
        query_emb = encoder.encode([query])[0]
        example_embs = encoder.encode([ex[0] for ex in example_pool])

        # 计算相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_emb], example_embs)[0]

        # 选择最相似的
        top_indices = np.argsort(similarities)[-n:][::-1]
        return [example_pool[i] for i in top_indices]

    elif method == 'diverse':
        # k-means 聚类选择多样化示例
        from sklearn.cluster import KMeans
        from sentence_transformers import SentenceTransformer

        encoder = SentenceTransformer('all-MiniLM-L6-v2')
        example_embs = encoder.encode([ex[0] for ex in example_pool])

        # 聚类
        kmeans = KMeans(n_clusters=n, random_state=42)
        kmeans.fit(example_embs)

        # 从每个簇选择最接近中心的示例
        selected = []
        for i in range(n):
            cluster_indices = np.where(kmeans.labels_ == i)[0]
            center = kmeans.cluster_centers_[i]

            # 找最接近中心的
            distances = np.linalg.norm(example_embs[cluster_indices] - center, axis=1)
            closest_idx = cluster_indices[np.argmin(distances)]
            selected.append(example_pool[closest_idx])

        return selected
```

---

## 13.6 指令微调 (Instruction Tuning)

### 🎯 从 GPT-3 到 InstructGPT

```
问题：
  GPT-3 虽然强大，但不总是遵循用户指令

解决：
  Instruction Tuning + RLHF

流程：
  1. 收集指令-响应数据
  2. 监督微调 (SFT)
  3. 收集人类偏好数据
  4. 训练奖励模型 (RM)
  5. 强化学习微调 (PPO)
```

### 💻 监督微调 (SFT)

```python
def instruction_tuning(model, instruction_dataset, num_epochs=3):
    """
    指令微调

    数据格式:
    {
        "instruction": "将以下英文翻译成中文",
        "input": "Hello, world!",
        "output": "你好，世界！"
    }
    """

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0

        for batch in instruction_dataset:
            # 格式化为 prompt
            prompts = []
            for item in batch:
                prompt = f"### Instruction:\n{item['instruction']}\n\n"
                if item.get('input'):
                    prompt += f"### Input:\n{item['input']}\n\n"
                prompt += f"### Response:\n{item['output']}"
                prompts.append(prompt)

            # Tokenize
            inputs = tokenizer(prompts, return_tensors='pt',
                             padding=True, truncation=True)
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # 前向传播
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss

            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f'Epoch {epoch+1}: Loss = {total_loss/len(instruction_dataset):.4f}')

    return model
```

---

### 🔹 RLHF (Reinforcement Learning from Human Feedback)

```python
class RewardModel(nn.Module):
    """奖励模型"""

    def __init__(self, base_model):
        super().__init__()
        self.base_model = base_model

        # 奖励头
        self.reward_head = nn.Linear(base_model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None):
        # 获取最后一个 token 的隐藏状态
        outputs = self.base_model(input_ids, attention_mask=attention_mask)
        last_hidden = outputs.last_hidden_state[:, -1, :]

        # 预测奖励
        reward = self.reward_head(last_hidden)

        return reward

def train_reward_model(model, comparison_dataset):
    """
    训练奖励模型

    数据格式：(prompt, response_A, response_B, preference)
    preference: 0 表示 A 更好，1 表示 B 更好
    """

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

    for epoch in range(num_epochs):
        for batch in comparison_dataset:
            prompts, responses_A, responses_B, preferences = batch

            # 计算奖励
            rewards_A = model(responses_A)
            rewards_B = model(responses_B)

            # 损失：偏好的响应应该有更高奖励
            loss = -torch.log(torch.sigmoid(
                (rewards_A - rewards_B) * (2 * preferences - 1)
            )).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return model

def rlhf_training(policy_model, reward_model, prompts):
    """
    使用 PPO 进行 RLHF

    简化版本（实际实现更复杂）
    """

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-6)

    for iteration in range(num_iterations):
        for prompt in prompts:
            # 生成响应
            with torch.no_grad():
                response = policy_model.generate(prompt)

            # 计算奖励
            reward = reward_model(response)

            # PPO 更新（简化）
            # 实际需要：old_log_probs, advantages, clip_epsilon 等
            log_probs = policy_model.compute_log_probs(prompt, response)
            policy_loss = -(log_probs * reward).mean()

            # KL 散度惩罚（防止偏离太远）
            ref_log_probs = ref_model.compute_log_probs(prompt, response)
            kl_penalty = (log_probs - ref_log_probs).mean()

            loss = policy_loss + 0.1 * kl_penalty

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
```

---

## 13.7 高效微调：LoRA 和 PEFT

### 🎯 为什么需要高效微调？

```
问题：
  大模型全参数微调成本高昂
  - GPT-3 (175B 参数)
  - 需要数百 GB 显存
  - 训练时间长

解决：
  Parameter-Efficient Fine-Tuning (PEFT)
  - 只训练少量参数
  - 保持性能
```

---

### 🔹 LoRA (Low-Rank Adaptation)

**核心思想**：低秩分解

```
原始权重更新：
  W' = W + ΔW

LoRA：
  W' = W + BA

  其中 B ∈ ℝ^(d×r), A ∈ ℝ^(r×k), r << min(d, k)

参数量：
  原始：d × k
  LoRA：r × (d + k)  （减少 >90%）
```

```python
class LoRALayer(nn.Module):
    """LoRA 层"""

    def __init__(self, in_features, out_features, rank=8, alpha=16):
        super().__init__()

        self.rank = rank
        self.alpha = alpha

        # 低秩矩阵
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        # 初始化
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 缩放因子
        self.scaling = alpha / rank

    def forward(self, x):
        # LoRA 路径：x @ A^T @ B^T
        lora_out = (x @ self.lora_A.T) @ self.lora_B.T
        return lora_out * self.scaling

class LoRALinear(nn.Module):
    """带 LoRA 的线性层"""

    def __init__(self, linear_layer, rank=8, alpha=16):
        super().__init__()

        # 冻结原始权重
        self.linear = linear_layer
        for param in self.linear.parameters():
            param.requires_grad = False

        # 添加 LoRA
        self.lora = LoRALayer(
            linear_layer.in_features,
            linear_layer.out_features,
            rank, alpha
        )

    def forward(self, x):
        # 原始输出 + LoRA 增量
        return self.linear(x) + self.lora(x)

# ==================== 应用 LoRA ====================

def apply_lora_to_model(model, rank=8, alpha=16, target_modules=['q_proj', 'v_proj']):
    """
    为模型添加 LoRA

    通常只对 attention 的 Q, V 矩阵添加 LoRA
    """

    for name, module in model.named_modules():
        # 检查是否是目标模块
        if any(target in name for target in target_modules):
            if isinstance(module, nn.Linear):
                # 替换为 LoRA 版本
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]

                parent_module = model.get_submodule(parent_name)
                lora_linear = LoRALinear(module, rank, alpha)

                setattr(parent_module, child_name, lora_linear)

                print(f"Applied LoRA to {name}")

    return model

# ==================== 训练 LoRA ====================

def train_with_lora(model, dataloader, num_epochs=3):
    """使用 LoRA 微调"""

    # 应用 LoRA
    model = apply_lora_to_model(model)

    # 只优化 LoRA 参数
    lora_params = [p for n, p in model.named_parameters() if 'lora' in n]
    optimizer = torch.optim.AdamW(lora_params, lr=1e-4)

    print(f"Total params: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Trainable params: {sum(p.numel() for p in lora_params):,}")

    # 训练循环（与普通微调相同）
    for epoch in range(num_epochs):
        # ... 训练代码
        pass

    return model
```

---

### 🔹 其他 PEFT 方法

#### **Adapter Tuning**

```python
class Adapter(nn.Module):
    """Adapter 模块"""

    def __init__(self, hidden_size, bottleneck_size=64):
        super().__init__()

        self.down_proj = nn.Linear(hidden_size, bottleneck_size)
        self.up_proj = nn.Linear(bottleneck_size, hidden_size)
        self.activation = nn.ReLU()

    def forward(self, x):
        residual = x
        x = self.down_proj(x)
        x = self.activation(x)
        x = self.up_proj(x)
        return x + residual  # 残差连接
```

#### **Prefix Tuning**

```python
class PrefixTuning(nn.Module):
    """Prefix Tuning"""

    def __init__(self, num_prefix_tokens, d_model):
        super().__init__()

        # 可学习的 prefix embeddings
        self.prefix_embeddings = nn.Parameter(
            torch.randn(num_prefix_tokens, d_model)
        )

    def forward(self, input_embeddings):
        batch_size = input_embeddings.size(0)

        # 扩展 prefix 到 batch
        prefix = self.prefix_embeddings.unsqueeze(0).expand(batch_size, -1, -1)

        # 拼接到输入前面
        return torch.cat([prefix, input_embeddings], dim=1)
```

---

## 13.8 LLM 应用范式

### 🔹 检索增强生成 (RAG)

```
RAG = Retrieval + Generation

流程：
  1. 检索相关文档
  2. 将文档作为上下文
  3. 生成答案
```

```python
class RAGSystem:
    """检索增强生成系统"""

    def __init__(self, llm, retriever, top_k=3):
        self.llm = llm
        self.retriever = retriever
        self.top_k = top_k

    def answer_question(self, question, knowledge_base):
        """
        基于知识库回答问题

        参数:
            question: 用户问题
            knowledge_base: 文档列表
        """
        # 1. 检索相关文档
        relevant_docs = self.retriever.retrieve(
            question, knowledge_base, top_k=self.top_k
        )

        # 2. 构建 prompt
        context = "\n\n".join([
            f"Document {i+1}:\n{doc}"
            for i, doc in enumerate(relevant_docs)
        ])

        prompt = f"""Answer the question based on the context below.

Context:
{context}

Question: {question}

Answer:"""

        # 3. 生成答案
        answer = self.llm.generate(prompt)

        return answer, relevant_docs

class SimpleRetriever:
    """简单的基于嵌入的检索器"""

    def __init__(self, encoder_model='all-MiniLM-L6-v2'):
        from sentence_transformers import SentenceTransformer
        self.encoder = SentenceTransformer(encoder_model)

    def retrieve(self, query, documents, top_k=3):
        """检索最相关的文档"""

        # 编码
        query_emb = self.encoder.encode([query])[0]
        doc_embs = self.encoder.encode(documents)

        # 计算相似度
        from sklearn.metrics.pairwise import cosine_similarity
        similarities = cosine_similarity([query_emb], doc_embs)[0]

        # 返回 top-k
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        return [documents[i] for i in top_indices]
```

---

### 🔹 Agent 系统

```python
class LLMAgent:
    """基于 LLM 的 Agent"""

    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}

    def run(self, task, max_steps=5):
        """
        执行任务

        流程：
        1. 思考下一步
        2. 选择工具
        3. 执行动作
        4. 观察结果
        5. 重复直到完成
        """

        history = []

        for step in range(max_steps):
            # 构建 prompt
            prompt = self._build_agent_prompt(task, history)

            # LLM 决策
            response = self.llm.generate(prompt)

            # 解析动作
            action = self._parse_action(response)

            if action['type'] == 'FINISH':
                return action['answer']

            # 执行工具
            tool_name = action['tool']
            tool_input = action['input']

            if tool_name in self.tools:
                observation = self.tools[tool_name].run(tool_input)
            else:
                observation = f"Tool {tool_name} not found."

            # 记录
            history.append({
                'thought': action.get('tho