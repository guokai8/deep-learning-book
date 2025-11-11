# 第十章：强化学习 (Reinforcement Learning)

## 📌 章节目标
- 理解强化学习的基本概念和框架
- 掌握 Q-Learning 和 DQN 算法
- 学习策略梯度方法
- 了解 Actor-Critic 架构
- 实战：训练 Agent 玩游戏

---

## 10.1 强化学习基础

### 🎯 什么是强化学习？

**定义**：智能体（Agent）通过与环境（Environment）交互，学习最优策略以最大化累积奖励。

**与监督学习的区别**：

```
监督学习：
  输入 → 模型 → 输出
  有明确的标签指导

强化学习：
  状态 → 动作 → 奖励 + 新状态
  没有明确标签，只有奖励信号
  需要探索和试错
```

### 📐 核心概念

#### **马尔可夫决策过程 (MDP)**

```
MDP = (S, A, P, R, γ)

S: 状态空间 (States)
A: 动作空间 (Actions)
P: 状态转移概率 P(s'|s,a)
R: 奖励函数 R(s,a,s')
γ: 折扣因子 (0 ≤ γ < 1)
```

**示例：迷宫游戏**

```
┌───┬───┬───┬───┐
│ S │   │   │   │  S = 起点
├───┼───┼───┼───┤  G = 终点
│   │ ▓ │   │   │  ▓ = 障碍
├───┼───┼───┼───┤
│   │   │ ▓ │   │  动作: ↑↓←→
├───┼───┼───┼───┤
│   │   │   │ G │  奖励: 到达G=+10, 碰壁=-1
└───┴───┴───┴───┘
```

---

#### **关键术语**

**1. 策略 (Policy) π**

```
π(a|s): 在状态 s 下选择动作 a 的概率

确定性策略: a = π(s)
随机策略: a ~ π(·|s)
```

**2. 价值函数 (Value Function)**

```
状态价值函数 V^π(s):
  V^π(s) = E[∑(t=0 to ∞) γ^t·r_t | s_0=s, π]

  表示：从状态 s 开始，遵循策略 π 的期望累积奖励

动作价值函数 Q^π(s,a):
  Q^π(s,a) = E[∑(t=0 to ∞) γ^t·r_t | s_0=s, a_0=a, π]

  表示：在状态 s 执行动作 a 后，遵循策略 π 的期望累积奖励
```

**3. 最优策略**

```
π* = argmax_π V^π(s)  对所有 s

最优价值函数:
  V*(s) = max_π V^π(s)
  Q*(s,a) = max_π Q^π(s,a)

贝尔曼最优方程:
  V*(s) = max_a [R(s,a) + γ·∑_s' P(s'|s,a)·V*(s')]
  Q*(s,a) = R(s,a) + γ·∑_s' P(s'|s,a)·max_a' Q*(s',a')
```

---

#### **探索 vs 利用 (Exploration vs Exploitation)**

```
利用 (Exploitation):
  选择当前已知最好的动作

探索 (Exploration):
  尝试新动作，发现更好的策略

平衡策略:
  ε-greedy: 以概率 ε 随机探索，否则利用
  Softmax: 根据 Q 值分布采样
  UCB (Upper Confidence Bound)
```

---

## 10.2 Q-Learning

### 📐 算法原理

**核心思想**：学习动作价值函数 Q(s,a)

**更新规则**：

```
Q(s,a) ← Q(s,a) + α·[r + γ·max_a' Q(s',a') - Q(s,a)]

其中：
  α: 学习率
  r: 即时奖励
  γ: 折扣因子
  s': 下一状态

时序差分误差 (TD Error):
  δ = r + γ·max_a' Q(s',a') - Q(s,a)
```

### 💻 从零实现 Q-Learning

```python
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class QLearningAgent:
    """Q-Learning 智能体"""

    def __init__(self, n_states, n_actions,
                 learning_rate=0.1, discount_factor=0.95,
                 epsilon=0.1, epsilon_decay=0.995, epsilon_min=0.01):
        self.n_states = n_states
        self.n_actions = n_actions
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        # Q 表：字典形式
        self.q_table = defaultdict(lambda: np.zeros(n_actions))

    def get_action(self, state, training=True):
        """
        选择动作（ε-greedy 策略）

        参数:
            state: 当前状态
            training: 是否训练模式
        """
        if training and np.random.random() < self.epsilon:
            # 探索：随机动作
            return np.random.randint(self.n_actions)
        else:
            # 利用：选择最优动作
            return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """
        更新 Q 值

        参数:
            state: 当前状态
            action: 执行的动作
            reward: 获得的奖励
            next_state: 下一状态
            done: 是否终止
        """
        # 当前 Q 值
        current_q = self.q_table[state][action]

        # 目标 Q 值
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.q_table[next_state])

        # 更新 Q 值
        self.q_table[state][action] += self.lr * (target_q - current_q)

        # 衰减 epsilon
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_q_values(self, state):
        """获取状态的所有 Q 值"""
        return self.q_table[state]

# ==================== 环境：网格世界 ====================

class GridWorld:
    """简单的网格世界环境"""

    def __init__(self, size=5):
        self.size = size
        self.n_states = size * size
        self.n_actions = 4  # 上下左右

        # 起点和终点
        self.start_pos = (0, 0)
        self.goal_pos = (size-1, size-1)

        # 障碍物
        self.obstacles = [(1, 1), (2, 2), (3, 1)]

        # 当前位置
        self.current_pos = self.start_pos

    def reset(self):
        """重置环境"""
        self.current_pos = self.start_pos
        return self._pos_to_state(self.current_pos)

    def _pos_to_state(self, pos):
        """位置转状态编号"""
        return pos[0] * self.size + pos[1]

    def _state_to_pos(self, state):
        """状态编号转位置"""
        return (state // self.size, state % self.size)

    def step(self, action):
        """
        执行动作

        动作编码: 0=上, 1=下, 2=左, 3=右

        返回: (next_state, reward, done, info)
        """
        row, col = self.current_pos

        # 移动
        if action == 0:  # 上
            row = max(0, row - 1)
        elif action == 1:  # 下
            row = min(self.size - 1, row + 1)
        elif action == 2:  # 左
            col = max(0, col - 1)
        elif action == 3:  # 右
            col = min(self.size - 1, col + 1)

        next_pos = (row, col)

        # 检查障碍物
        if next_pos in self.obstacles:
            next_pos = self.current_pos  # 碰壁不移动
            reward = -1
        elif next_pos == self.goal_pos:
            reward = 10  # 到达终点
        else:
            reward = -0.1  # 每步小惩罚

        self.current_pos = next_pos
        next_state = self._pos_to_state(next_pos)
        done = (next_pos == self.goal_pos)

        return next_state, reward, done, {}

    def render(self):
        """可视化环境"""
        grid = np.zeros((self.size, self.size))

        # 标记障碍物
        for obs in self.obstacles:
            grid[obs] = -1

        # 标记终点
        grid[self.goal_pos] = 2

        # 标记当前位置
        grid[self.current_pos] = 1

        return grid

# ==================== 训练 Q-Learning ====================

def train_qlearning(env, agent, num_episodes=1000):
    """训练 Q-Learning Agent"""

    rewards_history = []
    epsilon_history = []

    for episode in range(num_episodes):
        state = env.reset()
        total_reward = 0
        done = False
        steps = 0

        while not done and steps < 100:
            # 选择动作
            action = agent.get_action(state, training=True)

            # 执行动作
            next_state, reward, done, _ = env.step(action)

            # 更新 Q 值
            agent.update(state, action, reward, next_state, done)

            state = next_state
            total_reward += reward
            steps += 1

        rewards_history.append(total_reward)
        epsilon_history.append(agent.epsilon)

        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(rewards_history[-100:])
            print(f'Episode {episode+1}: Avg Reward = {avg_reward:.2f}, '
                  f'Epsilon = {agent.epsilon:.3f}')

    return rewards_history, epsilon_history

# ==================== 可视化 ====================

def visualize_policy(agent, env):
    """可视化学到的策略"""

    action_symbols = ['↑', '↓', '←', '→']

    fig, ax = plt.subplots(figsize=(8, 8))

    # 绘制网格
    for i in range(env.size):
        for j in range(env.size):
            pos = (i, j)
            state = env._pos_to_state(pos)

            # 背景色
            if pos == env.goal_pos:
                color = 'green'
            elif pos in env.obstacles:
                color = 'gray'
            else:
                color = 'white'

            rect = plt.Rectangle((j, env.size-1-i), 1, 1,
                                 facecolor=color, edgecolor='black')
            ax.add_patch(rect)

            # 最优动作
            if pos not in env.obstacles and pos != env.goal_pos:
                action = agent.get_action(state, training=False)
                ax.text(j+0.5, env.size-1-i+0.5, action_symbols[action],
                       ha='center', va='center', fontsize=20)

    ax.set_xlim(0, env.size)
    ax.set_ylim(0, env.size)
    ax.set_aspect('equal')
    ax.set_title('学到的策略', fontsize=16)
    ax.axis('off')

    plt.tight_layout()
    plt.savefig('policy.png', dpi=300)
    plt.show()

def plot_training_results(rewards_history, epsilon_history):
    """绘制训练结果"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 奖励曲线
    episodes = np.arange(len(rewards_history))
    axes[0].plot(episodes, rewards_history, alpha=0.3, label='原始')

    # 移动平均
    window = 100
    moving_avg = np.convolve(rewards_history,
                            np.ones(window)/window,
                            mode='valid')
    axes[0].plot(episodes[window-1:], moving_avg,
                label=f'{window} Episode 移动平均', linewidth=2)

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('训练奖励')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Epsilon 衰减
    axes[1].plot(epsilon_history)
    axes[1].set_xlabel('Episode')
    axes[1].set_ylabel('Epsilon')
    axes[1].set_title('探索率衰减')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('training_results.png', dpi=300)
    plt.show()

# ==================== 主程序 ====================

if __name__ == '__main__':
    # 创建环境
    env = GridWorld(size=5)

    # 创建 Agent
    agent = QLearningAgent(
        n_states=env.n_states,
        n_actions=env.n_actions,
        learning_rate=0.1,
        discount_factor=0.95,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01
    )

    # 训练
    print("开始训练...")
    rewards_history, epsilon_history = train_qlearning(env, agent, num_episodes=1000)

    # 可视化结果
    plot_training_results(rewards_history, epsilon_history)
    visualize_policy(agent, env)

    # 测试学到的策略
    print("\n测试策略:")
    state = env.reset()
    done = False
    steps = 0

    print("初始状态:")
    print(env.render())

    while not done and steps < 20:
        action = agent.get_action(state, training=False)
        state, reward, done, _ = env.step(action)
        steps += 1

        print(f"\n步骤 {steps}:")
        print(env.render())

        if done:
            print(f"\n成功到达终点！总步数: {steps}")
```

---

## 10.3 Deep Q-Network (DQN)

### 🎯 从 Q-Learning 到 DQN

**Q-Learning 的局限**：

```
问题：
  1. 状态空间大时，Q 表无法存储
     (如 Atari 游戏：像素状态空间巨大)

  2. 无法泛化到未见过的状态

解决：
  用神经网络逼近 Q 函数
  Q(s,a) ≈ Q(s,a;θ)
```

### 📐 DQN 核心创新

**1. 经验回放 (Experience Replay)**

```
问题：
  - 连续样本高度相关
  - 破坏 IID 假设

解决：
  - 存储经验到 Replay Buffer
  - 随机采样 mini-batch 训练

Buffer: (s, a, r, s', done)
```

**2. 目标网络 (Target Network)**

```
问题：
  - 目标 Q 值也在变化
  - 训练不稳定

解决：
  - 使用单独的目标网络 Q'(s,a;θ⁻)
  - 定期从主网络复制参数

损失函数:
  L(θ) = E[(r + γ·max_a' Q'(s',a';θ⁻) - Q(s,a;θ))²]
```

### 💻 实现 DQN

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque
import random
import gym
import numpy as np

# ==================== 神经网络 ====================

class DQN(nn.Module):
    """Deep Q-Network"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        前向传播

        参数:
            state: (batch_size, state_dim)

        返回:
            Q值: (batch_size, action_dim)
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

# ==================== 经验回放 ====================

class ReplayBuffer:
    """经验回放缓冲区"""

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """添加经验"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """随机采样"""
        batch = random.sample(self.buffer, batch_size)

        states, actions, rewards, next_states, dones = zip(*batch)

        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )

    def __len__(self):
        return len(self.buffer)

# ==================== DQN Agent ====================

class DQNAgent:
    """DQN 智能体"""

    def __init__(self, state_dim, action_dim,
                 learning_rate=1e-3, gamma=0.99,
                 epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995,
                 buffer_size=10000, batch_size=64,
                 target_update_freq=10):

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 主网络和目标网络
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # 优化器
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # 经验回放
        self.memory = ReplayBuffer(buffer_size)

        # 训练步数
        self.steps = 0

    def select_action(self, state, training=True):
        """选择动作（ε-greedy）"""
        if training and random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax(dim=1).item()

    def train_step(self):
        """训练一步"""
        if len(self.memory) < self.batch_size:
            return None

        # 采样经验
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)

        # 转为 tensor
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # 当前 Q 值
        current_q_values = self.policy_net(states).gather(1, actions)

        # 目标 Q 值
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # 计算损失
        loss = F.mse_loss(current_q_values, target_q_values)

        # 反向传播
        self.optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10)

        self.optimizer.step()

        # 更新目标网络
        self.steps += 1
        if self.steps % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        # 衰减 epsilon
        if self.epsilon > self.epsilon_end:
            self.epsilon *= self.epsilon_decay

        return loss.item()

    def save(self, path):
        """保存模型"""
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'steps': self.steps,
            'epsilon': self.epsilon
        }, path)

    def load(self, path):
        """加载模型"""
        checkpoint = torch.load(path)
        self.policy_net.load_state_dict(checkpoint['policy_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.steps = checkpoint['steps']
        self.epsilon = checkpoint['epsilon']

# ==================== 训练 DQN ====================

def train_dqn(env_name='CartPole-v1', num_episodes=500):
    """训练 DQN"""

    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 创建 Agent
    agent = DQNAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995,
        buffer_size=10000,
        batch_size=64,
        target_update_freq=10
    )

    # 训练历史
    rewards_history = []
    losses_history = []

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]  # gym 新版本返回 (state, info)

        total_reward = 0
        episode_losses = []
        done = False

        while not done:
            # 选择动作
            action = agent.select_action(state, training=True)

            # 执行动作
            result = env.step(action)
            if len(result) == 5:  # 新版本 gym
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:  # 旧版本
                next_state, reward, done, _ = result

            # 存储经验
            agent.memory.push(state, action, reward, next_state, done)

            # 训练
            loss = agent.train_step()
            if loss is not None:
                episode_losses.append(loss)

            state = next_state
            total_reward += reward

        rewards_history.append(total_reward)
        if episode_losses:
            losses_history.append(np.mean(episode_losses))

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f'Episode {episode+1}: Avg Reward = {avg_reward:.2f}, '
                  f'Epsilon = {agent.epsilon:.3f}, '
                  f'Buffer Size = {len(agent.memory)}')

    env.close()

    return agent, rewards_history, losses_history

# ==================== 可视化 ====================

def plot_dqn_results(rewards_history, losses_history):
    """绘制 DQN 训练结果"""

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # 奖励曲线
    episodes = np.arange(len(rewards_history))
    axes[0].plot(episodes, rewards_history, alpha=0.3)

    window = 10
    moving_avg = np.convolve(rewards_history,
                            np.ones(window)/window,
                            mode='valid')
    axes[0].plot(episodes[window-1:], moving_avg,
                linewidth=2, label=f'{window} Episode MA')

    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Total Reward')
    axes[0].set_title('DQN 训练奖励')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # 损失曲线
    if losses_history:
        axes[1].plot(losses_history)
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Loss')
        axes[1].set_title('DQN 训练损失')
        axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('dqn_results.png', dpi=300)
    plt.show()

# ==================== 主程序 ====================

if __name__ == '__main__':
    print("开始训练 DQN...")
    agent, rewards, losses = train_dqn('CartPole-v1', num_episodes=500)

    # 可视化
    plot_dqn_results(rewards, losses)

    # 保存模型
    agent.save('dqn_model.pth')

    # 测试
    print("\n测试训练好的 Agent:")
    env = gym.make('CartPole-v1', render_mode='human')
    state = env.reset()[0]

    total_reward = 0
    done = False

    while not done:
        action = agent.select_action(state, training=False)
        result = env.step(action)
        state, reward, terminated, truncated, _ = result
        done = terminated or truncated
        total_reward += reward

    print(f"测试总奖励: {total_reward}")
    env.close()
```

---

## 10.4 Policy Gradient 策略梯度

### 🎯 核心思想

**与 Q-Learning 的区别**：

```
Q-Learning (Value-based):
  学习价值函数 Q(s,a)
  间接得到策略：π(s) = argmax_a Q(s,a)

Policy Gradient (Policy-based):
  直接学习策略 π(a|s;θ)
  优化目标：最大化期望回报
```

### 📐 REINFORCE 算法

**目标函数**：

```
J(θ) = E_τ~π_θ [∑_t r_t]

梯度（策略梯度定理）：
∇_θ J(θ) = E_τ~π_θ [∑_t ∇_θ log π_θ(a_t|s_t) · G_t]

其中：
  G_t = ∑_{t'=t}^T γ^{t'-t} · r_{t'}  (累积回报)
```

**直觉理解**：

```
如果动作 a 带来了正回报：
  → 增加 log π(a|s)
  → 提高该动作的概率

如果动作 a 带来了负回报：
  → 减少 log π(a|s)
  → 降低该动作的概率
```

### 💻 实现 REINFORCE

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import gym
import numpy as np
import matplotlib.pyplot as plt

# ==================== 策略网络 ====================

class PolicyNetwork(nn.Module):
    """策略网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        前向传播

        返回动作概率分布
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        action_probs = F.softmax(self.fc3(x), dim=-1)
        return action_probs

# ==================== REINFORCE Agent ====================

class REINFORCEAgent:
    """REINFORCE 智能体"""

    def __init__(self, state_dim, action_dim,
                 learning_rate=1e-3, gamma=0.99):

        self.gamma = gamma
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 策略网络
        self.policy = PolicyNetwork(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)

        # 存储轨迹
        self.saved_log_probs = []
        self.rewards = []

    def select_action(self, state):
        """
        根据策略采样动作

        返回: 动作 + log 概率
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        # 获取动作概率分布
        action_probs = self.policy(state_tensor)

        # 采样动作
        dist = Categorical(action_probs)
        action = dist.sample()

        # 保存 log 概率
        self.saved_log_probs.append(dist.log_prob(action))

        return action.item()

    def compute_returns(self, rewards):
        """
        计算累积回报 (discounted returns)

        G_t = r_t + γ·r_{t+1} + γ²·r_{t+2} + ...
        """
        returns = []
        G = 0

        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)

        # 标准化（减小方差）
        returns = torch.tensor(returns).to(self.device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        return returns

    def update(self):
        """更新策略"""
        # 计算累积回报
        returns = self.compute_returns(self.rewards)

        # 计算策略梯度
        policy_loss = []
        for log_prob, G in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * G)

        policy_loss = torch.cat(policy_loss).sum()

        # 反向传播
        self.optimizer.zero_grad()
        policy_loss.backward()
        self.optimizer.step()

        # 清空轨迹
        self.saved_log_probs = []
        self.rewards = []

        return policy_loss.item()

# ==================== 训练 REINFORCE ====================

def train_reinforce(env_name='CartPole-v1', num_episodes=1000):
    """训练 REINFORCE"""

    # 创建环境
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # 创建 Agent
    agent = REINFORCEAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-3,
        gamma=0.99
    )

    rewards_history = []

    for episode in range(num_episodes):
        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        episode_reward = 0
        done = False

        # 收集一条轨迹
        while not done:
            action = agent.select_action(state)

            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            agent.rewards.append(reward)
            episode_reward += reward
            state = next_state

        # 更新策略
        loss = agent.update()

        rewards_history.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f'Episode {episode+1}: Avg Reward = {avg_reward:.2f}')

    env.close()

    return agent, rewards_history

# ==================== 主程序 ====================

if __name__ == '__main__':
    print("开始训练 REINFORCE...")
    agent, rewards = train_reinforce('CartPole-v1', num_episodes=1000)

    # 可视化
    plt.figure(figsize=(12, 5))

    episodes = np.arange(len(rewards))
    plt.plot(episodes, rewards, alpha=0.3)

    window = 10
    moving_avg = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(episodes[window-1:], moving_avg, linewidth=2,
            label=f'{window} Episode MA')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('REINFORCE 训练奖励')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('reinforce_results.png', dpi=300)
    plt.show()
```

---

## 10.5 Actor-Critic 算法

### 🎯 核心思想

**结合 Value-based 和 Policy-based**：

```
Actor (策略网络):
  π(a|s;θ)
  负责选择动作

Critic (价值网络):
  V(s;w) 或 Q(s,a;w)
  负责评估动作

优势：
  - Actor 提供策略
  - Critic 减小方差（不需要等到 episode 结束）
```

### 📐 优势函数 (Advantage Function)

```
A(s,a) = Q(s,a) - V(s)

含义：
  动作 a 比平均好多少

策略梯度：
  ∇_θ J(θ) = E[∇_θ log π_θ(a|s) · A(s,a)]
```

### 💻 实现 A2C (Advantage Actor-Critic)

```python
class ActorCritic(nn.Module):
    """Actor-Critic 网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()

        # 共享特征提取层
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Actor 头（策略）
        self.actor = nn.Linear(hidden_dim, action_dim)

        # Critic 头（价值）
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        """
        前向传播

        返回: action_probs, state_value
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))

        # Actor 输出动作概率
        action_probs = F.softmax(self.actor(x), dim=-1)

        # Critic 输出状态价值
        state_value = self.critic(x)

        return action_probs, state_value

class A2CAgent:
    """A2C 智能体"""

    def __init__(self, state_dim, action_dim,
                 learning_rate=1e-3, gamma=0.99, entropy_coef=0.01):

        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Actor-Critic 网络
        self.ac_net = ActorCritic(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.ac_net.parameters(), lr=learning_rate)

    def select_action(self, state):
        """选择动作"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)

        action_probs, state_value = self.ac_net(state_tensor)

        # 采样动作
        dist = Categorical(action_probs)
        action = dist.sample()

        return action.item(), dist.log_prob(action), dist.entropy(), state_value

    def update(self, states, actions, rewards, next_states, dones):
        """
        更新网络

        参数为一个 batch 的经验
        """
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # 计算当前状态的价值和动作概率
        action_probs, state_values = self.ac_net(states)

        # 计算下一状态的价值
        with torch.no_grad():
            _, next_state_values = self.ac_net(next_states)
            # TD 目标
            td_targets = rewards + self.gamma * next_state_values.squeeze() * (1 - dones)

        # 优势函数
        advantages = td_targets - state_values.squeeze()

        # Actor 损失（策略梯度）
        dist = Categorical(action_probs)
        log_probs = dist.log_prob(actions)
        actor_loss = -(log_probs * advantages.detach()).mean()

        # Critic 损失（TD error）
        critic_loss = F.mse_loss(state_values.squeeze(), td_targets)

        # 熵正则化（鼓励探索）
        entropy = dist.entropy().mean()

        # 总损失
        total_loss = actor_loss + 0.5 * critic_loss - self.entropy_coef * entropy

        # 反向传播
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.ac_net.parameters(), max_norm=0.5)
        self.optimizer.step()

        return total_loss.item(), actor_loss.item(), critic_loss.item()

# ==================== 训练 A2C ====================

def train_a2c(env_name='CartPole-v1', num_episodes=500, batch_size=5):
    """训练 A2C"""

    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    agent = A2CAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        learning_rate=1e-3,
        gamma=0.99,
        entropy_coef=0.01
    )

    rewards_history = []

    for episode in range(num_episodes):
        states, actions, rewards, next_states, dones = [], [], [], [], []

        state = env.reset()
        if isinstance(state, tuple):
            state = state[0]

        episode_reward = 0
        done = False

        # 收集 batch 个样本
        while not done:
            action, log_prob, entropy, state_value = agent.select_action(state)

            result = env.step(action)
            if len(result) == 5:
                next_state, reward, terminated, truncated, _ = result
                done = terminated or truncated
            else:
                next_state, reward, done, _ = result

            states.append(state)
            actions.append(action)
            rewards.append(reward)
            next_states.append(next_state)
            dones.append(done)

            state = next_state
            episode_reward += reward

            # 达到 batch_size 或 episode 结束，更新
            if len(states) >= batch_size or done:
                loss, actor_loss, critic_loss = agent.update(
                    states, actions, rewards, next_states, dones
                )
                states, actions, rewards, next_states, dones = [], [], [], [], []

        rewards_history.append(episode_reward)

        if (episode + 1) % 10 == 0:
            avg_reward = np.mean(rewards_history[-10:])
            print(f'Episode {episode+1}: Avg Reward = {avg_reward:.2f}')

    env.close()

    return agent, rewards_history

# 运行
if __name__ == '__main__':
    print("开始训练 A2C...")
    agent, rewards = train_a2c('CartPole-v1', num_episodes=500)
```

---

## 10.6 实战：Atari 游戏

### 🎮 环境设置

```python
import gym
from gym.wrappers import AtariPreprocessing, FrameStack
import ale_py

def make_atari_env(env_name='BreakoutNoFrameskip-v4', frame_stack=4):
    """
    创建 Atari 环境

    预处理：
      - 灰度化
      - 降采样到 84x84
      - Frame stacking (堆叠多帧)
    """
    env = gym.make(env_name)

    # Atari 预处理
    env = AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        grayscale_newaxis=False,
        scale_obs=True
    )

    # 堆叠帧
    env = FrameStack(env, num_stack=frame_stack)

    return env
```

### 🧠 CNN-based DQN

```python
class AtariDQN(nn.Module):
    """用于 Atari 游戏的 DQN"""

    def __init__(self, num_actions, frame_stack=4):
        super(AtariDQN, self).__init__()

        # 卷积层
        self.conv1 = nn.Conv2d(frame_stack, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # 全连接层
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def forward(self, x):
        """
        参数:
            x: (batch, frame_stack, 84, 84)
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(x.size(0), -1)  # Flatten

        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)

        return q_values

# 训练（与之前类似，但输入是图像）
# 注意：Atari 游戏训练需要大量计算资源和时间
```

---

## 10.7 高级话题

### 🔹 优先经验回放 (Prioritized Experience Replay)

**核心思想**：重要的经验更频繁地被采样

```python
class PrioritizedReplayBuffer:
    """优先经验回放"""

    def __init__(self, capacity=10000, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.alpha = alpha  # 优先级指数
        self.beta = beta    # 重要性采样指数
        self.buffer = []
        self.priorities = np.zeros(capacity)
        self.pos = 0

    def push(self, transition):
        """添加经验"""
        max_priority = self.priorities.max() if self.buffer else 1.0

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition

        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        """按优先级采样"""
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]

        # 计算采样概率
        probs = priorities ** self.alpha
        probs /= probs.sum()

        # 采样索引
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # 计算重要性采样权重
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        samples = [self.buffer[idx] for idx in indices]

        return samples, indices, weights

    def update_priorities(self, indices, priorities):
        """更新优先级"""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
```

---

### 🔹 Double DQN

**问题**：DQN 倾向于高估 Q 值

**解决**：用主网络选择动作，目标网络评估 Q 值

```python
# 标准 DQN
target_q = rewards + gamma * target_net(next_states).max(1)[0]

# Double DQN
best_actions = policy_net(next_states).argmax(1)
target_q = rewards + gamma * target_net(next_states).gather(1, best_actions)
```

---

### 🔹 Dueling DQN

**架构改进**：分离价值和优势

```python
class DuelingDQN(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()

        self.feature = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU()
        )

        # 状态价值流
        self.value_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 优势流
        self.advantage_stream = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        features = self.feature(state)

        value = self.value_stream(features)
        advantages = self.advantage_stream(features)

        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,·)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))

        return q_values
```

---

### 🔹 PPO (Proximal Policy Optimization)

**目前最流行的策略梯度算法**

```python
class PPOAgent:
    """PPO 智能体"""

    def __init__(self, state_dim, action_dim, clip_epsilon=0.2):
        self.clip_epsilon = clip_epsilon

        self.actor_critic = ActorCritic(state_dim, action_dim)
        self.optimizer = optim.Adam(self.actor_critic.parameters(), lr=3e-4)

    def compute_ppo_loss(self, states, actions, old_log_probs,
                         advantages, returns):
        """
        计算 PPO 损失

        PPO-Clip 目标:
          L = min(r_t(θ)·A_t, clip(r_t(θ), 1-ε, 1+ε)·A_t)

        其中 r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
        """
        # 新的动作概率
        action_probs, state_values = self.actor_critic(states)
        dist = Categorical(action_probs)
        new_log_probs = dist.log_prob(actions)

        # 概率比
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped objective
        surr1 = ratio * advantages
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                          1 + self.clip_epsilon) * advantages
        actor_loss = -torch.min(surr1, surr2).mean()

        # Critic 损失
        critic_loss = F.mse_loss(state_values.squeeze(), returns)

        # 熵 bonus
        entropy = dist.entropy().mean()

        total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

        return total_loss
```

---

## 📝 本章作业

### 作业 1：实现 Q-Learning

```python
# 在网格世界或 FrozenLake 环境中：
# 1. 实现 Q-Learning
# 2. 可视化 Q 表的演化过程
# 3. 对比不同超参数（lr, γ, ε）的影响
# 4. 分析收敛速度
```

### 作业 2：DQN 玩 CartPole

```python
# 1. 实现完整的 DQN
# 2. 添加经验回放和目标网络
# 3. 记录训练曲线
# 4. 对比：
#    - DQN vs Q-Learning
#    - 不同网络架构
#    - 不同超参数
```

### 作业 3：Policy Gradient

```python
# 1. 实现 REINFORCE 和 A2C
# 2. 在 CartPole 或 LunarLander 上训练
# 3. 对比两种算法的：
#    - 收敛速度
#    - 样本效率
#    - 最终性能
# 4. 可视化策略的演化
```

### 作业 4：挑战项目

```python
# 选择以下之一：
#
# 1. 实现 Double DQN 或 Dueling DQN
#    在 Atari 游戏上测试
#
# 2. 实现 PPO
#    训练连续控制任务（如 MuJoCo）
#
# 3. 多智能体强化学习
#    实现简单的合作/竞争环境
```

---

## 🔑 本章关键概念

| 概念 | 说明 |
|------|------|
| MDP | 马尔可夫决策过程 |
| 策略 | 状态到动作的映射 |
| 价值函数 | 期望累积奖励 |
| Q-Learning | 基于值的 RL |
| DQN | 深度 Q 网络 |
| 经验回放 | 打破样本相关性 |
| 目标网络 | 稳定训练 |
| Policy Gradient | 基于策略的 RL |
| REINFORCE | 蒙特卡洛策略梯度 |
| Actor-Critic | 结合价值和策略 |
| PPO | 近端策略优化 |

需要我继续写**第十一章：无监督学习**吗？

-----

