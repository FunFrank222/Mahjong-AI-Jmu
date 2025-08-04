# 麻将AI智能体 (Mahjong AI Agent)

一个基于深度学习的麻将AI智能体，使用卷积神经网络进行决策，支持监督学习训练和Botzone平台部署。

## 项目概述

本项目实现了一个完整的麻将AI系统，包括：
- 基于CNN的深度学习模型
- 完整的特征工程和数据预处理
- 监督学习训练流程
- 训练过程可视化
- Botzone平台兼容接口

## 主要特性

- **深度残差网络**: 20层残差块的CNN架构
- **特征工程**: 38通道4x9特征图表示游戏状态
- **动作空间**: 235维完整动作空间覆盖
- **数据增强**: 支持旋转等数据增强技术
- **训练可视化**: 实时训练曲线和性能监控
- **模型检查点**: 自动保存最佳模型

## 快速开始

### 环境要求

```bash
pip install torch torchvision numpy matplotlib json
```

### 数据准备

1. 准备麻将对局数据（JSON格式）
2. 运行数据预处理：

```bash
python preprocess.py
```

### 模型训练

```bash
python supervised.py
```

训练过程中会自动：
- 保存最佳模型检查点
- 生成训练可视化图表
- 记录详细训练日志

### 训练分析

```bash
python analyze_training.py
```

### Botzone部署

将训练好的模型和代码打包上传到Botzone平台，主程序会自动处理游戏请求。

## 模型架构详解

### CNN模型 (CNNModel)

- **输入**: 38通道 × 4×9 特征图
- **架构**: 
  - 初始卷积层 (38→256通道)
  - 20个残差块 (256通道)
  - 最终输出层 (256→235通道)
- **特性**: 
  - 残差连接防止梯度消失
  - 批归一化加速收敛
  - 无偏置卷积减少参数

### 特征表示

观察空间维度：`4 × 9 = 36`个基础特征位置
- **手牌表示**: 万、条、筒、字牌的数量和状态
- **公共信息**: 已出牌、其他玩家动作历史
- **游戏状态**: 当前轮次、可执行动作等

## 动作空间

总计235个可能动作：
- **出牌**: 34种不同牌型
- **吃牌**: 多种组合方式
- **碰牌**: 三张相同牌
- **杠牌**: 明杠、暗杠、补杠
- **胡牌**: 各种胡牌方式
- **过牌**: 放弃当前动作机会

## 训练可视化

项目包含完整的训练可视化系统：

- **训练曲线**: 损失值和准确率变化
- **损失分布**: 训练过程损失分布直方图
- **综合仪表板**: 多维度训练指标展示

可视化文件保存在 `log/visualizations/` 目录下。

## 使用示例

### 基本使用

```python
from model import CNNModel
from feature import FeatureAgent

# 初始化模型和智能体
model = CNNModel()
agent = FeatureAgent(model)

# 处理游戏请求
response = agent.action2response(action_id, request)
```

### 训练自定义模型

```python
from supervised import train_model
from dataset import MahjongGBDataset

# 加载数据集
dataset = MahjongGBDataset('data.npz')

# 开始训练
train_model(dataset, epochs=100)
```

## 配置说明

### 关键参数

- `OBS_SIZE = 38`: 观察空间维度
- `ACT_SIZE = 235`: 动作空间维度
- `RESIDUAL_BLOCKS = 20`: 残差块数量
- `CHANNELS = 256`: 卷积通道数

### 训练参数

- 学习率: 0.001 (Adam优化器)
- 批次大小: 可配置
- 训练轮数: 可配置
- 数据增强: 支持旋转变换

## 重要文件说明

| 文件 | 功能描述 |
|------|----------|
| `__main__.py` | Botzone平台主程序入口 |
| `model.py` | CNN模型定义和架构 |
| `feature.py` | 特征工程和动作空间映射 |
| `agent.py` | 智能体抽象基类 |
| `dataset.py` | 数据集加载和预处理 |
| `supervised.py` | 监督学习训练主程序 |
| `visualizer.py` | 训练过程可视化工具 |
| `preprocess.py` | 原始数据预处理脚本 |

## 性能表现

- **模型大小**: 约20M参数
- **训练速度**: 取决于硬件配置
- **准确率**: 在验证集上达到较高准确率
