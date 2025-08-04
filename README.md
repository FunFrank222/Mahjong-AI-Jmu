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

安装必要的第三方库（包括但不限于）：

```bash
pip install torch torchvision numpy matplotlib json
```
## 数据集准备

### 步骤1：下载数据集

从北大网盘下载数据集：

**下载链接**: https://disk.pku.edu.cn/anyshare/zh-cn/link/AA8CB7A57AFDCD48CAA7C749E04B5B6FAA?_tb=none&expires_at=2026-04-30T23%3A59%3A48%2B08%3A00&item_type=&password_required=false&title=data.zip&type=anonymous

### 步骤2：解压数据集

```bash
# 将下载的data.zip解压到项目根目录
unzip data.zip
# 确保解压后有data.txt文件
```

### 步骤3：数据预处理

```bash
# 运行数据预处理脚本
python preprocess.py
```

**预处理过程说明**：
- 读取原始游戏日志（data.txt）
- 提取游戏状态特征
- 转换为神经网络可用格式
- 生成训练用的.npz文件

## 模型训练详解

### 步骤1：理解模型架构

CNN模型包含：
- **输入层**: 38通道 × 4×9 特征图
- **特征提取**: 20个残差块
- **输出层**: 235维动作概率

### 步骤2：开始训练

```bash
# 启动监督学习训练
python supervised.py
```

**训练过程监控**：
- 实时显示损失值和准确率
- 自动保存最佳模型
- 生成训练可视化图表

### 步骤3：训练参数说明

```python
# 关键训练参数
learning_rate = 0.001    # 学习率
batch_size = 32          # 批次大小
epochs = 100             # 训练轮数
```

## 训练监控与分析

### 实时监控

训练过程中您将看到：

### 可视化分析

```bash
# 分析训练结果
python analyze_training.py
```

生成的可视化文件：
- `log/visualizations/training_curves.png` - 训练曲线
- `log/visualizations/loss_detail.png` - 损失详情
- `log/visualizations/training_dashboard.png` - 综合仪表板
- <img width="1593" height="1105" alt="image" src="https://github.com/user-attachments/assets/a461b0c3-dbcd-42c4-a157-d9011db8e655" />


## 模型部署

### Botzone平台部署

1. **准备部署文件**：
   ```bash
   # 打包必要文件
   zip -r submission.zip __main__.py agent.py model.py feature.py log/checkpoint/best_model.pkl
   ```

2. **上传到Botzone**：
   - 登录Botzone平台
   - 选择麻将比赛
   - 上传__main__.zip

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
