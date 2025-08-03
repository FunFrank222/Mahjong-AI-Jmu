# 麻将AI神经网络模型定义文件
# 使用深度残差卷积神经网络(ResNet)架构来学习麻将决策
import torch
from torch import nn
import torch.nn.functional as F

class CNNModel(nn.Module):
    """
    麻将AI的卷积神经网络模型
    
    网络架构:
    - 输入: 38通道的4x9特征图(代表麻将牌局状态)
    - 主干网络: 20个残差块，每个包含2个卷积层+批归一化+ReLU
    - 输出: 235维动作概率分布(对应所有可能的麻将动作)
    
    设计思路:
    1. 使用残差连接解决深度网络梯度消失问题
    2. 批归一化加速训练并提高稳定性
    3. 无偏置卷积减少参数量，依赖批归一化进行偏移
    """

    def __init__(self):
        nn.Module.__init__(self)
        
        # 第一层：特征提取和维度变换
        # 38→128→64: 先扩展特征维度再压缩，增强表达能力
        self._tower = nn.Sequential(
            # 输入38通道 → 128通道，3x3卷积核，步长1，填充1
            nn.Conv2d(38, 128, 3, 1, 1, bias=False),  # 特征扩展层
            # 128通道 → 64通道，降维但保持足够的特征表达能力
            nn.Conv2d(128, 64, 3, 1, 1, bias=False),   # 特征压缩层
            nn.BatchNorm2d(64),  # 批归一化，稳定训练
            nn.ReLU()            # 激活函数，引入非线性
        )
        
        # 残差块2-20：深度特征学习
        # 每个残差块包含：Conv2d → BatchNorm → ReLU → Conv2d → BatchNorm
        # 最后通过残差连接(skip connection)与输入相加
        
        # 残差块2：学习基础麻将模式
        self._tower2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),  # 保持64通道不变
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),  # 第二个卷积层
            nn.BatchNorm2d(64)  # 注意：最后没有ReLU，因为要与残差相加后再激活
        )

        # 残差块3：学习更复杂的牌型组合
        self._tower3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64)
        )

        # 残差块4-20：逐层学习更高级的麻将策略特征
        # 每一层都能学习到不同层次的抽象特征：
        # - 浅层：基本牌型识别(顺子、刻子等)
        # - 中层：牌型组合和听牌判断
        # - 深层：复杂策略和全局优化
        
        self._tower4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64)
        )
        
        # ... 继续定义tower5到tower20 ...
        # (为节省空间，这里省略重复的残差块定义)
        # 实际代码中包含完整的20个残差块
        
        # 最终输出层：将特征图转换为动作概率
        self._tower21 = nn.Sequential(
            nn.Flatten(),  # 将4x9x64的特征图展平为一维向量
            # 64*4*9 = 2304维特征 → 235维动作空间
            nn.Linear(64*4*9, 235),  # 全连接层，输出235个可能动作的logits
        )

        # 权重初始化：使用Kaiming初始化，适合ReLU激活函数
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)  # 防止梯度爆炸/消失

    def forward(self, input_dict):
        """
        前向传播函数
        
        Args:
            input_dict: 包含以下键的字典
                - 'is_training': 是否为训练模式
                - 'obs': 观察字典，包含:
                    - 'observation': 牌局状态特征 [batch, 38, 4, 9]
                    - 'action_mask': 动作掩码 [batch, 235]
        
        Returns:
            masked_logits: 经过动作掩码处理的动作概率分布 [batch, 235]
        """
        # 设置训练/评估模式，影响BatchNorm和Dropout的行为
        self.train(mode=input_dict.get("is_training", False))
        
        # 获取观察数据并转换为float类型
        obs = input_dict["obs"]["observation"].float()
        
        # 第一层特征提取
        action_logits = self._tower(obs)  # [batch, 64, 4, 9]
        
        # 残差块2：第一个残差连接
        action_logits2 = self._tower2(action_logits)
        output = F.relu(action_logits + action_logits2)  # 残差连接 + ReLU激活
        
        # 残差块3
        action_logits3 = self._tower3(output)
        output1 = F.relu(output + action_logits3)
        
        # 残差块4
        action_logits4 = self._tower4(output1)
        output2 = F.relu(output1 + action_logits4)

        # ... 继续所有残差块的前向传播 ...
        # (实际代码中包含完整的20个残差块前向传播)
        
        # 最终输出层：特征图 → 动作概率
        output19 = self._tower21(output18)  # [batch, 235]

        # 动作掩码处理：屏蔽无效动作
        action_mask = input_dict["obs"]["action_mask"].float()
        # 将掩码转换为对数形式，无效动作设为极小值(-1e38)
        inf_mask = torch.clamp(torch.log(action_mask), -1e38, 1e38)
        
        # 返回经过掩码处理的最终logits
        # 无效动作的概率会变得极小，有效动作保持原有的相对概率
        return output19 + inf_mask