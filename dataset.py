# 麻将数据集加载器
# 负责加载和管理麻将专家对局数据

from torch.utils.data import Dataset
import numpy as np
from bisect import bisect_right

class MahjongGBDataset(Dataset):
    """
    麻将数据集类，用于加载专家对局数据进行监督学习
    
    数据格式:
    - 每个.npz文件包含一局游戏的所有决策点
    - obs: 观察数据 [N, 38, 4, 9] - 牌局状态特征
    - mask: 动作掩码 [N, 235] - 标识哪些动作是合法的
    - act: 专家动作 [N] - 专家在每个决策点选择的动作
    """
    
    def __init__(self, begin=0, end=1, augment=False):
        """
        初始化数据集
        
        Args:
            begin: 数据集起始比例 (0-1)
            end: 数据集结束比例 (0-1)
            augment: 是否启用数据增强
        """
        import json
        
        # 加载数据统计信息
        with open('data/count.json') as f:
            self.match_samples = json.load(f)  # 每局游戏的样本数量列表
        
        # 数据集基本信息
        self.total_matches = len(self.match_samples)  # 总局数
        self.total_samples = sum(self.match_samples)  # 总样本数
        
        # 根据比例确定使用的数据范围
        self.begin = int(begin * self.total_matches)
        self.end = int(end * self.total_matches)
        
        # 截取指定范围的数据
        self.match_samples = self.match_samples[self.begin:self.end]
        self.matches = len(self.match_samples)  # 当前数据集的局数
        self.samples = sum(self.match_samples)  # 当前数据集的样本数
        
        self.augment = augment  # 数据增强标志
        
        # 构建累积样本索引，用于快速定位样本属于哪一局
        # 例如：[0, 100, 250, 400] 表示第0局有100个样本，第1局有150个样本等
        t = 0
        for i in range(self.matches):
            a = self.match_samples[i]
            self.match_samples[i] = t
            t += a
        
        # 预加载所有数据到内存，提高训练速度
        self.cache = {'obs': [], 'mask': [], 'act': []}
        
        print(f"正在加载 {self.matches} 局游戏数据...")
        for i in range(self.matches):
            if i % 128 == 0: 
                print(f'loading {i}/{self.matches}')
            
            # 加载第i局的数据文件
            d = np.load('data/%d.npz' % (i + self.begin))
            
            # 将数据添加到缓存中
            for k in d:
                self.cache[k].append(d[k])
        
        print(f"数据加载完成！共 {self.samples} 个训练样本")
    
    def __len__(self):
        """返回数据集大小"""
        return self.samples
    
    def __getitem__(self, index):
        """
        获取指定索引的数据样本
        
        Args:
            index: 样本索引
            
        Returns:
            tuple: (观察数据, 动作掩码, 专家动作)
        """
        # 使用二分查找快速定位样本属于哪一局
        match_id = bisect_right(self.match_samples, index, 0, self.matches) - 1
        
        # 计算样本在该局中的相对索引
        sample_id = index - self.match_samples[match_id]
        
        # 返回对应的数据
        return (
            self.cache['obs'][match_id][sample_id],   # 观察数据 [38, 4, 9]
            self.cache['mask'][match_id][sample_id],  # 动作掩码 [235]
            self.cache['act'][match_id][sample_id]    # 专家动作标签
        )