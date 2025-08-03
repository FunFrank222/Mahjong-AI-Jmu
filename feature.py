# 麻将AI特征提取模块
# 功能：处理游戏状态，提取特征向量，管理动作空间

from agent import MahjongGBAgent
from MahjongGB import MahjongFanCalculator
import numpy as np
from collections import defaultdict

class FeatureAgent(MahjongGBAgent):
    """麻将特征提取智能体
    
    继承自MahjongGBAgent，负责：
    1. 将游戏状态转换为神经网络可处理的特征向量
    2. 管理动作空间和动作掩码
    3. 处理各种游戏请求和响应
    4. 检查胡牌条件
    """
    
    # 观察空间大小：4个玩家 × 9种特征类型 × 36种牌型
    OBS_SIZE = 4 * 9
    # 动作空间大小：包含所有可能的麻将动作
    ACT_SIZE = 235
    
    # 观察特征的偏移量映射
    OFFSET_OBS = {
        'SEAT_WIND': 0,      # 座位风向（东南西北）
        'PREVALENT_WIND': 1,  # 场风
        'HAND': 2,           # 手牌（最多14张）
        'HALF_FLUSH': 6,     # 各玩家的明牌组合（4个玩家×4层）
        'DISCARD': 22,       # 各玩家的弃牌历史（4个玩家×4层）
    }
    
    # 动作类型的偏移量映射
    OFFSET_ACT = {
        'Pass': 0,      # 过牌
        'Hu': 1,        # 胡牌
        'Play': 2,      # 出牌（36种牌）
        'Chi': 38,      # 吃牌（63种组合）
        'Peng': 101,    # 碰牌（34种牌）
        'Gang': 135,    # 明杠（34种牌）
        'AnGang': 169,  # 暗杠（34种牌）
        'BuGang': 203,  # 补杠（34种牌）
    }
    
    # 牌型到索引的映射
    OFFSET_TILE = {
        # 万字牌：1万-9万
        'W1': 0, 'W2': 1, 'W3': 2, 'W4': 3, 'W5': 4, 'W6': 5, 'W7': 6, 'W8': 7, 'W9': 8,
        # 条字牌：1条-9条  
        'T1': 9, 'T2': 10, 'T3': 11, 'T4': 12, 'T5': 13, 'T6': 14, 'T7': 15, 'T8': 16, 'T9': 17,
        # 饼字牌：1饼-9饼
        'B1': 18, 'B2': 19, 'B3': 20, 'B4': 21, 'B5': 22, 'B6': 23, 'B7': 24, 'B8': 25, 'B9': 26,
        # 风牌：东南西北
        'F1': 27, 'F2': 28, 'F3': 29, 'F4': 30,
        # 箭牌：中发白
        'J1': 31, 'J2': 32, 'J3': 33
    }
    
    # 所有牌型列表（用于动作解码）
    TILE_LIST = ['W1', 'W2', 'W3', 'W4', 'W5', 'W6', 'W7', 'W8', 'W9',
                 'T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 
                 'B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9',
                 'F1', 'F2', 'F3', 'F4', 'J1', 'J2', 'J3']
    
    def __init__(self, seatWind):
        """初始化特征提取智能体
        
        Args:
            seatWind (int): 座位风向（0-3分别代表东南西北）
        """
        super().__init__(seatWind)
        self.seatWind = seatWind  # 座位风向
        self.prevalentWind = 0    # 场风
        self.hand = []            # 手牌列表
        self.valid = []           # 当前可执行的动作列表
        self.curTile = None       # 当前牌
        self.tileFrom = 0         # 牌的来源玩家
        
        # 各玩家的牌组（吃、碰、杠）
        self.packs = [[] for i in range(4)]
        # 各玩家的出牌历史
        self.history = [[] for i in range(4)]
        # 各玩家剩余牌墙数量
        self.tileWall = [21] * 4
        # 已显示的牌统计
        self.shownTiles = defaultdict(int)
        # 是否为牌墙最后一张
        self.wallLast = False
        # 是否与杠相关
        self.isAboutKong = False
        
        # 初始化观察特征矩阵（36维特征 × 36种牌型）
        self.obs = np.zeros((self.OBS_SIZE, 36))
        # 设置座位风向特征
        self.obs[self.OFFSET_OBS['SEAT_WIND']][self.OFFSET_TILE['F%d' % (self.seatWind + 1)]] = 1
    
    def request2obs(self, request):
        """将游戏请求转换为观察特征
        
        处理各种游戏事件：
        - Wind: 设置场风
        - Deal: 发牌
        - Draw: 摸牌
        - Play: 出牌
        - Chi/Peng/Gang: 吃碰杠操作
        - Hu: 胡牌
        - Huang: 流局
        
        Args:
            request (str): 游戏请求字符串
            
        Returns:
            dict: 包含观察特征和动作掩码的字典，如果不需要决策则返回None
        """
        t = request.split()
        
        if t[0] == 'Wind':
            # 设置场风
            self.prevalentWind = int(t[1])
            self.obs[self.OFFSET_OBS['PREVALENT_WIND']][self.OFFSET_TILE['F%d' % (self.prevalentWind + 1)]] = 1
            return
            
        if t[0] == 'Deal':
            # 发牌阶段，获取初始手牌
            self.hand = t[1:]
            self._hand_embedding_update()
            return
            
        if t[0] == 'Huang':
            # 流局，清空可执行动作
            self.valid = []
            return self._obs()
            
        if t[0] == 'Draw':
            # 摸牌阶段，需要决策出牌或胡牌
            self.tileWall[0] -= 1  # 减少自己的牌墙数量
            self.wallLast = self.tileWall[1] == 0  # 检查是否为最后一张
            tile = t[1]
            self.valid = []
            
            # 检查是否可以胡牌（自摸）
            if self._check_mahjong(tile, isSelfDrawn=True, isAboutKong=self.isAboutKong):
                self.valid.append(self.OFFSET_ACT['Hu'])
            
            self.isAboutKong = False
            self.hand.append(tile)
            self._hand_embedding_update()
            
            # 添加所有可能的出牌动作
            for tile in set(self.hand):
                self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                # 检查是否可以暗杠
                if self.hand.count(tile) == 4 and not self.wallLast and self.tileWall[0] > 0:
                    self.valid.append(self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[tile])
            
            # 检查是否可以补杠
            if not self.wallLast and self.tileWall[0] > 0:
                for packType, tile, offer in self.packs[0]:
                    if packType == 'PENG' and tile in self.hand:
                        self.valid.append(self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[tile])
            
            return self._obs()
        
        # 处理其他玩家的动作
        p = (int(t[1]) + 4 - self.seatWind) % 4  # 转换为相对座位号
        
        if t[2] == 'Draw':
            # 其他玩家摸牌
            self.tileWall[p] -= 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            return
            
        if t[2] in ['Invalid', 'Hu']:
            # 无效动作或胡牌，清空可执行动作
            self.valid = []
            return self._obs()
            
        if t[2] == 'Play':
            # 出牌动作
            self.tileFrom = p
            self.curTile = t[3]
            self.shownTiles[self.curTile] += 1
            self.history[p].append(self.curTile)
            
            if p == 0:
                # 自己出牌，从手牌中移除
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
                return
            else:
                # 其他玩家出牌，检查可执行的动作
                self.valid = []
                
                # 检查是否可以胡牌（点炮）
                if self._check_mahjong(self.curTile):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                
                if not self.wallLast:
                    # 检查是否可以碰牌或明杠
                    if self.hand.count(self.curTile) >= 2:
                        self.valid.append(self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[self.curTile])
                        if self.hand.count(self.curTile) == 3 and self.tileWall[0]:
                            self.valid.append(self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[self.curTile])
                    
                    # 检查是否可以吃牌（只能吃上家的牌）
                    color = self.curTile[0]
                    if p == 3 and color in 'WTB':  # 只有数字牌可以吃
                        num = int(self.curTile[1])
                        tmp = []
                        for i in range(-2, 3): 
                            tmp.append(color + str(num + i))
                        
                        # 检查三种吃牌组合
                        if tmp[0] in self.hand and tmp[1] in self.hand:  # 吃成123
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 3) * 3 + 2)
                        if tmp[1] in self.hand and tmp[3] in self.hand:  # 吃成234
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 2) * 3 + 1)
                        if tmp[3] in self.hand and tmp[4] in self.hand:  # 吃成345
                            self.valid.append(self.OFFSET_ACT['Chi'] + 'WTB'.index(color) * 21 + (num - 1) * 3)
                
                # 总是可以选择过牌
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        
        # 处理吃碰杠等操作...
        if t[2] == 'Chi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].append(('CHI', tile, int(self.curTile[1]) - num + 2))
            self.shownTiles[self.curTile] -= 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] += 1
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                self.hand.append(self.curTile)
                for i in range(-1, 2):
                    self.hand.remove(color + str(num + i))
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == 'UnChi':
            tile = t[3]
            color = tile[0]
            num = int(tile[1])
            self.packs[p].pop()
            self.shownTiles[self.curTile] += 1
            for i in range(-1, 2):
                self.shownTiles[color + str(num + i)] -= 1
            if p == 0:
                for i in range(-1, 2):
                    self.hand.append(color + str(num + i))
                self.hand.remove(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'Peng':
            self.packs[p].append(('PENG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 2
            self.wallLast = self.tileWall[(p + 1) % 4] == 0
            if p == 0:
                # Available: Play
                self.valid = []
                for i in range(2):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                for tile in set(self.hand):
                    self.valid.append(self.OFFSET_ACT['Play'] + self.OFFSET_TILE[tile])
                return self._obs()
            else:
                return
        if t[2] == 'UnPeng':
            self.packs[p].pop()
            self.shownTiles[self.curTile] -= 2
            if p == 0:
                for i in range(2):
                    self.hand.append(self.curTile)
                self._hand_embedding_update()
            return
        if t[2] == 'Gang':
            self.packs[p].append(('GANG', self.curTile, (4 + p - self.tileFrom) % 4))
            self.shownTiles[self.curTile] += 3
            if p == 0:
                for i in range(3):
                    self.hand.remove(self.curTile)
                self._hand_embedding_update()
                self.isAboutKong = True
            return
        if t[2] == 'AnGang':
            tile = 'CONCEALED' if p else t[3]
            self.packs[p].append(('GANG', tile, 0))
            if p == 0:
                self.isAboutKong = True
                for i in range(4):
                    self.hand.remove(tile)
            else:
                self.isAboutKong = False
            return
        if t[2] == 'BuGang':
            tile = t[3]
            for i in range(len(self.packs[p])):
                if tile == self.packs[p][i][1]:
                    self.packs[p][i] = ('GANG', tile, self.packs[p][i][2])
                    break
            self.shownTiles[tile] += 1
            if p == 0:
                self.hand.remove(tile)
                self._hand_embedding_update()
                self.isAboutKong = True
                return
            else:
                # Available: Hu/Pass
                self.valid = []
                if self._check_mahjong(tile, isSelfDrawn = False, isAboutKong = True):
                    self.valid.append(self.OFFSET_ACT['Hu'])
                self.valid.append(self.OFFSET_ACT['Pass'])
                return self._obs()
        raise NotImplementedError('Unknown request %s!' % request)
    
    def action2response(self, action):
        """将动作编号转换为游戏响应字符串
        
        Args:
            action (int): 动作编号
            
        Returns:
            str: 游戏响应字符串
        """
        if action < self.OFFSET_ACT['Hu']:
            return 'Pass'
        if action < self.OFFSET_ACT['Play']:
            return 'Hu'
        if action < self.OFFSET_ACT['Chi']:
            return 'Play ' + self.TILE_LIST[action - self.OFFSET_ACT['Play']]
        if action < self.OFFSET_ACT['Peng']:
            t = (action - self.OFFSET_ACT['Chi']) // 3
            return 'Chi ' + 'WTB'[t // 7] + str(t % 7 + 2)
        if action < self.OFFSET_ACT['Gang']:
            return 'Peng'
        if action < self.OFFSET_ACT['AnGang']:
            return 'Gang'
        if action < self.OFFSET_ACT['BuGang']:
            return 'Gang ' + self.TILE_LIST[action - self.OFFSET_ACT['AnGang']]
        return 'BuGang ' + self.TILE_LIST[action - self.OFFSET_ACT['BuGang']]
    
    def response2action(self, response):
        """将游戏响应字符串转换为动作编号
        
        Args:
            response (str): 游戏响应字符串
            
        Returns:
            int: 动作编号
        """
        t = response.split()
        if t[0] == 'Pass': return self.OFFSET_ACT['Pass']
        if t[0] == 'Hu': return self.OFFSET_ACT['Hu']
        if t[0] == 'Play': return self.OFFSET_ACT['Play'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Chi': return self.OFFSET_ACT['Chi'] + 'WTB'.index(t[1][0]) * 7 * 3 + (int(t[2][1]) - 2) * 3 + int(t[1][1]) - int(t[2][1]) + 1
        if t[0] == 'Peng': return self.OFFSET_ACT['Peng'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'Gang': return self.OFFSET_ACT['Gang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'AnGang': return self.OFFSET_ACT['AnGang'] + self.OFFSET_TILE[t[1]]
        if t[0] == 'BuGang': return self.OFFSET_ACT['BuGang'] + self.OFFSET_TILE[t[1]]
        return self.OFFSET_ACT['Pass']
    
    def _obs(self):
        """生成当前状态的观察特征和动作掩码
        
        Returns:
            dict: 包含observation（特征矩阵）和action_mask（动作掩码）的字典
        """
        # 创建动作掩码，只有有效动作对应位置为1
        mask = np.zeros(self.ACT_SIZE)
        for a in self.valid:
            mask[a] = 1
        
        return {
            'observation': self.obs.reshape((self.OBS_SIZE, 4, 9)).copy(),  # 重塑为4×9的特征矩阵
            'action_mask': mask
        }
    
    def _hand_embedding_update(self):
        """更新手牌、明牌组合和弃牌历史的特征嵌入"""
        # 清空手牌相关的特征
        self.obs[self.OFFSET_OBS['HAND']:] = 0
        
        # 更新手牌特征
        d = defaultdict(int)
        for tile in self.hand:
            d[tile] += 1
        for tile in d:
            self.obs[self.OFFSET_OBS['HAND']:self.OFFSET_OBS['HAND'] + d[tile], self.OFFSET_TILE[tile]] = 1
        
        # 更新各玩家的明牌组合特征
        packs = self.packs
        packs = [sum([
            [tri[1], tri[1][:1] + str(int(tri[1][1:]) - 1), tri[1][:1] + str(int(tri[1][1:]) + 1)] if tri[0] == 'CHI' else
            [tri[1]] * 3 if tri[0] == 'PENG' else
            [tri[1]] * 4 if tri[1] != 'CONCEALED' else []
            for tri in packs[i]
        ], list()) for i in range(4)]
        
        for i, tile_list in enumerate(packs):
            d = defaultdict(int)
            for tile in tile_list:
                d[tile] += 1
            for tile in d:
                self.obs[self.OFFSET_OBS['HALF_FLUSH'] + 4 * i:self.OFFSET_OBS['HALF_FLUSH'] + 4 * i + d[tile], self.OFFSET_TILE[tile]] = 1
        
        # 更新各玩家的弃牌历史特征
        for i, tile_list in enumerate(self.history):
            d = defaultdict(int)
            for tile in tile_list:
                d[tile] += 1
            for tile in d:
                self.obs[self.OFFSET_OBS['DISCARD'] + 4 * i:self.OFFSET_OBS['DISCARD'] + 4 * i + d[tile], self.OFFSET_TILE[tile]] = 1
    
    def _check_mahjong(self, winTile, isSelfDrawn=False, isAboutKong=False):
        """检查是否可以胡牌
        
        Args:
            winTile (str): 胡牌的牌
            isSelfDrawn (bool): 是否为自摸
            isAboutKong (bool): 是否与杠相关
            
        Returns:
            bool: 是否可以胡牌
        """
        try:
            # 使用麻将番型计算器检查胡牌条件
            fans = MahjongFanCalculator(
                pack=tuple(self.packs[0]),      # 明牌组合
                hand=tuple(self.hand),          # 手牌
                winTile=winTile,                # 胡牌的牌
                flowerCount=0,                  # 花牌数量
                isSelfDrawn=isSelfDrawn,        # 是否自摸
                is4thTile=self.shownTiles[winTile] == 4,  # 是否为第四张
                isAboutKong=isAboutKong,        # 是否与杠相关
                isWallLast=self.wallLast,       # 是否为最后一张
                seatWind=self.seatWind,         # 座位风
                prevalentWind=self.prevalentWind, # 场风
                verbose=True
            )
            
            # 计算总番数
            fanCnt = 0
            for fanPoint, cnt, fanName, fanNameEn in fans:
                fanCnt += fanPoint * cnt
            
            # 至少需要8番才能胡牌
            if fanCnt < 8: 
                raise Exception('Not Enough Fans')
        except:
            return False
        return True