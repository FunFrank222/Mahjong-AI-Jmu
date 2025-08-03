# 麻将智能体基础接口
# 定义了麻将AI需要实现的基本方法

class MahjongGBAgent:
    """麻将智能体基础类
    
    这是一个抽象基类，定义了麻将AI需要实现的核心接口：
    1. request2obs: 将游戏请求转换为观察状态
    2. action2response: 将动作转换为游戏响应
    
    所有具体的麻将AI实现都应该继承这个类并实现这些方法。
    """
    
    def __init__(self, seatWind):
        """初始化智能体
        
        Args:
            seatWind (int): 座位风向（0-3分别代表东南西北）
        """
        pass
    
    def request2obs(self, request):
        """将游戏请求转换为观察状态
        
        这个方法需要处理各种游戏事件，包括：
        - Wind 0..3: 设置场风
        - Deal XX XX ...: 发牌
        - Player N Draw: 玩家摸牌
        - Player N Gang: 玩家杠牌
        - Player N(me) Play XX: 自己出牌
        - Player N(me) BuGang XX: 自己补杠
        - Player N(not me) Peng: 其他玩家碰牌
        - Player N(not me) Chi XX: 其他玩家吃牌
        - Player N(me) UnPeng: 撤销碰牌
        - Player N(me) UnChi XX: 撤销吃牌
        - Player N Hu: 玩家胡牌
        - Huang: 流局
        - Player N Invalid: 无效动作
        - Draw XX: 摸牌
        - Player N(not me) Play XX: 其他玩家出牌
        - Player N(not me) BuGang XX: 其他玩家补杠
        - Player N(me) Peng: 自己碰牌
        - Player N(me) Chi XX: 自己吃牌
        
        Args:
            request (str): 游戏请求字符串
            
        Returns:
            观察状态（具体格式由子类定义）
        """
        pass
    
    def action2response(self, action):
        """将动作转换为游戏响应
        
        需要支持的动作类型：
        - Hu: 胡牌
        - Play XX: 出牌
        - (An)Gang XX: 暗杠
        - BuGang XX: 补杠
        - Gang: 明杠
        - Peng: 碰牌
        - Chi XX: 吃牌
        - Pass: 过牌
        
        Args:
            action: 动作（具体格式由子类定义）
            
        Returns:
            str: 游戏响应字符串
        """
        pass