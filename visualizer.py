# 麻将AI训练可视化模块
# 功能：记录训练过程数据，生成各种训练图表和仪表板

import matplotlib.pyplot as plt
import numpy as np
import json
import torch
from collections import defaultdict
import seaborn as sns
from matplotlib.animation import FuncAnimation
import os

class MahjongTrainingVisualizer:
    """麻将AI训练可视化器
    
    主要功能：
    1. 记录训练过程中的损失值和准确率
    2. 生成训练曲线图
    3. 创建损失分布直方图
    4. 生成综合训练仪表板
    5. 保存和加载训练日志
    """
    
    def __init__(self, log_dir='log/'):
        """初始化可视化器
        
        Args:
            log_dir (str): 日志文件保存目录
        """
        self.log_dir = log_dir  # 日志保存目录
        self.train_losses = []  # 每个epoch的平均训练损失
        self.val_accuracies = []  # 每个epoch的验证准确率
        self.epochs = []  # epoch编号列表
        self.iterations = []  # 迭代次数列表
        self.iteration_losses = []  # 每次迭代的损失值
        
        # 设置matplotlib中文字体支持
        plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'DejaVu Sans']
        plt.rcParams['axes.unicode_minus'] = False
        # 忽略字体相关警告
        import warnings
        warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')
        
        # 创建可视化文件保存目录
        os.makedirs(os.path.join(log_dir, 'visualizations'), exist_ok=True)
    
    def log_training_step(self, epoch, iteration, loss):
        """记录单次训练步骤的数据
        
        Args:
            epoch (int): 当前epoch
            iteration (int): 当前迭代次数
            loss (float): 当前步骤的损失值
        """
        self.iterations.append(len(self.iteration_losses))
        self.iteration_losses.append(loss)
    
    def log_epoch(self, epoch, val_accuracy):
        """记录每个epoch结束时的数据
        
        Args:
            epoch (int): epoch编号
            val_accuracy (float): 验证集准确率
        """
        self.epochs.append(epoch)
        self.val_accuracies.append(val_accuracy)
        
        # 计算当前epoch的平均训练损失
        if self.iteration_losses:
            # 计算当前epoch对应的损失值范围
            epoch_start = len(self.train_losses) * len(self.iteration_losses) // max(1, len(self.epochs)-1) if len(self.epochs) > 1 else 0
            epoch_losses = self.iteration_losses[epoch_start:]
            if epoch_losses:
                avg_loss = np.mean(epoch_losses)
                self.train_losses.append(avg_loss)
    
    def plot_training_curves(self, save=True, show=False):
        """绘制训练损失和验证准确率曲线
        
        Args:
            save (bool): 是否保存图片
            show (bool): 是否显示图片
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # 左图：训练损失曲线
        if self.train_losses:
            ax1.plot(self.epochs[:len(self.train_losses)], self.train_losses, 'b-', label='训练损失', linewidth=2)
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('损失值')
            ax1.set_title('训练损失曲线')
            ax1.grid(True, alpha=0.3)
            ax1.legend()
        
        # 右图：验证准确率曲线
        if self.val_accuracies:
            ax2.plot(self.epochs, self.val_accuracies, 'r-', label='验证准确率', linewidth=2)
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('准确率')
            ax2.set_title('验证准确率曲线')
            ax2.grid(True, alpha=0.3)
            ax2.legend()
            ax2.set_ylim(0, 1)  # 准确率范围0-1
        
        plt.tight_layout()
        
        if save:
            plt.savefig(os.path.join(self.log_dir, 'visualizations', 'training_curves.png'), dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    def plot_loss_detail(self, save=True, show=False):
        """绘制详细的损失变化图（包含移动平均和趋势线）
        
        Args:
            save (bool): 是否保存图片
            show (bool): 是否显示图片
        """
        if not self.iteration_losses:
            return
            
        plt.figure(figsize=(12, 6))
        
        # 当数据点较多时，使用移动平均平滑曲线
        if len(self.iteration_losses) > 50:
            window = min(50, len(self.iteration_losses) // 10)  # 动态调整窗口大小
            moving_avg = np.convolve(self.iteration_losses, np.ones(window)/window, mode='valid')
            plt.plot(range(window-1, len(self.iteration_losses)), moving_avg, 'b-', linewidth=2, label=f'移动平均({window})')
            
            # 添加更平滑的趋势线
            if len(moving_avg) > 100:
                smooth_window = len(moving_avg) // 20
                trend = np.convolve(moving_avg, np.ones(smooth_window)/smooth_window, mode='valid')
                plt.plot(range(window-1+smooth_window-1, len(self.iteration_losses)), trend, 'r-', linewidth=3, label='趋势线')
        else:
            # 数据点较少时，显示原始数据和简单移动平均
            plt.plot(self.iterations, self.iteration_losses, alpha=0.3, linewidth=1, color='gray', label='原始数据')
            if len(self.iteration_losses) > 10:
                window = max(3, len(self.iteration_losses) // 5)
                moving_avg = np.convolve(self.iteration_losses, np.ones(window)/window, mode='valid')
                plt.plot(range(window-1, len(self.iteration_losses)), moving_avg, 'b-', linewidth=2, label=f'移动平均({window})')
        
        plt.legend()
        plt.xlabel('迭代次数')
        plt.ylabel('损失值')
        plt.title('训练损失详细变化')
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(os.path.join(self.log_dir, 'visualizations', 'loss_detail.png'), dpi=300, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close()
    
    def create_training_dashboard(self):
        """创建综合训练仪表板
        
        包含6个子图：
        1. 训练损失曲线（平滑后）
        2. 验证准确率曲线
        3. 训练状态信息文本
        4. 损失分布直方图
        5. 准确率变化趋势
        6. 学习进度图
        """
        fig = plt.figure(figsize=(16, 10))
        
        # 创建3x3的子图布局
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
        
        # 子图1：训练损失曲线（占据前两列）
        ax1 = fig.add_subplot(gs[0, :2])
        if self.iteration_losses:
            if len(self.iteration_losses) > 20:
                window = min(20, len(self.iteration_losses) // 5)
                moving_avg = np.convolve(self.iteration_losses, np.ones(window)/window, mode='valid')
                ax1.plot(range(window-1, len(self.iteration_losses)), moving_avg, 'b-', linewidth=2, label='训练损失(平滑)')
            else:
                ax1.plot(self.iterations, self.iteration_losses, 'b-', linewidth=2, label='训练损失')
            ax1.legend()
        ax1.set_title('训练损失')
        ax1.grid(True, alpha=0.3)
        
        # 子图2：验证准确率
        ax2 = fig.add_subplot(gs[0, 2])
        if self.val_accuracies:
            ax2.plot(self.epochs, self.val_accuracies, 'g-', linewidth=2, marker='o')
        ax2.set_title('验证准确率')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        # 子图3：训练状态信息（占据整个第二行）
        ax3 = fig.add_subplot(gs[1, :])
        ax3.axis('off')  # 不显示坐标轴
        
        # 计算关键指标
        latest_val_acc = f"{self.val_accuracies[-1]:.4f}" if self.val_accuracies else 'N/A'
        latest_loss = f"{self.iteration_losses[-1]:.4f}" if self.iteration_losses else 'N/A'
        best_val_acc = f"{max(self.val_accuracies):.4f}" if self.val_accuracies else 'N/A'
        
        # 创建状态信息文本
        status_text = f"""
        训练状态总览:
        • 当前Epoch: {len(self.epochs)}
        • 总迭代次数: {len(self.iteration_losses)}
        • 最新验证准确率: {latest_val_acc}
        • 最新训练损失: {latest_loss}
        • 最佳验证准确率: {best_val_acc}
        """
        
        ax3.text(0.1, 0.5, status_text, fontsize=12, verticalalignment='center',
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.5))
        
        # 子图4：损失分布直方图
        ax4 = fig.add_subplot(gs[2, 0])
        if self.iteration_losses:
            ax4.hist(self.iteration_losses, bins=30, alpha=0.7, edgecolor='black')
        ax4.set_title('损失分布')
        ax4.grid(True, alpha=0.3)
        
        # 子图5：准确率变化趋势
        ax5 = fig.add_subplot(gs[2, 1])
        if len(self.val_accuracies) > 1:
            improvements = np.diff(self.val_accuracies)  # 计算相邻epoch间的准确率变化
            ax5.bar(range(len(improvements)), improvements, 
                   color=['green' if x > 0 else 'red' for x in improvements])
        ax5.set_title('准确率变化')
        ax5.grid(True, alpha=0.3)
        
        # 子图6：学习进度
        ax6 = fig.add_subplot(gs[2, 2])
        if self.val_accuracies:
            progress = np.array(self.val_accuracies)
            ax6.fill_between(self.epochs, 0, progress, alpha=0.3)  # 填充区域
            ax6.plot(self.epochs, progress, linewidth=2)
        ax6.set_title('学习进度')
        ax6.set_ylim(0, 1)
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('麻将AI训练仪表板', fontsize=16, fontweight='bold')
        plt.savefig(os.path.join(self.log_dir, 'visualizations', 'training_dashboard.png'), dpi=300, bbox_inches='tight')
        plt.close()  # 直接关闭，不显示
    
    def save_training_log(self):
        """将训练数据保存为JSON格式"""
        log_data = {
            'epochs': self.epochs,
            'train_losses': self.train_losses,
            'val_accuracies': self.val_accuracies,
            'iteration_losses': self.iteration_losses,
            'iterations': self.iterations
        }
        
        with open(os.path.join(self.log_dir, 'training_log.json'), 'w') as f:
            json.dump(log_data, f, indent=2)
    
    def load_training_log(self):
        """从JSON文件加载训练数据"""
        try:
            with open(os.path.join(self.log_dir, 'training_log.json'), 'r') as f:
                log_data = json.load(f)
            
            self.epochs = log_data.get('epochs', [])
            self.train_losses = log_data.get('train_losses', [])
            self.val_accuracies = log_data.get('val_accuracies', [])
            self.iteration_losses = log_data.get('iteration_losses', [])
            self.iterations = log_data.get('iterations', [])
            
            print("成功加载训练日志")
        except FileNotFoundError:
            print("未找到训练日志文件，将创建新的日志")
        except Exception as e:
            print(f"加载训练日志时出错: {e}")