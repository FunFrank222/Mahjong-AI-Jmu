from visualizer import MahjongTrainingVisualizer
from model import CNNModel
import torch
import os

def analyze_existing_training(log_dir='log/'):
    """分析现有的训练结果"""
    visualizer = MahjongTrainingVisualizer(log_dir)
    
    # 尝试加载训练日志
    visualizer.load_training_log()
    
    # 如果有数据，生成可视化
    if visualizer.epochs or visualizer.iteration_losses:
        print("正在生成训练分析报告...")
        visualizer.plot_training_curves()
        visualizer.plot_loss_detail()
        visualizer.create_training_dashboard()
    else:
        print("未找到训练数据，正在分析数据集和模型...")
    
    # 分析数据集
    visualizer.analyze_dataset()
    
    # 分析模型（如果有保存的模型）
    try:
        model = CNNModel()
        checkpoint_dir = os.path.join(log_dir, 'checkpoint')
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.pkl')]
            if checkpoints:
                latest_checkpoint = max(checkpoints, key=lambda x: int(x.split('.')[0]))
                model.load_state_dict(torch.load(os.path.join(checkpoint_dir, latest_checkpoint), map_location='cpu'))
                print(f"加载模型检查点: {latest_checkpoint}")
                visualizer.plot_model_complexity(model)
    except Exception as e:
        print(f"分析模型时出错: {e}")
    
    print(f"分析完成，结果保存在 {log_dir}visualizations/ 目录")

if __name__ == '__main__':
    analyze_existing_training()