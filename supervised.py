# 麻将AI监督学习训练脚本
# 使用专家数据训练CNN模型学习麻将决策

from dataset import MahjongGBDataset
from torch.utils.data import DataLoader
from model import CNNModel
import torch.nn.functional as F
import torch
from visualizer import MahjongTrainingVisualizer
import os
import shutil

if __name__ == '__main__':
    # 训练日志目录
    logdir = 'log/'
    
    # 清理旧的训练记录，确保全新开始
    training_log_path = os.path.join(logdir, 'training_log.json')
    visualizations_dir = os.path.join(logdir, 'visualizations')
    
    if os.path.exists(training_log_path):
        os.remove(training_log_path)
        print("已清理旧的训练日志")
    
    if os.path.exists(visualizations_dir):
        shutil.rmtree(visualizations_dir)
        print("已清理旧的可视化文件")
    
    # 初始化训练可视化器
    visualizer = MahjongTrainingVisualizer(logdir)
    print("开始全新的训练过程...")
    
    # 数据集加载和划分
    splitRatio = 0.9  # 90%用于训练，10%用于验证
    batchSize = 1024  # 批次大小，影响训练速度和内存使用
    
    # 创建训练和验证数据集
    trainDataset = MahjongGBDataset(0, splitRatio, True)  # 训练集，启用数据增强
    validateDataset = MahjongGBDataset(splitRatio, 1, False)  # 验证集，不使用数据增强
    
    # 创建数据加载器
    loader = DataLoader(dataset=trainDataset, batch_size=batchSize, shuffle=True)  # 训练时打乱数据
    vloader = DataLoader(dataset=validateDataset, batch_size=batchSize, shuffle=False)  # 验证时不打乱
    
    # 模型初始化
    model = CNNModel().to('cuda')  # 将模型移到GPU加速训练
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)  # Adam优化器，学习率5e-4
    
    # 最佳模型跟踪变量
    best_acc = 0.0      # 最佳验证准确率
    best_epoch = 0      # 最佳模型对应的epoch
    best_model_path = '' # 最佳模型文件路径
    
    print("开始训练100轮...")
    
    # 主训练循环：100个epoch
    for e in range(100):
        print(f'Epoch {e+1}/100')
        
        # 模型检查点保存路径
        model_path = logdir + 'checkpoint/%d.pkl' % e
        
        # 确保检查点目录存在
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        # 保存当前epoch的模型状态
        torch.save(model.state_dict(), model_path)
        
        # 训练阶段
        epoch_losses = []  # 记录当前epoch的所有损失值
        
        for i, d in enumerate(loader):
            # 构建输入字典
            input_dict = {
                'is_training': True,  # 训练模式
                'obs': {
                    'observation': d[0].cuda(),  # 牌局观察 [batch, 38, 4, 9]
                    'action_mask': d[1].cuda()   # 动作掩码 [batch, 235]
                }
            }
            
            # 前向传播：获取模型预测
            logits = model(input_dict)  # [batch, 235]
            
            # 计算交叉熵损失
            # d[2]是专家动作标签，logits是模型预测的动作概率分布
            loss = F.cross_entropy(logits, d[2].long().cuda())
            
            # 记录训练步骤数据用于可视化
            visualizer.log_training_step(e, i, loss.item())
            epoch_losses.append(loss.item())
            
            # 每128个batch打印一次进度
            if i % 128 == 0:
                print('Iteration %d/%d' % (i, len(trainDataset) // batchSize + 1), 
                      'policy_loss', loss.item())
            
            # 反向传播和参数更新
            optimizer.zero_grad()  # 清零梯度
            loss.backward()        # 计算梯度
            optimizer.step()       # 更新参数
        
        # 验证阶段
        print('Run validation:')
        correct = 0  # 正确预测的样本数
        
        # 在验证集上评估模型性能
        for i, d in enumerate(vloader):
            input_dict = {
                'is_training': False,  # 评估模式
                'obs': {
                    'observation': d[0].cuda(),
                    'action_mask': d[1].cuda()
                }
            }
            
            # 不计算梯度，节省内存和计算
            with torch.no_grad():
                logits = model(input_dict)
                pred = logits.argmax(dim=1)  # 获取预测的动作
                correct += torch.eq(pred, d[2].cuda()).sum().item()  # 统计正确预测数
        
        # 计算验证准确率
        acc = correct / len(validateDataset)
        print(f'Epoch {e + 1}, 验证准确率: {acc:.4f}')
        
        # 检查是否是历史最佳模型
        if acc > best_acc:
            best_acc = acc
            best_epoch = e
            best_model_path = model_path
            print(f"🏆 新的最佳模型！Epoch {e+1}, 验证准确率: {acc:.4f}")
        
        # 记录epoch数据用于可视化
        visualizer.log_epoch(e, acc)
        
        # 每10个epoch生成一次可视化图表
        if (e + 1) % 10 == 0:
            print(f"正在生成训练可视化... ({e+1}/100)")
            visualizer.plot_training_curves(show=False)
            visualizer.plot_loss_detail(show=False)
            visualizer.create_training_dashboard()
            visualizer.save_training_log()
    
    # 训练完成后的总结
    print("\n" + "="*60)
    print("🎯 训练完成！最佳模型信息：")
    print(f"📊 最佳Epoch: {best_epoch + 1}")
    print(f"🎯 最佳验证准确率: {best_acc:.4f}")
    print(f"📁 最佳模型文件: {best_model_path}")
    print("="*60)
    
    # 复制最佳模型到根目录，方便部署使用
    best_model_for_botzone = 'best_model.pkl'
    shutil.copy2(best_model_path, best_model_for_botzone)
    print(f"✅ 最佳模型已复制到: {best_model_for_botzone}")
    print(f"🚀 请使用 {best_model_for_botzone} 进行Botzone对决！")
    
    # 生成最终的训练报告和可视化
    print("\n正在生成最终可视化报告...")
    visualizer.plot_training_curves()
    visualizer.plot_loss_detail()
    visualizer.create_training_dashboard()
    visualizer.save_training_log()
    
    print(f"所有可视化图表已保存到 {logdir}visualizations/ 目录")
    print("🎉 训练完成！现在你将获得干净简洁的可视化图像！")