import torch
from torch_geometric.loader import DataLoader
from src.graph_builder import GNSSGraphDataset
from src.model import STGraphTransformer
from src import config
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import os
import shutil  # <--- 记得在文件最顶部的 import 区域加上这个，或者直接放在 main 里也行

def main():
    print("="*40)
    print("      GNSS 模型训练与评估系统 (Physics-Aware)       ")
    print("="*40)
    
    # === 0. [新增] 自动清理旧缓存 (懒人专用) ===
    # 这样你就不用每次手动去文件夹里删 .pt 文件了
    processed_dir = os.path.join(config.DATASET_DIR, 'processed')
    if os.path.exists(processed_dir):
        try:
            shutil.rmtree(processed_dir)
            print(f"🧹 已自动清理旧缓存: {processed_dir}")
        except Exception as e:
            print(f"⚠️ 清理缓存失败 (可能文件被占用): {e}")
    # ==========================================

    # 1. 加载数据集
    dataset = GNSSGraphDataset(root=config.DATASET_DIR)
    
    # 2. 智能切分 (根据构建时的 train_mask)
    train_data = [data for data in dataset if data.train_mask == True]
    test_data  = [data for data in dataset if data.train_mask == False]
    
    print(f"📊 数据集加载完毕:")
    print(f"  - 训练集 (Static+Dynamic Clean + ds4): {len(train_data)} 样本")
    print(f"  - 测试集 (Unknown Scenarios): {len(test_data)} 样本")
    
    if len(train_data) == 0:
        print("❌ 错误: 训练集为空！请检查 config.py 或 graph_builder.py。")
        return

    # 3. 创建 DataLoader
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 4. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  Training on: {device}")
    
    # 自动适配输入维度 (Dataset里现在是1维节点特征, 2维边特征)
    model = STGraphTransformer(in_channels=dataset.num_features, edge_dim=2).to(device)
    
    # === 修改 1: 加入 weight_decay 防止过拟合 ===
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss()
    
    # === 修改 2: 加入学习率调度器 ===
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # === 训练循环 ===
    print("\n🚀 开始训练...")
    train_losses = []
    test_accs = []
    
    best_acc = 0.0
    
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            # 必须传入 edge_attr
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # 简单测试
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data in test_loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
                total += data.y.size(0)
        
        acc = correct / total if total > 0 else 0
        test_accs.append(acc)
        
        # 更新学习率
        scheduler.step(avg_loss)
        
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, 'best_model.pth'))
            
        if epoch % 5 == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Test Acc: {acc*100:.2f}% | LR: {current_lr:.6f}")

    print(f"\n✅ 训练结束! 最佳测试准确率: {best_acc*100:.2f}%")

    # === 5. 结果可视化 ===
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(test_accs, label='Test Accuracy', color='orange')
    plt.title('Generalization Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(config.LOG_DIR, 'training_curves.png'))
    print("📈 训练曲线已保存至 logs/training_curves.png")

    # === 6. 深度评估 (分场景) ===
    print("\n🔍 正在生成深度评估报告...")
    model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, 'best_model.pth'), weights_only=True))
    model.eval()
    
    # 提取所有场景
    test_scenarios = sorted(list(set([d.scenario for d in test_data])))
    
    print("\n" + "="*50)
    print("📊 分场景详细性能评估 (Scenario-wise Evaluation)")
    print("="*50)
    print(f"{'Scenario':<15} | {'Samples':<8} | {'Acc (%)':<10} | {'Status'}")
    print("-" * 50)

    performance_dict = {}

    y_true_all = []
    y_pred_all = []

    for scenario in test_scenarios:
        scenario_data = [d for d in test_data if d.scenario == scenario]
        if len(scenario_data) == 0: continue

        loader = DataLoader(scenario_data, batch_size=config.BATCH_SIZE, shuffle=False)
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data in loader:
                data = data.to(device)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                pred = out.argmax(dim=1)
                
                correct += int((pred == data.y).sum())
                total += data.y.size(0)
                
                y_true_all.extend(data.y.cpu().numpy())
                y_pred_all.extend(pred.cpu().numpy())
        
        acc = correct / total * 100
        performance_dict[scenario] = acc
        
        status = "✅ PASS" if acc > 90 else "⚠️ WEAK"
        print(f"{scenario:<15} | {total:<8} | {acc:.2f}%     | {status}")
    
    print("-" * 50)
    
    # 生成总的混淆矩阵
    cm = confusion_matrix(y_true_all, y_pred_all)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Clean', 'Spoofed'], yticklabels=['Clean', 'Spoofed'])
    plt.title('Confusion Matrix (Total)')
    plt.savefig(os.path.join(config.LOG_DIR, 'confusion_matrix.png'))
    print("🟦 混淆矩阵已保存至 logs/confusion_matrix.png")
    
    # 保存分场景图
    scenarios = list(performance_dict.keys())
    accs = list(performance_dict.values())
    plt.figure(figsize=(12, 6))
    plt.bar(scenarios, accs, color=['#4CAF50' if a > 90 else '#F44336' for a in accs])
    plt.axhline(y=90, color='r', linestyle='--', label='Target (90%)')
    plt.title('Detection Accuracy per Scenario')
    plt.ylim(0, 105)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(config.LOG_DIR, 'scenario_accuracy.png'))

if __name__ == "__main__":
    main()