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

def main():
    print("="*40)
    print("      GNSS 模型训练与评估系统       ")
    print("="*40)
    
    # 1. 加载数据集
    dataset = GNSSGraphDataset(root=config.DATASET_DIR)
    
    # 2. 智能切分 (根据构建时的 train_mask)
    # 注意: dataset[i] 是一个 Data 对象
    train_data = [data for data in dataset if data.train_mask == True]
    test_data  = [data for data in dataset if data.train_mask == False]
    
    print(f"📊 数据集加载完毕:")
    print(f"  - 训练集 (Known Scenarios): {len(train_data)} 样本")
    print(f"  - 测试集 (Unknown Scenarios): {len(test_data)} 样本")
    
    if len(train_data) == 0:
        print("❌ 错误: 训练集为空！请检查 config.py 中的 DATA_FILES 设置。")
        return

    # 3. 创建 DataLoader
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    # 测试集不打乱，方便后续分析
    test_loader  = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # 4. 初始化模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  Training on: {device}")
    
    # model = STGraphTransformer(in_channels=dataset.num_features).to(device)
    model = STGraphTransformer(in_channels=dataset.num_features, edge_dim=2).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR)
    criterion = torch.nn.CrossEntropyLoss()
    
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
            # out = model(data.x, data.edge_index, data.batch)
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
                # out = model(data.x, data.edge_index, data.batch)
                out = model(data.x, data.edge_index, data.edge_attr, data.batch)
                pred = out.argmax(dim=1)
                correct += int((pred == data.y).sum())
                total += data.y.size(0)
        
        acc = correct / total if total > 0 else 0
        test_accs.append(acc)
        
        if acc > best_acc:
            best_acc = acc
            # 保存最佳模型
            torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, 'best_model.pth'))
            
        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Test Acc: {acc*100:.2f}%")

    print(f"\n✅ 训练结束! 最佳测试准确率: {best_acc*100:.2f}%")

    # === 5. 结果可视化 (写论文用) ===
    # 图1: 训练曲线
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

    # === 6. 深度评估 (混淆矩阵) ===
    print("\n🔍 正在生成深度评估报告...")
    model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, 'best_model.pth'), weights_only=True))
    model.eval()
    
    y_true = []
    y_pred = []
    
    with torch.no_grad():
        for data in test_loader:
            data = data.to(device)
            # out = model(data.x, data.edge_index, data.batch)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            pred = out.argmax(dim=1)
            
            y_true.extend(data.y.cpu().numpy())
            y_pred.extend(pred.cpu().numpy())
            
    # 生成混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    target_names = ['Clean', 'Spoofed']
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix (Test Set)')
    plt.savefig(os.path.join(config.LOG_DIR, 'confusion_matrix.png'))
    print("🟦 混淆矩阵已保存至 logs/confusion_matrix.png")
    
    # 打印详细报告
    print("\n" + classification_report(y_true, y_pred, target_names=target_names, digits=4))

    # ===  7. 分场景详细性能评估 (Scenario-wise Evaluation) ===
    # 这是论文 "Experimental Results" 章节的核心数据来源
    print("\n" + "="*50)
    print("📊 分场景详细性能评估 (Scenario-wise Evaluation)")
    print("="*50)
    print(f"{'Scenario':<15} | {'Samples':<8} | {'Acc (%)':<10} | {'Status'}")
    print("-" * 50)

    # 提取所有唯一的场景名
    # 注意: 需要先重新加载 dataset 或者确保 test_data 里的对象有 scenario 属性
    # 如果报错 AttributeError，说明你还没运行上面的 graph_builder 修改并删除旧 .pt 文件
    test_scenarios = sorted(list(set([d.scenario for d in test_data])))
    
    performance_dict = {}

    for scenario in test_scenarios:
        # 筛选出属于该场景的数据
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
        
        acc = correct / total * 100
        performance_dict[scenario] = acc
        
        # 状态判定
        status = "✅ PASS" if acc > 90 else "⚠️ WEAK"
        print(f"{scenario:<15} | {total:<8} | {acc:.2f}%     | {status}")
    
    print("-" * 50)
    
    # 画个柱状图保存下来 (论文 Fig. X)
    scenarios = list(performance_dict.keys())
    accs = list(performance_dict.values())
    
    plt.figure(figsize=(12, 6))
    bars = plt.bar(scenarios, accs, color=['#4CAF50' if a > 90 else '#F44336' for a in accs])
    plt.axhline(y=90, color='r', linestyle='--', label='Target (90%)')
    plt.title('Detection Accuracy per Attack Scenario')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    plt.xticks(rotation=45)
    
    # 在柱子上标数值
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom')
                
    plt.tight_layout()
    plt.savefig(os.path.join(config.LOG_DIR, 'scenario_accuracy.png'))
    print("📈 分场景精度对比图已保存: logs/scenario_accuracy.png")

if __name__ == "__main__":
    main()