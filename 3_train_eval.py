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
import shutil
import sys
from datetime import datetime

# === 日志记录器 ===
class Logger(object):
    def __init__(self, filename=None):
        self.terminal = sys.stdout
        self.log = None
        if filename:
            self.log = open(filename, "a", encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        if self.log:
            self.log.write(message)
            self.log.flush()

    def flush(self):
        self.terminal.flush()
        if self.log:
            self.log.flush()

def main():
    # === 1. 询问保存 ===
    print("="*40)
    user_input = input("💾 是否保存本次实验结果? (y/n): ").strip().lower()
    save_results = (user_input == 'y')
    
    if save_results:
        current_time_str = datetime.now().strftime('%Y%m%d%H%M')
        config.LOG_DIR = os.path.join(config.LOG_BASE_DIR, current_time_str)
        os.makedirs(config.LOG_DIR, exist_ok=True)
        
        log_file_path = os.path.join(config.LOG_DIR, 'console_log.txt')
        sys.stdout = Logger(log_file_path)
        print(f"📂 结果将保存至: {config.LOG_DIR}")

        # 保存参数小票
        param_file_path = os.path.join(config.LOG_DIR, 'experiment_config.txt')
        with open(param_file_path, 'w', encoding='utf-8') as f:
            f.write("="*30 + "\n")
            f.write("   🧪 实验配置报告 (Config)   \n")
            f.write("="*30 + "\n")
            f.write(f"Timestamp    : {current_time_str}\n")
            f.write(f"Model Type   : ST-GraphTransformer (Tanh Norm)\n")
            f.write("-" * 30 + "\n")
            f.write(f"Epochs       : {config.EPOCHS}\n")
            f.write(f"Batch Size   : {config.BATCH_SIZE}\n")
            f.write(f"Learning Rate: {config.LR}\n")
            f.write(f"Hidden Dim   : {config.HIDDEN_DIM}\n")
            f.write(f"Dropout      : {config.DROPOUT}\n")
            f.write("-" * 30 + "\n")
            f.write(f"Dataset Path : {config.DATASET_DIR}\n")
            f.write("="*30 + "\n")
    else:
        print("🚫 本次运行不保存结果")
        sys.stdout = Logger(None)

    print("="*40)
    print("      GNSS 模型训练 (Tanh Fix + Capacity Restore)       ")
    print("="*40)
    
    # === 自动清理旧缓存 ===
    processed_dir = os.path.join(config.DATASET_DIR, 'processed')
    if os.path.exists(processed_dir):
        try:
            shutil.rmtree(processed_dir)
            print(f"🧹 已清理旧缓存，强制重新构建数据...")
        except: pass
    
    # 2. 加载数据
    dataset = GNSSGraphDataset(root=config.DATASET_DIR)
    
    train_data = [d for d in dataset if d.train_mask == True]
    test_data  = [d for d in dataset if d.train_mask == False]
    
    # 统计分布
    train_labels = [d.y.item() for d in train_data]
    n_clean = train_labels.count(0)
    n_spoof = train_labels.count(1)
    print(f"📊 训练集分布: Clean={n_clean}, Spoof={n_spoof}")
    
    # [修改] 移除自动加权，使用均等权重
    class_weights = torch.tensor([1.0, 1.0]).float()
    
    train_loader = DataLoader(train_data, batch_size=config.BATCH_SIZE, shuffle=True)
    test_loader  = DataLoader(test_data, batch_size=config.BATCH_SIZE, shuffle=False)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"⚙️  Class Weights: {class_weights} (Training on {device})")
    
    # 3. 初始化模型
    # 自动读取 config 里的 hidden_dim
    model = STGraphTransformer(in_channels=dataset.num_features, 
                               hidden_channels=config.HIDDEN_DIM,
                               edge_dim=2,
                               dropout=config.DROPOUT).to(device)
    
    # [修改] Weight Decay 恢复为 1e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LR, weight_decay=1e-4)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weights.to(device))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    
    # 4. 训练
    best_acc = 0.0
    train_losses = []
    test_accs = []

    print("\n🚀 开始训练...")
    for epoch in range(1, config.EPOCHS + 1):
        model.train()
        total_loss = 0
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        train_losses.append(avg_loss)
        
        # Test
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
        scheduler.step(avg_loss)
        
        if acc > best_acc:
            best_acc = acc
            if save_results:
                torch.save(model.state_dict(), os.path.join(config.MODEL_DIR, 'best_model.pth'))
            
        # 每轮都打印，方便观察
        if epoch % 5 == 0:
            print(f"Epoch {epoch:02d} | Loss: {avg_loss:.4f} | Test Acc: {acc*100:.2f}%")

    print(f"\n✅ 训练结束! 最佳测试准确率: {best_acc*100:.2f}%")

    # === 5. 可视化 & 评估 ===
    if save_results:
        # 训练曲线
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.plot(train_losses, label='Train Loss')
        plt.title('Training Loss')
        plt.subplot(1, 2, 2)
        plt.plot(test_accs, label='Test Accuracy', color='orange')
        plt.title('Generalization Accuracy')
        plt.savefig(os.path.join(config.LOG_DIR, 'training_curves.png'))
        
        # 加载最佳模型进行分场景评估
        if os.path.exists(os.path.join(config.MODEL_DIR, 'best_model.pth')):
            model.load_state_dict(torch.load(os.path.join(config.MODEL_DIR, 'best_model.pth'), weights_only=True))
    
    # 分场景评估
    print("\n" + "="*50)
    print("📊 最终分场景评估 (Scenario Eval)")
    print("="*50)
    print(f"{'Scenario':<15} | {'Acc (%)':<10} | {'Status'}")
    print("-" * 50)
    
    model.eval()
    scenarios = sorted(list(set([d.scenario for d in test_data if hasattr(d, 'scenario')])))
    performance_dict = {}
    
    # 收集全量预测用于混淆矩阵
    y_true_all = []
    y_pred_all = []

    for sce in scenarios:
        subset = [d for d in test_data if getattr(d, 'scenario', '') == sce]
        if not subset: continue
        
        loader = DataLoader(subset, batch_size=config.BATCH_SIZE)
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
        performance_dict[sce] = acc
        status = "✅" if acc > 85 else "⚠️"
        print(f"{sce:<15} | {acc:.2f}%     | {status}")

    if save_results:
        # 混淆矩阵
        cm = confusion_matrix(y_true_all, y_pred_all)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Clean', 'Spoofed'], yticklabels=['Clean', 'Spoofed'])
        plt.title('Confusion Matrix')
        plt.savefig(os.path.join(config.LOG_DIR, 'confusion_matrix.png'))
        
        # 柱状图
        plt.figure(figsize=(12, 6))
        keys = list(performance_dict.keys())
        vals = list(performance_dict.values())
        plt.bar(keys, vals, color=['#4CAF50' if a > 85 else '#F44336' for a in vals])
        plt.axhline(y=85, color='r', linestyle='--')
        plt.title('Scenario Accuracy')
        plt.ylim(0, 105)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(config.LOG_DIR, 'scenario_accuracy.png'))
        
        print(f"✅ 结果已保存至 {config.LOG_DIR}")

if __name__ == "__main__":
    main()