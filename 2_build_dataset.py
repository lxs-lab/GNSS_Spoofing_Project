from src.graph_builder import GNSSGraphDataset
from src import config
import os
import shutil
import torch

def main():
    print("="*40)
    print("      GNSS 图数据集构建系统       ")
    print("="*40)

    # 1. 强制清理旧缓存 (防止数据更新后 .pt 文件没更新)
    processed_dir = os.path.join(config.DATA_PROC_DIR, 'dataset', 'processed')
    if os.path.exists(processed_dir):
        print("🧹 清理旧缓存...")
        shutil.rmtree(processed_dir)

    # 2. 触发构建
    # root 目录设置为 data/processed/dataset
    dataset = GNSSGraphDataset(root=config.DATASET_DIR)

    # 3. 统计数据集信息 (写论文 Result 部分用)
    print("\n📊 数据集统计报告:")
    print(f"  - 总样本数: {len(dataset)}")
    print(f"  - 特征维度: {dataset.num_features} (CN0, Doppler)")
    print(f"  - 类别数: {dataset.num_classes}")
    
    # 统计训练/测试集分布
    train_count = 0
    test_count = 0
    clean_count = 0
    spoof_count = 0
    
    for data in dataset:
        if data.train_mask:
            train_count += 1
        else:
            test_count += 1
            
        if data.y.item() == 0:
            clean_count += 1
        else:
            spoof_count += 1

    print(f"  - 训练集样本 (ds0 + ds4): {train_count}")
    print(f"  - 测试集样本 (ds1/2/3...): {test_count}")
    print(f"  - 正负样本比例: Clean {clean_count} vs Spoof {spoof_count}")

    print("\n✅ 数据集准备就绪！下一步请运行 3_train_eval.py")

if __name__ == "__main__":
    main()