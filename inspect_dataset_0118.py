import torch
import numpy as np
from src.graph_builder import GNSSGraphDataset
from src import config
import pandas as pd

def inverse_tanh(y, scale=1000.0):
    """将 Tanh 归一化后的数值反推回物理值 (仅用于展示)"""
    # 避免 arctanh 在 +/-1 处爆炸
    y = np.clip(y, -0.99999, 0.99999)
    return np.arctanh(y) * scale

def inverse_cn0(y):
    """将归一化 CN0 反推回 dBHz"""
    return y * 10.0 + 45.0

def inspect_sample(data):
    print(f"\n{'='*20} 场景: {data.scenario} (Label: {data.y.item()}) {'='*20}")
    print(f"时间戳: {data.timestamp:.2f}")
    
    # --- 1. 检查节点特征 (Node Features) ---
    print(f"\n[1] 节点特征矩阵 X (Shape: {data.x.shape})")
    print(f"{'SV_Idx':<8} | {'CN0 (Norm)':<12} | {'CN0 (Real)':<12} | {'Doppler (Norm)':<15} | {'Doppler (Real)':<15}")
    print("-" * 80)
    
    num_nodes = data.x.shape[0]
    for i in range(min(num_nodes, 5)): # 只打印前5颗星
        cn0_norm = data.x[i, 0].item()
        dop_norm = data.x[i, 1].item()
        
        cn0_real = inverse_cn0(cn0_norm)
        dop_real = inverse_tanh(dop_norm, scale=1000.0)
        
        print(f"Node {i:<3} | {cn0_norm:<12.4f} | {cn0_real:<12.1f} | {dop_norm:<15.4f} | {dop_real:<15.1f}")
    
    if num_nodes > 5:
        print(f"... (还有 {num_nodes-5} 个节点)")

    # --- 2. 检查边特征 (Edge Features) ---
    print(f"\n[2] 边特征矩阵 Edge_Attr (Shape: {data.edge_attr.shape})")
    print(f"{'Link':<10} | {'d_CN0 (Norm)':<12} | {'d_Dop (Norm)':<15} | {'d_Dop (Real Hz)':<15}")
    print("-" * 80)
    
    num_edges = data.edge_attr.shape[0]
    for i in range(min(num_edges, 5)):
        src = data.edge_index[0, i].item()
        dst = data.edge_index[1, i].item()
        
        d_cn0_norm = data.edge_attr[i, 0].item()
        d_dop_norm = data.edge_attr[i, 1].item()
        
        # 差分的多普勒也是用 Tanh(x/1000) 归一化的
        d_dop_real = inverse_tanh(d_dop_norm, scale=1000.0)
        
        print(f"{src}->{dst:<5} | {d_cn0_norm:<12.4f} | {d_dop_norm:<15.4f} | {d_dop_real:<15.1f}")
    
    # --- 3. 统计 Tanh 饱和度 ---
    dop_vals = data.x[:, 1].abs()
    saturated = (dop_vals > 0.95).sum().item()
    print(f"\n⚠️ Tanh 饱和检查: {saturated}/{num_nodes} 个节点的 Doppler 归一化值 > 0.95")
    if saturated > num_nodes / 2:
        print("   (提示: 动态太大，Tanh 接近饱和，可能丢失细节，但这在动态场景下是预期的)")

def main():
    print("正在加载数据集 (不重新构建)...")
    dataset = GNSSGraphDataset(root=config.DATASET_DIR)
    
    # 1. 找一个 Clean Static 样本
    clean_data = next(d for d in dataset if 'cleanStatic' in d.scenario)
    inspect_sample(clean_data)
    
    # 2. 找一个 欺骗 (DS7) 样本
    try:
        spoof_data = next(d for d in dataset if 'ds7' in d.scenario)
        inspect_sample(spoof_data)
    except StopIteration:
        print("未找到 ds7 样本")

if __name__ == "__main__":
    main()