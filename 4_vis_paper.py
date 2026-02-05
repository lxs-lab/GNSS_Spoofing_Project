import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import re
import torch
import sys

# 尝试导入项目配置，如果失败则使用本地回退逻辑
try:
    from src.graph_builder import GNSSGraphDataset
    from src import config
    HAS_CONFIG = True
except ImportError:
    HAS_CONFIG = False
    print("⚠️ 未能导入 src 模块，将使用纯文件路径模式运行...")

# === 0. 全局设置 ===
SAVE_DIR = "paper_plots"
os.makedirs(SAVE_DIR, exist_ok=True)

# 字体设置 (IEEE 风格)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 600

# === 1. 核心工具：智能路径寻找 ===
def get_csv_dir():
    """
    智能寻找 data/processed 目录
    1. 优先检查当前脚本目录下的 data/processed (相对路径)
    2. 其次检查 config 中定义的绝对路径
    """
    # 方案 A: 相对路径 (最稳)
    current_dir = os.path.dirname(os.path.abspath(__file__))
    relative_path = os.path.join(current_dir, 'data', 'processed')
    
    if os.path.exists(relative_path):
        return relative_path
    
    # 方案 B: Config 路径
    if HAS_CONFIG and os.path.exists(config.DATA_PROC_DIR):
        return config.DATA_PROC_DIR
        
    # 方案 C: 暴力搜索
    if os.path.exists("data/processed"):
        return "data/processed"
        
    return None

def find_latest_log():
    """寻找最新的 console_log.txt"""
    # 1. 根目录
    if os.path.exists("console_log.txt"):
        return "console_log.txt"
    # 2. logs 目录
    if os.path.exists("logs"):
        # 找最新的子文件夹
        subdirs = sorted([d for d in os.listdir("logs") if os.path.isdir(os.path.join("logs", d))])
        if subdirs:
            return os.path.join("logs", subdirs[-1], "console_log.txt")
    return None

# === 2. 绘图函数 ===

def plot_training_curves():
    print("📈 [Fig 1] 正在绘制训练动态曲线...")
    log_path = find_latest_log()
    if not log_path:
        print("   ❌ 未找到 console_log.txt，跳过 Fig 1")
        return

    epochs, losses, accs = [], [], []
    try:
        with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                # 解析: Epoch 05 | Loss: 0.4061 | Test Acc: 63.24%
                match = re.search(r'Epoch (\d+) \| Loss: ([\d\.]+) \| Test Acc: ([\d\.]+)%', line)
                if match:
                    epochs.append(int(match.group(1)))
                    losses.append(float(match.group(2)))
                    accs.append(float(match.group(3)))
    except Exception as e:
        print(f"   ⚠️ 读取 Log 出错: {e}")
        return

    if not epochs:
        print("   ⚠️ Log 中未提取到数据，请检查格式。")
        return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    
    color = '#1f77b4'
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Training Loss', color=color, fontweight='bold')
    l1, = ax1.plot(epochs, losses, color=color, linewidth=2, marker='o', markersize=4, label='Cross Entropy Loss')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = ax1.twinx()  
    color = '#ff7f0e'
    ax2.set_ylabel('Generalization Accuracy (%)', color=color, fontweight='bold')
    l2, = ax2.plot(epochs, accs, color=color, linewidth=2, marker='s', markersize=4, linestyle='--', label='Test Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)
    
    # 标注最佳点
    best_idx = np.argmax(accs)
    best_acc = accs[best_idx]
    best_epoch = epochs[best_idx]
    ax2.annotate(f'Best: {best_acc:.2f}%', xy=(best_epoch, best_acc), xytext=(best_epoch-10, best_acc-5),
                 arrowprops=dict(facecolor='black', arrowstyle='->'), fontweight='bold')

    plt.title('Training & Generalization Dynamics')
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'Fig1_Training_Dynamics.pdf')
    plt.savefig(save_path)
    print(f"   ✅ 已保存: {save_path}")
    plt.close()

def plot_feature_distribution():
    print("🎻 [Fig 2] 正在绘制特征分布小提琴图...")
    csv_dir = get_csv_dir()
    if not csv_dir:
        print("   ❌ 无法找到 data/processed 目录，跳过 Fig 2")
        return
    
    print(f"   📂 CSV搜索目录: {os.path.abspath(csv_dir)}")

    # 定义要对比的场景和文件名 (确保文件名对应)
    targets = [
        ('cleanStatic80_features.csv', 'Clean (Static)'),
        ('cleanDynamic_features.csv', 'Clean (Dynamic)'),
        ('ds7_features.csv', 'Spoofing (SCER)'),
    ]
    
    dfs = []
    for fname, label in targets:
        fpath = os.path.join(csv_dir, fname)
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                # 采样以加快绘图
                if len(df) > 3000: df = df.sample(3000, random_state=42)
                
                # 模拟 Tanh 归一化
                df['Doppler_Norm'] = np.tanh(df['Doppler'] / 1000.0)
                df['Scenario'] = label
                dfs.append(df[['Doppler_Norm', 'Scenario']])
                print(f"     - 已加载: {fname} ({len(df)} 行)")
            except Exception as e:
                print(f"     ⚠️ 读取 {fname} 失败: {e}")
        else:
            print(f"     ⚠️ 文件未找到: {fname}")

    if not dfs:
        print("   ❌ 没有加载到任何 CSV 数据，跳过 Fig 2")
        return

    full_df = pd.concat(dfs)

    plt.figure(figsize=(8, 5))
    sns.violinplot(data=full_df, x='Scenario', y='Doppler_Norm', palette="Set2", inner="quartile", linewidth=1)
    
    plt.axhline(0, color='grey', linestyle='--', alpha=0.5)
    plt.ylabel(r'Normalized Doppler ($\tanh(\Delta f / 1000)$)')
    plt.xlabel('')
    plt.title('Feature Distribution after Physics-Aware Normalization')
    plt.ylim(-1.1, 1.1)
    
    # 标注
    plt.text(0, -0.95, "Concentrated (Stable)", ha='center', fontsize=8, color='green')
    plt.text(1, 0.95, "Dispersed (High Dynamic)", ha='center', fontsize=8, color='orange')
    plt.text(2, -0.95, "Anomalous Pattern", ha='center', fontsize=8, color='red')
    
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'Fig2_Feature_Distribution.pdf')
    plt.savefig(save_path)
    print(f"   ✅ 已保存: {save_path}")
    plt.close()

def plot_edge_physics():
    print("🕸️ [Fig 3] 正在绘制边特征(物理一致性)分析...")
    if not HAS_CONFIG:
        print("   ❌ 缺少 src 配置，无法加载 PyG 数据集，跳过 Fig 3")
        return

    try:
        # 这里的 root 必须指向 data/processed/dataset (不包含 processed 子文件夹)
        # 我们手动构建路径以防 config 有误
        csv_dir = get_csv_dir()
        if csv_dir:
            dataset_root = os.path.join(csv_dir, 'dataset')
        else:
            dataset_root = config.DATASET_DIR

        print(f"   📂 数据集加载路径: {dataset_root}")
        if not os.path.exists(dataset_root):
             print(f"   ❌ 路径不存在: {dataset_root}")
             return

        dataset = GNSSGraphDataset(root=dataset_root)
        print(f"     - 成功加载数据集: {len(dataset)} 个样本")
    except Exception as e:
        print(f"   ⚠️ 加载数据集失败: {e}，跳过 Fig 3")
        return

    # 提取边特征
    clean_edges = []
    spoof_edges = []
    
    # 随机采样 500 个样本
    indices = np.random.choice(len(dataset), min(len(dataset), 500), replace=False)
    
    for idx in indices:
        data = dataset[idx]
        if data.edge_attr is None or data.edge_attr.shape[0] == 0:
            continue
            
        # edge_attr[:, 1] 是 Delta Doppler (Tanh后的)
        d_dop = data.edge_attr[:, 1].numpy()
        
        if data.y.item() == 0: # Clean
            clean_edges.extend(d_dop)
        else:
            spoof_edges.extend(d_dop)
    
    if not clean_edges or not spoof_edges:
        print("   ⚠️ 未提取到足够的边特征数据。")
        return

    plt.figure(figsize=(8, 5))
    # 使用 KDE 绘制密度图
    sns.kdeplot(clean_edges, label='Clean Signals', color='green', fill=True, alpha=0.3, linewidth=2)
    sns.kdeplot(spoof_edges, label='Spoofing Signals', color='red', fill=True, alpha=0.3, linewidth=2)
    
    plt.xlabel(r'Edge Attribute: $\Delta$Doppler (Normalized)')
    plt.ylabel('Density Probability')
    plt.title('Physics Consistency Check: Edge Feature Distribution')
    plt.legend()
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.xlim(-0.5, 0.5) # 聚焦在中心区域看差异
    
    plt.text(0, plt.ylim()[1]*0.9, "Physically Consistent\n(Peaked at 0)", ha='center', color='green', fontweight='bold', fontsize=9)
    
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'Fig3_Edge_Physics.pdf')
    plt.savefig(save_path)
    print(f"   ✅ 已保存: {save_path}")
    plt.close()

def plot_tanh_theory():
    print("📐 [Fig 4] 正在绘制 Tanh 理论示意图...")
    x = np.linspace(-5000, 5000, 1000)
    y = np.tanh(x / 1000.0)
    
    plt.figure(figsize=(6, 4))
    plt.plot(x, y, linewidth=3, color='#8e44ad')
    
    # 标注区域
    plt.axvspan(-1000, 1000, color='green', alpha=0.1, label='Linear Region (Static)')
    plt.axvspan(1000, 5000, color='orange', alpha=0.1, label='Compression Region (Dynamic)')
    plt.axvspan(-5000, -1000, color='orange', alpha=0.1)
    
    plt.xlabel('Raw Doppler (Hz)')
    plt.ylabel('Normalized Feature Value')
    plt.title('Soft-Normalization Strategy (Tanh)')
    plt.legend(loc='lower right')
    plt.grid(True, linestyle='--')
    
    plt.tight_layout()
    save_path = os.path.join(SAVE_DIR, 'Fig4_Tanh_Theory.pdf')
    plt.savefig(save_path)
    print(f"   ✅ 已保存: {save_path}")
    plt.close()

if __name__ == "__main__":
    print(f"🚀 开始生成科研绘图 (Smart Path Mode)...")
    print(f"📂 输出目录: {os.path.abspath(SAVE_DIR)}")
    print("-" * 50)
    
    plot_training_curves()
    print("-" * 20)
    plot_feature_distribution()
    print("-" * 20)
    plot_edge_physics()
    print("-" * 20)
    plot_tanh_theory()
    
    print("-" * 50)
    print("🎉 所有任务完成！请查看 paper_plots 文件夹。")