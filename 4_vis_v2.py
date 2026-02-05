import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
import re
import sys

# ================= 🎛️ 绘图控制面板 (在这里决定生成哪些图) =================
PLOT_CONFIG = {
    # --- [新增] 基础信号分析 (你最想要的) ---
    "1_raw_signal_time_series": True,   # 绘制 C/N0 和 Doppler 随时间的变化曲线 (直观!)

    # --- [原有] 论文高级图表 (给审稿人看的) ---
    "2_training_curve":       False,    # 训练 Loss 和 准确率 曲线
    "3_feature_distribution": False,   # Fig2: 小提琴图 (数据分布对比)  % False
    "4_physics_consistency":  False,   # Fig3: 物理一致性 (红绿密度对比)
    "5_tanh_theory":          False,   # Fig4: Tanh 理论示意图
    
    # --- 输出设置 ---
    "save_fmt": "png",                 # 保存格式: 'png' 或 'pdf'
    "dpi": 300                         # 清晰度
}
# =======================================================================

# 全局路径设置
SAVE_DIR = "paper_plots_v2"
os.makedirs(SAVE_DIR, exist_ok=True)

# 字体设置 (支持中文显示需额外配置，这里使用通用英文科研风)
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3

# === 1. 工具函数 ===
def get_csv_dir():
    """智能寻找数据目录"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    # 优先找 data/processed
    paths = [
        os.path.join(current_dir, 'data', 'processed'),
        os.path.join(current_dir, '..', 'data', 'processed')
    ]
    for p in paths:
        if os.path.exists(p): return p
    return None

def find_latest_log():
    """寻找最新的 console_log.txt"""
    if os.path.exists("console_log.txt"): return "console_log.txt"
    if os.path.exists("logs"):
        subdirs = sorted([d for d in os.listdir("logs") if os.path.isdir(os.path.join("logs", d))])
        if subdirs: return os.path.join("logs", subdirs[-1], "console_log.txt")
    return None

# === 2. [新增] 绘制原始信号时间序列 ===
def plot_raw_signals():
    if not PLOT_CONFIG["1_raw_signal_time_series"]: return
    print("📈 [1] 正在绘制原始信号时间序列 (C/N0 & Doppler)...")
    
    csv_dir = get_csv_dir()
    if not csv_dir:
        print("   ❌ 找不到数据目录，跳过")
        return

    # 选择两个典型场景进行对比：静态正常 vs 静态欺骗
    scenarios = [
        ("cleanStatic_features.csv", "Clean Signal (Static)"),
        ("ds7_features.csv", "Spoofing Signal (SCER)") ,
        ("cleanDynamic_features.csv", "Clean Signal (Dynamic)"),
        ("ds5_features.csv", "Dynamic(Switching)"),
        ("ds6_features.csv", "Dynamic(Power Matching)")
    ]

    for fname, title_prefix in scenarios:
        fpath = os.path.join(csv_dir, fname)
        if not os.path.exists(fpath):
            print(f"   ⚠️ 文件不存在: {fname}")
            continue
            
        try:
            df = pd.read_csv(fpath)
            # 确保按 PRN 分组
            if 'PRN' not in df.columns:
                print(f"   ⚠️ {fname} 中没有 PRN 列，无法按卫星绘图")
                continue

            # 取前 5 颗卫星的数据
            top_prns = df['PRN'].unique()[:5]
            
            # --- 绘图: C/N0 ---
            plt.figure(figsize=(10, 4))
            for prn in top_prns:
                subset = df[df['PRN'] == prn]
                # 简单降采样防止点太多
                if len(subset) > 1000: subset = subset.iloc[::10]
                plt.plot(subset['Time'], subset['CN0_dBHz'], label=f'PRN {int(prn)}', linewidth=1.5)
            
            plt.xlabel('Time (s)')
            plt.ylabel('C/N0 (dB-Hz)')
            plt.title(f'{title_prefix} - Signal Strength over Time')
            plt.legend(loc='lower right', ncol=5)
            plt.tight_layout()
            save_name = f"1_Raw_CN0_{fname.split('_')[0]}.{PLOT_CONFIG['save_fmt']}"
            plt.savefig(os.path.join(SAVE_DIR, save_name), dpi=PLOT_CONFIG['dpi'])
            print(f"   ✅ 已保存: {save_name}")
            plt.close()

            # --- 绘图: Doppler ---
            plt.figure(figsize=(10, 4))
            for prn in top_prns:
                subset = df[df['PRN'] == prn]
                if len(subset) > 1000: subset = subset.iloc[::10]
                plt.plot(subset['Time'], subset['Doppler'], label=f'PRN {int(prn)}', linewidth=1.5)
            
            plt.xlabel('Time (s)')
            plt.ylabel('Doppler Frequency (Hz)')
            plt.title(f'{title_prefix} - Doppler Shift over Time')
            plt.legend(loc='upper right', ncol=5)
            plt.tight_layout()
            save_name = f"1_Raw_Doppler_{fname.split('_')[0]}.{PLOT_CONFIG['save_fmt']}"
            plt.savefig(os.path.join(SAVE_DIR, save_name), dpi=PLOT_CONFIG['dpi'])
            print(f"   ✅ 已保存: {save_name}")
            plt.close()
            
        except Exception as e:
            print(f"   ❌ 绘图失败 {fname}: {e}")

# === 3. 绘制训练曲线 (原有逻辑，适配配置) ===
def plot_training():
    if not PLOT_CONFIG["2_training_curve"]: return
    print("📉 [2] 正在绘制训练曲线...")
    
    log_path = find_latest_log()
    if not log_path: 
        print("   ❌ 无日志文件")
        return

    epochs, losses, accs = [], [], []
    with open(log_path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            match = re.search(r'Epoch (\d+) \| Loss: ([\d\.]+) \| Test Acc: ([\d\.]+)%', line)
            if match:
                epochs.append(int(match.group(1)))
                losses.append(float(match.group(2)))
                accs.append(float(match.group(3)))
    
    if not epochs: return

    fig, ax1 = plt.subplots(figsize=(8, 5))
    ax1.plot(epochs, losses, 'b-o', label='Loss')
    ax1.set_ylabel('Loss', color='b')
    ax2 = ax1.twinx()
    ax2.plot(epochs, accs, 'r--s', label='Accuracy')
    ax2.set_ylabel('Accuracy (%)', color='r')
    
    plt.title('Training Performance')
    save_name = f"2_Training_Curve.{PLOT_CONFIG['save_fmt']}"
    plt.savefig(os.path.join(SAVE_DIR, save_name), dpi=PLOT_CONFIG['dpi'])
    print(f"   ✅ 已保存: {save_name}")
    plt.close()

# === 4. 绘制特征分布 (原有逻辑) ===
def plot_dist():
    if not PLOT_CONFIG["3_feature_distribution"]: return
    print("🎻 [3] 正在绘制特征分布 (Fig2)...")
    csv_dir = get_csv_dir()
    if not csv_dir: return
    
    dfs = []
    targets = [('cleanStatic80_features.csv', 'Static'), ('cleanDynamic_features.csv', 'Dynamic'), ('ds7_features.csv', 'Spoofing')]
    for fname, label in targets:
        p = os.path.join(csv_dir, fname)
        if os.path.exists(p):
            try:
                d = pd.read_csv(p)
                if len(d)>2000: d = d.sample(2000)
                d['Norm_Doppler'] = np.tanh(d['Doppler']/1000.0)
                d['Type'] = label
                dfs.append(d[['Norm_Doppler', 'Type']])
            except: pass
            
    if dfs:
        full = pd.concat(dfs)
        plt.figure(figsize=(8, 5))
        sns.violinplot(data=full, x='Type', y='Norm_Doppler')
        plt.title('Feature Distribution (Tanh Normalized)')
        plt.tight_layout()
        save_name = f"3_Feature_Dist.{PLOT_CONFIG['save_fmt']}"
        plt.savefig(os.path.join(SAVE_DIR, save_name), dpi=PLOT_CONFIG['dpi'])
        print(f"   ✅ 已保存: {save_name}")
        plt.close()

# === 主程序 ===
if __name__ == "__main__":
    print(f"🚀 开始绘图 (保存格式: {PLOT_CONFIG['save_fmt']})")
    print(f"📂 保存目录: {os.path.abspath(SAVE_DIR)}")
    
    plot_raw_signals()  # 新增的
    plot_training()     # 训练曲线
    plot_dist()         # 分布图
    
    # 其他函数逻辑类似，按需开启即可
    print("🎉 绘图完成！")