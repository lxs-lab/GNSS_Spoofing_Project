import os

# === 1. 路径配置 (自动适配 Windows/Linux) ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_RAW_DIR = r'F:\李小双博士资料\2.观测数据\SDR Dataset\TEXBAT file'
DATA_PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
DATASET_DIR = os.path.join(DATA_PROC_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')
LOG_DIR = os.path.join(BASE_DIR, 'logs')

# 确保目录存在
for d in [DATA_PROC_DIR, DATASET_DIR, MODEL_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# === 2. 信号处理参数 (SDR Core) ===
FS = 25e6            # 原始采样率
DECIMATION = 6       # 降采样倍数 (加速)
WORK_FS = FS / DECIMATION
INT_TIME_MS = 4      # 积分时间
SEARCH_BAND = 6000   # 多普勒搜索范围
STEP_HZ = 250        # 多普勒步长 (精度)
ACQ_THRESHOLD = 35.0 # CN0 提取阈值 (稍低一点，保留更多特征给 GNN 判断)

# === 3. 数据集定义 (实验设计的核心) ===
# 格式: '文件名': Label (0=Clean, 1=Spoof)
DATA_FILES = {
    'cleanStatic80.bin': 0,  # [训练用] 绝对纯净的负样本
    'cleanStatic.bin': 0,
    'ds4.bin': 1,  # [训练用] 最难的静态欺骗
    # [测试用] 用来证明泛化能力 (Generalization)
    'ds1.bin': 1,  # 切换攻击
    'ds2.bin': 1,  # 高功率
    'ds3.bin': 1,  # 匹配功率
    'ds7.bin': 1,  # 功率匹配的时间推移攻击
    'ds8.bin': 1,  # 基于ds7的SCER（安全码估算与重放攻击）

    # 动态数据如果有也可以加进来
    'cleanDynamic.bin': 0,
    'ds5.bin': 1, 
    'ds6.bin': 1 


}

# === 4. 模型训练参数 ===
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
HIDDEN_DIM = 64
DROPOUT = 0.2