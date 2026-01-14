import os

# === 1. 路径配置 ===
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# DATA_RAW_DIR = os.path.join(BASE_DIR, 'data', 'raw')
DATA_RAW_DIR = r'F:\李小双博士资料\2.观测数据\SDR Dataset\TEXBAT file'
DATA_PROC_DIR = os.path.join(BASE_DIR, 'data', 'processed')
DATASET_DIR = os.path.join(DATA_PROC_DIR, 'dataset')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

# === 修改点：只定义基础 logs 目录，不自动创建子文件夹 ===
LOG_BASE_DIR = os.path.join(BASE_DIR, 'logs')
# 默认 LOG_DIR 就指向 logs，后续由主程序决定是否更新为子文件夹
LOG_DIR = LOG_BASE_DIR 

# 确保基础目录存在
for d in [DATA_PROC_DIR, DATASET_DIR, MODEL_DIR, LOG_BASE_DIR]:
    os.makedirs(d, exist_ok=True)

# === 2. 信号处理参数 ===
FS = 25e6            
DECIMATION = 6       
WORK_FS = FS / DECIMATION
INT_TIME_MS = 4      
SEARCH_BAND = 6000   
STEP_HZ = 250        
ACQ_THRESHOLD = 35.0 

# === 3. 数据集定义 ===
DATA_FILES = {
    'cleanStatic80.bin': 0, 
    'cleanStatic.bin': 0,
    'cleanDynamic.bin': 0, 
    'ds4.bin': 1,  
    'ds1.bin': 1, 
    'ds2.bin': 1, 
    'ds3.bin': 1, 
    'ds5.bin': 1, 
    'ds6.bin': 1, 
    'ds7.bin': 1, 
    'ds8.bin': 1, 
}

# === 4. 模型训练参数 ===
BATCH_SIZE = 64
EPOCHS = 50
LR = 0.001
HIDDEN_DIM = 64
DROPOUT = 0.2