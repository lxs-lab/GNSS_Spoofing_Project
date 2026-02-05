from src.extractor import FeatureExtractor
from src import config
import time

# remark

def main():
    print("="*40)
    print("      GNSS 特征批量提取系统       ")
    print("="*40)
    
    extractor = FeatureExtractor()
    
    # 遍历 Config 中定义的所有文件
    total_files = len(config.DATA_FILES)
    print(f"📝 计划处理 {total_files} 个文件: {list(config.DATA_FILES.keys())}")
    
    start_time = time.time()
    
    for filename, label in config.DATA_FILES.items():
        print(f"\n📂 正在处理: {filename} (Label: {label})")
        extractor.process_single_file(filename)
        
    print(f"\n✅ 所有文件处理完毕! 总耗时: {(time.time() - start_time)/60:.1f} 分钟")
    print(f"📁 结果保存在: {config.DATA_PROC_DIR}")

if __name__ == "__main__":
    main()