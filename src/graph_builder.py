import torch
from torch_geometric.data import Data, InMemoryDataset
import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import shutil
from . import config

class GNSSGraphDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(GNSSGraphDataset, self).__init__(root, transform, pre_transform)
        # 显式设置 weights_only=False 修复 PyTorch 2.6+ 安全限制
        if os.path.exists(self.processed_paths[0]):
            try:
                self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
            except Exception:
                print("⚠️ 旧数据文件无法加载，正在重新构建...")
                self.process()
                self.data, self.slices = torch.load(self.processed_paths[0], weights_only=False)
        else:
            self.process()

    @property
    def raw_file_names(self):
        return []

    @property
    def processed_file_names(self):
        return ['gnss_full_dataset.pt']

    def download(self):
        pass

    def process(self):
        data_list = []
        
        # === 修改 1: 完善训练集定义 ===
        # 必须包含 cleanDynamic (动态正常)，防止模型把"运动"当成"欺骗"
        TRAIN_FILES = ['cleanStatic.bin', 'cleanDynamic.bin', 'ds4.bin'] 

        print(f"🏗️ 开始构建全量物理图数据集...")
        print(f"🎯 训练集文件定义: {TRAIN_FILES}")

        # 遍历 config 中定义的所有文件
        for filename, label in config.DATA_FILES.items():
            csv_name = filename.replace('.bin', '_features.csv')
            csv_path = os.path.join(config.DATA_PROC_DIR, csv_name)
            
            if not os.path.exists(csv_path):
                print(f"⚠️ 跳过缺失文件: {csv_name}")
                continue
            
            # 判断该文件属于训练集还是测试集
            is_train = (filename in TRAIN_FILES)
            dataset_type = "TRAIN" if is_train else "TEST"
            print(f"📦 处理 {filename} -> Label: {label} [{dataset_type}]")

            # 读取 CSV
            df = pd.read_csv(csv_path)
            
            # 按时间戳分组构建图
            grouped = df.groupby('Time')
            for time, group in tqdm(grouped, desc=f"  Parsing {filename}", leave=False):
                
                # === 修改 2: 节点特征只保留 CN0 (去除绝对多普勒) ===
                # 目的：切断模型直接获取"接收机是否运动"的途径，强迫它学习相对关系
                # 归一化: CN0 / 50.0
                x_np = group[['CN0_dBHz']].values.astype(float)
                x_np[:, 0] /= 50.0  
                x = torch.tensor(x_np, dtype=torch.float)
                
                # 过滤掉卫星数太少的帧
                num_nodes = x.shape[0]
                if num_nodes < 4: 
                    continue
                
                # === 修改 3: 构建全连接边并计算物理差值 ===
                edge_index = []
                edge_attr = []
                
                # 为了计算差值，我们需要原始的 CN0 和 Doppler 数据
                raw_data = group[['CN0_dBHz', 'Doppler']].values.astype(float)

                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:
                            edge_index.append([i, j])
                            
                            # [物理核心] 计算节点 i 和 j 的相对差值
                            # 1. 相对信号强度
                            diff_cn0 = (raw_data[i, 0] - raw_data[j, 0]) / 50.0
                            # 2. 相对多普勒 (关键特征！反映几何一致性)
                            diff_doppler = (raw_data[i, 1] - raw_data[j, 1]) / 5000.0
                            
                            edge_attr.append([diff_cn0, diff_doppler])
                
                edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
                
                # 标签
                y = torch.tensor([label], dtype=torch.long)
                
                # 创建图对象
                data = Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, y=y)
                
                # 附加元数据
                data.timestamp = float(time)
                data.train_mask = is_train
                # === 修改 4: 注入场景标签 ===
                data.scenario = filename.replace('.bin', '') # 例如 'ds7'
                
                data_list.append(data)

        if len(data_list) == 0:
            print("❌ 严重错误: 没有构建出任何图数据！")
            return

        # 随机打乱
        import random
        random.shuffle(data_list)

        print(f"💾 正在保存 {len(data_list)} 个图样本到磁盘...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("✅ 数据集构建完成！")