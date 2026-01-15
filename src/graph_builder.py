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
        
        # 训练集包含动态正常数据
        TRAIN_FILES = ['cleanStatic80.bin', 'cleanDynamic.bin', 'ds4.bin'] 

        print(f"🏗️ 开始构建全量物理图数据集 (Tanh Normalized)...")
        print(f"🎯 训练集文件定义: {TRAIN_FILES}")

        for filename, label in config.DATA_FILES.items():
            csv_name = filename.replace('.bin', '_features.csv')
            csv_path = os.path.join(config.DATA_PROC_DIR, csv_name)
            
            if not os.path.exists(csv_path):
                print(f"⚠️ 跳过缺失文件: {csv_name}")
                continue
            
            is_train = (filename in TRAIN_FILES)
            dataset_type = "TRAIN" if is_train else "TEST"
            print(f"📦 处理 {filename} -> Label: {label} [{dataset_type}]")

            df = pd.read_csv(csv_path)
            grouped = df.groupby('Time')
            
            for time, group in tqdm(grouped, desc=f"  Parsing {filename}", leave=False):
                # === 1. 节点特征构建 (使用 Tanh 优化) ===
                raw_cn0 = group['CN0_dBHz'].values.astype(float)
                raw_doppler = group['Doppler'].values.astype(float)
                
                # 创建特征矩阵 [num_sats, 2]
                x_np = np.stack([raw_cn0, raw_doppler], axis=1)
                
                # CN0 归一化: (x - 45) / 10
                x_np[:, 0] = (x_np[:, 0] - 45.0) / 10.0
                
                # Doppler 归一化: Tanh(x / 1000)
                # 0Hz -> 0; 1000Hz -> 0.76; 5000Hz -> 0.999
                x_np[:, 1] = np.tanh(x_np[:, 1] / 1000.0)
                
                x = torch.tensor(x_np, dtype=torch.float)
                
                num_nodes = x.shape[0]
                if num_nodes < 4: 
                    continue
                
                # === 2. 边特征构建 (差分 + Tanh) ===
                edge_index = []
                edge_attr = []
                
                # 使用原始值计算差分，避免精度损失
                raw_data = group[['CN0_dBHz', 'Doppler']].values.astype(float)

                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:
                            edge_index.append([i, j])
                            
                            # CN0 差分
                            d_cn0 = (raw_data[i, 0] - raw_data[j, 0]) / 10.0
                            
                            # Doppler 差分: 同样使用 Tanh
                            d_doppler_val = raw_data[i, 1] - raw_data[j, 1]
                            d_doppler = np.tanh(d_doppler_val / 1000.0)
                            
                            edge_attr.append([d_cn0, d_doppler])
                
                edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float)
                
                y = torch.tensor([label], dtype=torch.long)
                
                data = Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, y=y)
                data.timestamp = float(time)
                data.train_mask = is_train
                data.scenario = filename.replace('.bin', '') 
                
                data_list.append(data)

        if len(data_list) == 0:
            print("❌ 严重错误: 没有构建出任何图数据！")
            return

        import random
        random.shuffle(data_list)

        print(f"💾 正在保存 {len(data_list)} 个图样本到磁盘...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("✅ 数据集构建完成！")