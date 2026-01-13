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
        
        # 定义训练集包含的文件 (根据你的策略)
        # 这里的名字必须和 config.DATA_FILES 里的键(Key)一致
        # ds0/cleanStatic80 用于学习正常，ds4 用于学习隐蔽欺骗
        TRAIN_FILES = ['cleanStatic.bin', 'cleanDynamic.bin', 'ds4.bin'] 

        print(f"🏗️ 开始构建图数据集...")
        print(f"🎯 训练集文件定义: {TRAIN_FILES}")

        # 遍历 config 中定义的所有文件
        for filename, label in config.DATA_FILES.items():
            # ... (这部分寻找CSV路径的代码保持不变) ...
            csv_name = filename.replace('.bin', '_features.csv')
            csv_path = os.path.join(config.DATA_PROC_DIR, csv_name)
            
            if not os.path.exists(csv_path):
                print(f"⚠️ 跳过缺失文件: {csv_name}")
                continue
            
            # 判断该文件属于训练集还是测试集
            is_train = (filename in TRAIN_FILES)

            # 读取 CSV
            df = pd.read_csv(csv_path)
            # 按时间戳分组构建图
            grouped = df.groupby('Time')

            for time, group in tqdm(grouped, desc=f"  Parsing {filename}", leave=False):
                # 1. 节点特征 (Node Features)
                # 归一化: CN0/50, Doppler/5000
                x_np = group[['CN0_dBHz', 'Doppler']].values.astype(float)
                x_np[:, 0] /= 50.0  
                x_np[:, 1] /= 5000.0
                x = torch.tensor(x_np, dtype=torch.float)
                
                # 2. 过滤掉卫星数太少的帧 (无法构图)
                num_nodes = x.shape[0]
                if num_nodes < 4: 
                    continue
                
                # 3. 构建全连接边并计算物理差值特征
                # # 假设所有可见卫星之间都存在潜在的空间几何关联
                # edge_index = []
                # for i in range(num_nodes):
                #     for j in range(num_nodes):
                #         if i != j:
                #             edge_index.append([i, j])
                
                # edge_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                
                # # 4. 标签与掩码
                # y = torch.tensor([label], dtype=torch.long)
                
                # # 创建图对象
                # data = Data(x=x, edge_index=edge_tensor, y=y)

                # 修改开始
                edge_index = []
                edge_attr = [] # 新增列表存放边特征

                for i in range(num_nodes):
                    for j in range(num_nodes):
                        if i != j:
                            edge_index.append([i, j])
                            
                            # [物理核心] 计算节点 i 和 j 的特征差值
                            # x[i, 0] 是归一化后的 CN0, x[i, 1] 是归一化后的 Doppler
                            diff_cn0 = x[i, 0] - x[j, 0]
                            diff_doppler = x[i, 1] - x[j, 1]
                            
                            # 将差值作为这条边的属性
                            edge_attr.append([diff_cn0, diff_doppler])
                
                edge_index_tensor = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
                edge_attr_tensor = torch.tensor(edge_attr, dtype=torch.float) # 转换为 Tensor
                # 修改结束

                # 4. 标签与掩码
                y = torch.tensor([label], dtype=torch.long)
                
                # 创建图对象
                data = Data(x=x, edge_index=edge_index_tensor, edge_attr=edge_attr_tensor, y=y)

                # 附加元数据 (Metadata)
                data.timestamp = float(time)
                data.train_mask = is_train 
                
                # === 修改 2: 注入场景标签 (Scenario Label) ===
                # 这行代码是你当前缺少的！没有它，我们就没法画出“DS1 vs DS7”的对比图。
                # 我们直接把文件名（去掉后缀）作为场景名存进去
                data.scenario = filename.split('.')[0] 
                
                data_list.append(data)

        if len(data_list) == 0:
            print("❌ 严重错误: 没有构建出任何图数据！请检查 CSV 文件是否为空。")
            return

        # 随机打乱数据 (Shuffle)
        # 注意：虽然我们在内部打乱了，但在训练时我们可以根据 train_mask 再次筛选
        import random
        random.shuffle(data_list)

        print(f"💾 正在保存 {len(data_list)} 个图样本到磁盘...")
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print("✅ 数据集构建完成！")