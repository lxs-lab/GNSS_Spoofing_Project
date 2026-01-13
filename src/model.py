import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool

class STGraphTransformer(torch.nn.Module):
    def __init__(self, in_channels=1, hidden_channels=64, out_channels=2, edge_dim=2, heads=4, dropout=0.2):
        """
        时空图 Transformer 模型 (Spatio-Temporal Graph Transformer)
        参数更新:
            in_channels: 1 (只保留 CN0，去除绝对 Doppler)
            edge_dim: 2 (边特征: CN0差值, Doppler差值)
        """
        super(STGraphTransformer, self).__init__()
        
        # === 1. 图卷积层 (支持边特征) ===
        # 第一层: 接收节点特征和边特征
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, 
                                   dropout=dropout, edge_dim=edge_dim)
        
        # 第二层: 聚合全局信息
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=1, 
                                   concat=False, dropout=dropout, edge_dim=edge_dim)

        # === 2. 读出层 & 分类器 ===
        # 将一张图里的所有卫星特征聚合成一个向量
        self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        
        self.dropout_p = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        """
        前向传播: 必须传入 edge_attr
        """
        
        # --- Layer 1 ---
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        # --- Layer 2 ---
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        # --- Pooling (Graph Level Embedding) ---
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1) 
        
        # --- Classifier ---
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)