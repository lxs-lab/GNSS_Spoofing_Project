import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool

class STGraphTransformer(torch.nn.Module):
    def __init__(self, in_channels=2, hidden_channels=64, out_channels=2, heads=4, dropout=0.2):
        """
        时空图 Transformer 模型 (Spatio-Temporal Graph Transformer)
        参数:
            in_channels: 节点特征维度 (CN0, Doppler) -> 2
            hidden_channels: 隐层维度 -> 64
            heads: 多头注意力的头数 -> 4 (让模型从不同角度看数据)
        """
        super(STGraphTransformer, self).__init__()
        
        # === 1. 图卷积层 (提取空间特征) ===
        # 第一层: 将低维特征映射到高维，寻找局部关联
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        
        # 第二层: 深层聚合，捕捉全局星座异常
        # 输入维度是 hidden_channels * heads (因为多头拼接了)
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=1, concat=False, dropout=dropout)

        # === 2. 读出层 (Readout) ===
        # 将一张图里的所有卫星特征聚合成一个向量
        # 我们同时使用 Mean 和 Max Pooling，既看整体(Mean)也看极端异常(Max)
        # 这是一个提升效果的小 Trick
        
        # === 3. 全连接层 (分类器) ===
        # 输入维度 x2 是因为拼接了 mean 和 max
        self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        
        self.dropout_p = dropout

    def forward(self, x, edge_index, batch):
        """
        x: [Num_Nodes, 2] 节点特征
        edge_index: [2, Num_Edges] 边连接
        batch: [Num_Nodes] 批次索引 (指示每个点属于哪张图)
        """
        
        # --- Layer 1 ---
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        # --- Layer 2 ---
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        # --- Pooling (Graph Level Embedding) ---
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1) # 融合特征
        
        # --- Classifier ---
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)