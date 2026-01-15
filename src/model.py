import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool

class STGraphTransformer(torch.nn.Module):
    # 修改 1: 默认 in_channels 改回 2 (CN0 + Doppler)，这是效果最好的设置
    def __init__(self, in_channels=2, hidden_channels=64, out_channels=2, edge_dim=2, heads=4, dropout=0.2):
        """
        时空图 Transformer 模型 (Spatio-Temporal Graph Transformer)
        参数:
            in_channels: 2 (CN0, Doppler)
            edge_dim: 2 (CN0差值, Doppler差值)
        """
        super(STGraphTransformer, self).__init__()
        
        # === 1. 图卷积层 (支持边特征) ===
        # 这一行就是报错的源头：如果你旧代码里没有 edge_dim，这里就会崩
        # 现在的版本明确加入了 edge_dim=edge_dim
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, 
                                   dropout=dropout, edge_dim=edge_dim)
        
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=1, 
                                   concat=False, dropout=dropout, edge_dim=edge_dim)

        # === 2. 读出层 & 分类器 ===
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
        
        # --- Pooling ---
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1) 
        
        # --- Classifier ---
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)