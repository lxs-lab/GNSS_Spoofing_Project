import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool

class STGraphTransformer(torch.nn.Module):
    def __init__(self, in_channels=2, hidden_channels=64, out_channels=2, edge_dim=2, heads=4, dropout=0.2):
        """
        参数更新:
            edge_dim: 边特征维度 (CN0差值, Doppler差值) -> 2
        """
        super(STGraphTransformer, self).__init__()
        
        # === 1. 图卷积层 (升级版) ===
        # 关键修改: 加入 edge_dim 参数
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, 
                                   dropout=dropout, edge_dim=edge_dim)
        
        # 第二层也需要 edge_dim，因为边特征在传递过程中是保留的
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=1, 
                                   concat=False, dropout=dropout, edge_dim=edge_dim)

        # === 2. 读出层 & 3. 分类器 (保持原结构不变) ===
        self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        
        self.dropout_p = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        """
        前向传播更新: 必须接收 edge_attr
        """
        
        # --- Layer 1 ---
        # 关键修改: 传入 edge_attr
        x = self.conv1(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        # --- Layer 2 ---
        # 关键修改: 传入 edge_attr
        x = self.conv2(x, edge_index, edge_attr)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        # --- Pooling (不变) ---
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1)
        
        # --- Classifier (不变) ---
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)