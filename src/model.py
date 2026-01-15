import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool, BatchNorm

class STGraphTransformer(torch.nn.Module):
    def __init__(self, in_channels=2, hidden_channels=32, out_channels=2, edge_dim=2, heads=2, dropout=0.5):
        """
        升级版模型：加入 BatchNorm 和更少的 Heads
        """
        super(STGraphTransformer, self).__init__()
        # 在模型里加入 BatchNorm，这对于处理 GNSS 这种数值范围波动大的物理信号（CN0, Doppler）有奇效。
        # 1. 输入特征归一化 (关键！防止 Doppler 数值过大影响梯度)
        self.bn_node = BatchNorm(in_channels)
        self.bn_edge = BatchNorm(edge_dim)

        # 2. 图卷积层
        # heads 改为 2 (减少参数量)
        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, 
                                   dropout=dropout, edge_dim=edge_dim)
        
        self.bn1 = BatchNorm(hidden_channels * heads) # 层间归一化
        
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=1, 
                                   concat=False, dropout=dropout, edge_dim=edge_dim)
        
        self.bn2 = BatchNorm(hidden_channels)

        # 3. 全连接层
        self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        
        self.dropout_p = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        # --- Pre-processing ---
        x = self.bn_node(x)           # 先把输入的 CN0/Doppler 归一化
        edge_attr = self.bn_edge(edge_attr) # 把输入的差分特征也归一化
        
        # --- Layer 1 ---
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)               # 归一化
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        # --- Layer 2 ---
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)               # 归一化
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        # --- Pooling ---
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1) 
        
        # --- Classifier ---
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training) # 最后一层也要 Dropout
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)