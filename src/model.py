import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool, BatchNorm

class STGraphTransformer(torch.nn.Module):
    # [修改] 默认 heads 改回 4
    def __init__(self, in_channels=2, hidden_channels=64, out_channels=2, edge_dim=2, heads=4, dropout=0.2):
        super(STGraphTransformer, self).__init__()
        
        self.bn_node = BatchNorm(in_channels)
        self.bn_edge = BatchNorm(edge_dim)

        self.conv1 = TransformerConv(in_channels, hidden_channels, heads=heads, 
                                   dropout=dropout, edge_dim=edge_dim)
        
        self.bn1 = BatchNorm(hidden_channels * heads)
        
        self.conv2 = TransformerConv(hidden_channels * heads, hidden_channels, heads=1, 
                                   concat=False, dropout=dropout, edge_dim=edge_dim)
        
        self.bn2 = BatchNorm(hidden_channels)

        self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)
        
        self.dropout_p = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        x = self.bn_node(x)
        edge_attr = self.bn_edge(edge_attr)
        
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        
        x_mean = global_mean_pool(x, batch)
        x_max = global_max_pool(x, batch)
        x = torch.cat([x_mean, x_max], dim=1) 
        
        x = self.lin1(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)
        
        return F.log_softmax(x, dim=1)