import torch
import torch.nn.functional as F
from torch_geometric.nn import TransformerConv, global_mean_pool, global_max_pool, BatchNorm


class STGraphTransformer(torch.nn.Module):
    """
    Spatio-Temporal Graph Transformer for GNSS spoofing detection.

    Fix log: removed F.log_softmax from forward() — CrossEntropyLoss already
    applies log_softmax internally. Passing log_softmax output into
    CrossEntropyLoss causes double-compression of gradients and prevents
    the model from learning.  Use NLLLoss if you want log_softmax back.
    """

    def __init__(self, in_channels=2, hidden_channels=64, out_channels=2,
                 edge_dim=2, heads=4, dropout=0.2):
        super().__init__()

        # Input normalisation
        self.bn_node = BatchNorm(in_channels)
        self.bn_edge = BatchNorm(edge_dim)

        # Layer 1: multi-head, output dim = hidden * heads
        self.conv1 = TransformerConv(
            in_channels,
            hidden_channels,
            heads=heads,
            dropout=dropout,
            edge_dim=edge_dim,
            beta=True,          # residual gating — improves stability
        )
        self.bn1 = BatchNorm(hidden_channels * heads)

        # Layer 2: single head, reduces back to hidden_channels
        self.conv2 = TransformerConv(
            hidden_channels * heads,
            hidden_channels,
            heads=1,
            concat=False,
            dropout=dropout,
            edge_dim=edge_dim,
            beta=True,
        )
        self.bn2 = BatchNorm(hidden_channels)

        # Classifier head (mean + max pooling → concat → MLP)
        self.lin1 = torch.nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = torch.nn.Linear(hidden_channels, out_channels)

        self.dropout_p = dropout

    def forward(self, x, edge_index, edge_attr, batch):
        # --- Normalise inputs ---
        x = self.bn_node(x)
        edge_attr = self.bn_edge(edge_attr)

        # --- Spatial GNN layers ---
        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.elu(x)                          # ELU keeps negative gradients alive
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        x = self.conv2(x, edge_index, edge_attr)
        x = self.bn2(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)

        # --- Readout: dual pooling ---
        x_mean = global_mean_pool(x, batch)   # captures average constellation state
        x_max  = global_max_pool(x, batch)    # captures the worst anomaly in the graph
        x = torch.cat([x_mean, x_max], dim=1)

        # --- MLP classifier ---
        x = self.lin1(x)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout_p, training=self.training)
        x = self.lin2(x)

        # *** KEY FIX ***
        # Return raw logits. CrossEntropyLoss = log_softmax + NLLLoss internally.
        # Do NOT apply log_softmax here — it was the primary cause of training failure.
        return x

    def forward_with_attention(self, x, edge_index, edge_attr, batch):
        """
        Inference-only forward pass that also returns attention weights
        from layer 2, enabling XAI (satellite-level anomaly localisation).
        Returns: (logits, alpha_dict)  where alpha_dict = {edge_index, alpha}
        """
        x = self.bn_node(x)
        edge_attr = self.bn_edge(edge_attr)

        x = self.conv1(x, edge_index, edge_attr)
        x = self.bn1(x)
        x = F.elu(x)

        out = self.conv2(x, edge_index, edge_attr,
                         return_attention_weights=True)
        x_conv, (att_edge_index, att_weights) = out

        x_conv = self.bn2(x_conv)
        x_conv = F.elu(x_conv)

        x_mean = global_mean_pool(x_conv, batch)
        x_max  = global_max_pool(x_conv, batch)
        x_pool = torch.cat([x_mean, x_max], dim=1)

        logits = self.lin2(F.elu(self.lin1(x_pool)))

        return logits, {'edge_index': att_edge_index, 'alpha': att_weights}