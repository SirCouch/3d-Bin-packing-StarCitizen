import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from torch_geometric.nn import global_mean_pool, GATConv

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Node type constants (must match DRLBinPackingEnv)
NODE_TYPE_CONTAINER = 0
NODE_TYPE_PLACED = 1
NODE_TYPE_ITEM = 2
NODE_TYPE_MER = 3


class SharedGNNBackbone(nn.Module):
    """Shared 3-layer GAT backbone used by both actor and critic."""

    def __init__(self, node_feature_dim, hidden_dim=128, dropout_rate=0.2):
        super().__init__()
        self.conv1 = GATConv(node_feature_dim, hidden_dim // 4, heads=4, dropout=dropout_rate)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.dropout1 = nn.Dropout(dropout_rate)

        self.conv2 = GATConv(hidden_dim, hidden_dim // 4, heads=4, dropout=dropout_rate)
        self.bn2 = nn.BatchNorm1d(hidden_dim)
        self.dropout2 = nn.Dropout(dropout_rate)

        self.conv3 = GATConv(hidden_dim, hidden_dim, heads=1, dropout=dropout_rate)
        self.bn3 = nn.BatchNorm1d(hidden_dim)
        self.dropout3 = nn.Dropout(dropout_rate)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)

        x = self.conv2(x, edge_index)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)

        x = self.conv3(x, edge_index)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        return x


class ActorGNN(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim=128, dropout_rate=0.2, backbone=None):
        super().__init__()
        self.backbone = backbone or SharedGNNBackbone(node_feature_dim, hidden_dim, dropout_rate)
        self.hidden_dim = hidden_dim

        # Cross-attention: item embedding queries MER embeddings
        self.query_proj = nn.Linear(hidden_dim, hidden_dim)
        self.key_proj = nn.Linear(hidden_dim, hidden_dim)

        # Score head: outputs 2 logits per MER (one per rotation)
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, data, feasibility_mask=None):
        if hasattr(data, 'x') and data.x.device != device:
            data = data.to(device)

        x = self.backbone(data.x, data.edge_index)
        node_type = data.node_type

        mer_mask = (node_type == NODE_TYPE_MER)
        item_mask = (node_type == NODE_TYPE_ITEM)

        if not torch.any(mer_mask):
            return Categorical(logits=torch.zeros(1, device=x.device))

        mer_embeddings = x[mer_mask]  # [num_mers, H]

        if torch.any(item_mask):
            # Cross-attention: item queries, MER keys
            item_embed = x[item_mask].mean(dim=0, keepdim=True)  # [1, H]
            query = self.query_proj(item_embed)   # [1, H]
            keys = self.key_proj(mer_embeddings)  # [num_mers, H]

            # Attention-weighted MER embeddings
            attn_weights = torch.matmul(query, keys.T) / (self.hidden_dim ** 0.5)  # [1, num_mers]
            attn_weights = F.softmax(attn_weights, dim=-1)
            # Modulate MER embeddings with attention
            attended = mer_embeddings * attn_weights.T  # [num_mers, H] element-wise scaling
            mer_scores = self.score_head(mer_embeddings + attended)  # residual
        else:
            mer_scores = self.score_head(mer_embeddings)

        mer_scores = mer_scores.view(-1)  # [num_mers * 2]

        if feasibility_mask is not None:
            mask_tensor = torch.BoolTensor(feasibility_mask).to(device)
            mer_scores = mer_scores.masked_fill(~mask_tensor, float('-inf'))

        return Categorical(logits=mer_scores)


class CriticGNN(nn.Module):
    def __init__(self, node_feature_dim, hidden_dim=128, dropout_rate=0.2, backbone=None):
        super().__init__()
        self.backbone = backbone or SharedGNNBackbone(node_feature_dim, hidden_dim, dropout_rate)

        self.value_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, data):
        if hasattr(data, 'x') and data.x.device != device:
            data = data.to(device)

        x = self.backbone(data.x, data.edge_index)

        batch_vec = data.batch if hasattr(data, 'batch') and data.batch is not None else torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        global_features = global_mean_pool(x, batch_vec)

        return self.value_head(global_features)
