import torch
import torch.nn as nn
import torch.nn.functional as F

class EdgeConv(nn.Module):

    def __init__(self, in_channels, out_channels, k=3):
        super(EdgeConv, self).__init__()
        self.k = k

        self.conv = nn.Sequential(
            nn.Conv2d(2 * in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.SiLU(inplace=True)
        )

    def forward(self, x, edge_index):
        """
        参数：
        - x: [num_nodes, in_channels]
        - edge_index: [2, num_edges]
        返回：
        - x_out: [num_nodes, out_channels]
        """
        num_nodes = x.size(0)
        device = x.device

        # 获取源节点和目标节点的索引
        row, col = edge_index

        # 获取源节点和目标节点的特征
        x_i = x[row]  # [num_edges, in_channels]
        x_j = x[col]  # [num_edges, in_channels]

        # 计算边特征：e_ij = concat(x_i, x_j - x_i)
        edge_features = torch.cat((x_i, x_j - x_i), dim=1)  # [num_edges, 2 * in_channels]

        # Reshape 为卷积操作准备
        edge_features = edge_features.unsqueeze(-1).unsqueeze(-1)  # [num_edges, 2 * in_channels, 1, 1]

        # 通过卷积层
        edge_features = self.conv(edge_features)  # [num_edges, out_channels, 1, 1]
        edge_features = edge_features.squeeze(-1).squeeze(-1)  # [num_edges, out_channels]

        # 初始化输出
        x_out = torch.zeros(num_nodes, edge_features.size(1), device=device)  # [num_nodes, out_channels]

        # 聚合邻域信息（累加操作）
        x_out = x_out.index_add_(0, row, edge_features)  # [num_nodes, out_channels]

        return x_out

class AttentionModule(nn.Module):
    """
    注意力模块，对节点特征进行加权。
    """
    def __init__(self, in_channels, num_layers, keypoint_groups):
        super(AttentionModule, self).__init__()
        self.num_layers = num_layers
        self.keypoint_groups = keypoint_groups

        inter_channels = in_channels // 4

        self.conv_down = nn.Sequential(
            nn.Linear(in_channels, inter_channels),
            nn.BatchNorm1d(inter_channels),
            nn.SiLU(inplace=True)
        )

        self.edge_conv = EdgeConv(inter_channels, inter_channels, k=3)

        self.aggregate = nn.Linear(inter_channels, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        参数：
        - x: [num_nodes, in_channels]
        返回：
        - out: [num_nodes, in_channels]
        """
        # 降维
        x_down = self.conv_down(x)  # [num_nodes, inter_channels]

        # 对每个关键点组进行采样和聚合
        x_sampled = []
        for group in self.keypoint_groups:
            group_features = x_down[group]  # [group_size, inter_channels]
            group_mean = group_features.mean(dim=0, keepdim=True)  # [1, inter_channels]
            x_sampled.append(group_mean)
        x_sampled = torch.cat(x_sampled, dim=0)  # [num_groups, inter_channels]

        # 通过 EdgeConv
        edge_index = self.build_fully_connected_edge_index(x_sampled.size(0), x.device)
        att = self.edge_conv(x_sampled, edge_index)  # [num_groups, inter_channels]

        # 聚合
        att = self.aggregate(att)  # [num_groups, in_channels]
        att = self.sigmoid(att)  # [num_groups, in_channels]

        # 将注意力权重应用到原始特征上
        out = x * att.mean(dim=0, keepdim=True)  # [num_nodes, in_channels]

        return out

    def build_fully_connected_edge_index(self, num_nodes, device):
        """
        构建全连接的 edge_index。
        """
        row = torch.arange(num_nodes, device=device).unsqueeze(1).repeat(1, num_nodes).view(-1)
        col = torch.arange(num_nodes, device=device).unsqueeze(0).repeat(num_nodes, 1).view(-1)
        edge_index = torch.stack([row, col], dim=0)
        return edge_index
    
class GHAGCNBlockModule(nn.Module):
    """
    高维图卷积模块，包含多层图卷积和注意力机制。
    """
    def __init__(self, gcn_cfg, keypoint_connections):
        super(GHAGCNBlockModule, self).__init__()
        self.num_layers = gcn_cfg['num_layers']
        self.hidden_dim = gcn_cfg['hidden_dim']
        self.input_dim = gcn_cfg['input_dim']
        self.k = gcn_cfg.get('k', 3)  # 邻居数，默认3
        # 存储关键点名称和连接关系
        self.keypoint_connections = keypoint_connections

        # 初始化自适应邻接矩阵
        num_edges = len(keypoint_connections)
        self.A = nn.Parameter(torch.randn(self.num_layers, num_edges), requires_grad=True)

        # 定义输入降维层
        self.conv_input = nn.Linear(self.input_dim, self.hidden_dim)
        self.bn_input = nn.BatchNorm1d(self.hidden_dim)

        # 定义图卷积层
        self.edgeconv_layers = nn.ModuleList()
        self.bn_layers = nn.ModuleList()
        self.attention_modules = nn.ModuleList()
        for i in range(self.num_layers):
            self.edgeconv_layers.append(EdgeConv(self.hidden_dim, self.hidden_dim, k=self.k))
            self.attention_modules.append(AttentionModule(self.hidden_dim, self.num_layers, self.get_keypoint_groups()))
            self.bn_layers.append(nn.BatchNorm1d(self.hidden_dim))

        # 注意力模块
        self.attention_module = AttentionModule(self.hidden_dim, self.num_layers, self.get_keypoint_groups())


    def get_keypoint_groups(self):
        """
        根据关键点连接关系，构建关键点组。
        """
        groups = []
        for src_idx, dst_idx in self.keypoint_connections:
            groups.append([src_idx, dst_idx])
        return groups

    def build_edge_index(self, valid_indices, device):
        """
        构建边的索引 edge_index。
        """
        # 确保 valid_indices 是集合以加速查找
        valid_indices_set = set(valid_indices)  # 将 valid_indices 转换为集合
        edge_index = []
        for src_idx, dst_idx in self.keypoint_connections:
            if src_idx in valid_indices_set and dst_idx in valid_indices_set:
                edge_index.append([src_idx, dst_idx])
                edge_index.append([dst_idx, src_idx])  # 无向图

        if len(edge_index) == 0:
            # 如果没有边，创建自环
            num_nodes = len(valid_indices)
            edge_index = [[i, i] for i in range(num_nodes)]

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous().to(device)  # [2, num_edges]
        return edge_index

    def forward(self, keypoint_embeddings):
        """
        参数：
        - keypoint_embeddings: [B, K, C]
        - keypoint_masks: [B, K]
        返回：
        - x_out: [B, K, hidden_dim]
        """
        B, K, C = keypoint_embeddings.shape  # [B, K, C]
        device = keypoint_embeddings.device
        x_out = torch.zeros(B, K, self.hidden_dim, device=device)  # [B, K, hidden_dim]

        for b in range(B):
            # 获取有效的关键点
            node_features = keypoint_embeddings[b]  # [K, C]

            x = self.conv_input(node_features)  # [num_nodes, hidden_dim]
            x = self.bn_input(x)
            x = F.silu(x)

            edge_index = self.build_edge_index(range(K), device)

            # 通过多层 EdgeConv
            for i, (ec, attn, bn) in enumerate(zip(self.edgeconv_layers, self.attention_modules, self.bn_layers)):
                x_residual = x
                x = ec(x, edge_index)  # [num_nodes, hidden_dim]
                x = bn(x)
                x = attn(x)
                x = x + x_residual
                x = F.silu(x)

            x_out[b] = x

        return x_out  # [B, K, hidden_dim]
