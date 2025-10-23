import math

import numpy as np
import torch
import torch.nn as nn
import torch.utils.data
import torch.nn.functional as F

def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)

class JointBoneEmbedding(nn.Module):
    """
    Joint + Bone 特征融合模块（优化版）
    ✅ 面向 NTU-RGBD 数据集的高效实现
    - 张量化骨骼计算（无循环）
    - Parent-based Bone 定义（J_i - J_parent(i)）
    - LeakyReLU 激活 + Kaiming 初始化
    """
    def __init__(self, embed_dim=64):
        super(JointBoneEmbedding, self).__init__()

        # -------------------------------
        # NTU RGB+D 骨架的父节点定义 (index 从 0 开始)
        # 每个节点 i 的父节点 parent[i] = j 表示骨骼方向 J_i - J_j
        # 参考 2s-AGCN / CTR-GCN 标准拓扑
        # 节点 0-24 对应人体25个关节
        # -------------------------------
        parent_index = [
            1, 21, 21, 3, 21, 5, 6, 7, 21, 9, 10, 11, 
            1, 13, 14, 15, 1, 17, 18, 19, 2, 8, 8, 12, 12
        ]
        self.parent = torch.tensor(parent_index, dtype=torch.long) - 1  # 转成 0-based 索引

        # -------------------------------
        # 嵌入层：Joint + Bone → 投影到 embed_dim
        # -------------------------------
        self.to_embed = nn.Sequential(
            nn.Conv2d(6, embed_dim, kernel_size=1),
            nn.BatchNorm2d(embed_dim),
            nn.LeakyReLU(0.1)
        )

        self.init_weights()

    # -------------------------------
    # 初始化函数
    # -------------------------------
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='leaky_relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    # -------------------------------
    # 计算骨骼特征 (张量化)
    # -------------------------------
    def compute_bone_feature(self, J):
        """
        Args:
            J: (N, 3, T, V)
        Returns:
            B: (N, 3, T, V)  -> 每个节点对应其骨骼向量
        """
        device = J.device
        parent = self.parent.to(device)  # shape: [V]

        # 通过 parent 索引实现批量张量减法 (无循环)
        J_parent = J[:, :, :, parent]  # (N, 3, T, V)
        B = J - J_parent               # bone vector J_i - J_parent(i)

        # 对骨骼方向归一化（保持方向信息，消除尺度）
        B = F.normalize(B, dim=1)
        return B

    # -------------------------------
    # 前向传播
    # -------------------------------
    def forward(self, J_raw):
        """
        Args:
            J_raw: (N, 3, T, V)
        Returns:
            X_embed: (N, embed_dim, T, V)
        """
        # Step 1: 计算骨骼方向向量
        B_node = self.compute_bone_feature(J_raw)

        # Step 2: 拼接关节与骨骼特征 [3+3=6通道]
        X = torch.cat([J_raw, B_node], dim=1)  # (N, 6, T, V)

        # Step 3: 映射到嵌入空间
        X_embed = self.to_embed(X)  # (N, embed_dim, T, V)

        return X_embed

class HYPERGC(nn.Module):
    """
    SSK + PAS Hypergraph Convolution Module
    核心功能:
    1. SSK (Semantic Subspace K-NN): 自适应生成语义超图 A_sem。
    2. PAS (Prior-Aware Selection): 根据语义置信度，决定是完全回退到先验图，还是与先验图进行软融合。
    3. 动态控制: 通过 forward 参数实现 PAS 启用/禁用的外部课程学习控制。
    """

    def __init__(
        self,
        in_channels, out_channels, vertex_nums, virtual_num, A,
        A_prior=None, use_pas=False, hyper=True, num_subset=8, 
        rel_reduction=4, k_select=9, conf_threshold=0.25
    ):
        super(HYPERGC, self).__init__()
        
        # 模块参数
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.vertex_nums = vertex_nums
        self.virtual_num = virtual_num
        self.rel_reduction = rel_reduction
        self.num_subset = num_subset  # 语义子空间/注意力头数 (H)
        self.hyper = hyper
        self.k_select = k_select      # Top-K 选择数 (K)
        self.use_pas = use_pas        # 模块默认 PAS 状态 (初始化时)
        self.conf_threshold = conf_threshold # PAS 回退阈值
        
        # --- SSK Core: 自适应超图生成 ---
        if self.hyper:
            self.head_dim = in_channels // rel_reduction
            # Q, K 投影 (多头)
            self.to_Q = nn.Conv1d(in_channels, num_subset * self.head_dim, kernel_size=1, groups=num_subset)
            self.to_K = nn.Conv1d(in_channels, num_subset * self.head_dim, kernel_size=1, groups=num_subset)
            
            # Hyperedge Confidence (W) - 用于多头融合权重 omega_h
            self.to_W = nn.Sequential(
                nn.Conv1d(in_channels, num_subset * self.head_dim, kernel_size=1, groups=num_subset),
                nn.LeakyReLU(),
                nn.Conv1d(num_subset * self.head_dim, num_subset, kernel_size=1),
                nn.Tanh()
            )
            self.conf_gate = nn.Parameter(torch.ones(1, num_subset, 1, 1))
            self.softmax_h = nn.Softmax(dim=-1)
            self.softmax_gate = nn.Softmax(dim=1)
            self.alpha = nn.Parameter(torch.ones(1)) # 拓扑融合的学习权重
            self.scale = self.head_dim ** -0.5

        # --- Graph Priors ---
        # A: 可学习的物理结构先验图 (A_learn)
        self.PA = nn.Parameter(torch.from_numpy(A.astype(np.float32)), requires_grad=True)
        self.edge_importance = nn.Parameter(torch.ones(A.shape))
        # A_prior: 结构锚定的先验图 (用于 PAS)
        if A_prior is not None:
            if isinstance(A_prior, torch.Tensor):
                self.A_prior = A_prior.float()  # 如果已经是Tensor，确保是float32
            else:
                self.A_prior = torch.from_numpy(A_prior.astype(np.float32))  # NumPy数组转Tensor
        else:
            self.A_prior = None

        # --- Conv Transform and Residual ---
        self.conv_d = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        if in_channels != out_channels:
            self.down = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.down = lambda x: x
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        
        # 初始化
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

    # -----------------------------
    # PAS配置获取方法
    # -----------------------------
    def get_pas_config(self, epoch):
        """
        基于epoch的PAS控制策略
        Args:
            epoch: 当前训练epoch
        Returns:
            dict: PAS配置参数
        """
        if epoch <= 30:
            # 训练初期 (Epochs 1-30): 启用PAS，保守策略
            return {
                'use_pas': True,
                'conf_threshold': 0.15,  # 较低阈值，更容易触发回退
                'sem_weight': 0.3,       # 较少信任语义超图
                'prior_weight': 0.7      # 更多依赖结构先验
            }
        elif epoch <= 80:
            # 训练中期 (Epochs 31-80): 启用PAS，平衡策略
            return {
                'use_pas': True,
                'conf_threshold': 0.25,   # 标准阈值
                'sem_weight': 0.6,       # 逐渐增加对语义超图的信任
                'prior_weight': 0.4      # 减少对结构先验的依赖
            }
        else:
            # 训练后期 (Epochs 81-140): 禁用PAS，激进策略
            return {
                'use_pas': False,        # 完全禁用PAS
                'conf_threshold': 0.25,  # 保持阈值（虽然不会使用）
                'sem_weight': 1.0,       # 完全信任语义超图
                'prior_weight': 0.0      # 不再使用结构先验
            }

    # -----------------------------
    # 归一化工具 (行和为1)
    # -----------------------------
    def a_norm(self, A):
        # 归一化 A，使其每行（即每个节点的出度）和为1
        d_r = torch.norm(A, 1, dim=2, keepdim=True) + 1e-8
        return A / d_r

    # -----------------------------
    # 前向传播
    # -----------------------------
    def forward(self, x, use_pas_override=None, blend_sem_weight=None, blend_prior_weight=None, epoch=None):
        N, C, T, V = x.size()
        device = x.device
        
        # 1. 基于epoch的动态PAS控制策略
        if epoch is not None:
            pas_config = self.get_pas_config(epoch)
            pas_active = pas_config['use_pas']
            blend_sem_weight = pas_config['sem_weight']
            blend_prior_weight = pas_config['prior_weight']
            conf_threshold = pas_config['conf_threshold']
        else:
            # 兼容原有接口
            pas_active = self.use_pas if use_pas_override is None else use_pas_override
            conf_threshold = self.conf_threshold

        # 2. Learnable Prior (A_learn) - 可学习的物理拓扑
        A_base = (self.edge_importance * self.PA.to(device))
        
        # 确保A_base是2D张量
        if A_base.dim() == 1:
            # 如果是1D，尝试reshape为2D
            A_base = A_base.view(int(A_base.size(0)**0.5), -1)
        elif A_base.dim() == 3:
            # 如果是3D，取第一个矩阵 (假设是ensemble的情况)
            A_base = A_base[0]  # 取第一个25x25矩阵
        elif A_base.dim() > 3:
            # 如果是更高维，取前两维
            A_base = A_base.view(A_base.size(0), A_base.size(1))
        
        # A_learn: (N, V, V)
        A_learn = self.a_norm(A_base.unsqueeze(0).repeat(N, 1, 1))

        # --- Adaptive SSK Hypergraph Generation ---
        if self.hyper:
            t_x = x.mean(2) # (N, C, V) - 时间维度平均
            
            # Q, K 投影和维度重排
            Q = self.to_Q(t_x).view(N, self.num_subset, self.head_dim, V).permute(0, 1, 3, 2) # (N, H, V, d_k)
            K = self.to_K(t_x).view(N, self.num_subset, self.head_dim, V).permute(0, 1, 3, 2) # (N, H, V, d_k)

            A_h = torch.matmul(Q, K.transpose(-1, -2)) * self.scale # (N, H, V, V)

            # Top-K 稀疏化和 Softmax
            topk_v, topk_indices = torch.topk(A_h, self.k_select, dim=-1)
            H_mask = torch.zeros_like(A_h, device=device).scatter_(-1, topk_indices, 1)
            
            # 正确处理 Softmax Masking：非 Top-K 设为负无穷
            H_masked = A_h.masked_fill(H_mask == 0, -float('inf')) 
            H_h_sparse = self.softmax_h(H_masked)
            H_h_sparse = H_h_sparse.masked_fill(H_mask == 0, 0) # 确保非 Top-K 元素为零
            
            # Confidence weighting and Multi-Head Fusion
            W_raw = self.to_W(t_x).mean(-1).unsqueeze(-1).unsqueeze(-1) # (N, H, 1, 1)
            omega_h = self.softmax_gate(self.conf_gate.repeat(N, 1, 1, 1) + W_raw) 
            
            H_sem_weighted = H_h_sparse * omega_h
            H_sem = H_sem_weighted.sum(dim=1)
            A_sem = self.a_norm(H_sem) # 规范化语义超图 A_sem (N, V, V)

            # --- PAS mechanism ---
            if pas_active and (self.A_prior is not None):
                A_prior = self.A_prior.to(device)
                # 规范化 A_prior 并扩展到批次维度
                A_prior_norm = self.a_norm(A_prior.unsqueeze(0).repeat(N, 1, 1)) 

                # 计算全局置信度 (基于融合权重W的平均绝对值)
                conf_value = torch.mean(torch.abs(W_raw)).item() 
                
                # 获取动态混合权重，如果未提供则使用默认值
                sem_weight = blend_sem_weight if blend_sem_weight is not None else 0.8
                prior_weight = blend_prior_weight if blend_prior_weight is not None else 0.2

                if conf_value < conf_threshold:
                    # Fallback: 置信度低，完全替换为结构锚定 A_prior
                    A_sem = A_prior_norm * 1.0 + 0.0 * A_sem
                else:
                    # Soft Blend: 使用动态/固定权重进行混合
                    A_sem = blend_sem_weight * A_sem + blend_prior_weight * A_prior_norm

        # --- 3. Topology Fusion ---
        # A_fused = A_learn + ReLU(alpha) * A_sem
        A_fused = A_learn + self.relu(self.alpha) * A_sem # A_learn 和 A_sem 都是 (N, V, V)

        # --- 4. Hypergraph Convolution Operation ---
        d_x = self.conv_d(x) # (N, C_out, T, V)
        
        # 将输入特征 (N, C_out, T, V) 调整为 (N, T, C_out, V)
        d_x_flat = d_x.permute(0, 2, 1, 3).contiguous() 
        
        # Hypergraph Convolution: X' = A * X
        # einsum: (N, T, C_out, V) * (N, V, V) -> (N, T, C_out, V)
        y = torch.einsum('ntcv,nuv->ntcu', d_x_flat, A_fused).contiguous() 
        
        # 恢复为 (N, C_out, T, V)
        y = y.permute(0, 2, 1, 3).contiguous() 

        # --- 5. Residual and Activation ---
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        
        return y, A_fused # 返回 A_fused 方便调试

class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size-1) * (dilation-1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1),
            padding_mode='replicate')

        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class MultiScale_TemporalConv(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 dilations=[1,2,3,4],
                 residual=True,
                 residual_kernel_size=1):

        super().__init__()
        assert out_channels % (len(dilations) + 2) == 0, '# out channels should be multiples of # branches'

        # Multiple branches of temporal convolution
        self.num_branches = len(dilations) + 2
        branch_channels = out_channels // self.num_branches
        branch_mid_channels = out_channels - branch_channels * (self.num_branches - 1)
        if type(kernel_size) == list:
            assert len(kernel_size) == len(dilations)
        else:
            kernel_size = [kernel_size]*len(dilations)
        # Temporal Convolution branches
        self.branches = nn.ModuleList()
        for ks, dilation in zip(kernel_size, dilations):
            self.branches.append(nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        branch_channels,
                        kernel_size=1,
                        padding=0),
                    nn.BatchNorm2d(branch_channels),
                    nn.ReLU(inplace=False),
                    TemporalConv(
                        branch_channels,
                        branch_channels,
                        kernel_size=ks,
                        stride=stride,
                        dilation=dilation),
                )
            )

        # Additional Max & 1x1 branch
        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=False),
            nn.MaxPool2d(kernel_size=(3, 1), stride=(stride, 1), padding=(1, 0)),
            nn.BatchNorm2d(branch_channels),  # 为什么还要加bn
        ))

        self.branches.append(nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, kernel_size=1, padding=0, stride=(stride,1)),
            nn.BatchNorm2d(branch_channels)
        ))

        # Residual connection
        if not residual:
            self.residual = lambda x: 0
        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x
        else:
            self.residual = TemporalConv(in_channels, out_channels, kernel_size=residual_kernel_size, stride=stride)

        # initialize
        self.apply(weights_init)

    def forward(self, x):
        # Input dim: (N,C,T,V)
        res = self.residual(x)
        branch_outs = []
        for tempconv in self.branches:
            out = tempconv(x)
            branch_outs.append(out)
        out = torch.cat(branch_outs, dim=1)
        out += res
        return out


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=(kernel_size, 1), padding=(pad, 0),
                              stride=(stride, 1), padding_mode='replicate')

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=False)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class TCN_GCN_unit(nn.Module):
    def __init__(self, in_channels, out_channels, num_point, hyper_joints, A, stride=1, residual=True, kernel_size=5, dilations=[1, 2], hyper=True, A_prior=None):
        super(TCN_GCN_unit, self).__init__()
        self.gcn1 = HYPERGC(in_channels, out_channels, num_point, hyper_joints, A, A_prior=A_prior, hyper=hyper)
        self.tcn1 = MultiScale_TemporalConv(out_channels, out_channels, kernel_size=kernel_size, stride=stride, dilations=dilations,
                                        residual=False)
        self.relu = nn.ReLU(inplace=False)
        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=stride)

    def forward(self, x, epoch=None):
        y, _ = self.gcn1(x, epoch=epoch)  # 传递epoch参数
        y = self.relu(self.tcn1(y) + self.residual(x))
        return y, None  # 不再返回hyper_joint


class Model(nn.Module):
    def __init__(self, num_class=60, num_point=25, num_person=2, graph=None, graph_args=dict(), in_channels=3, hyper_joints=0,
                 drop_out=0):
        super(Model, self).__init__()

        if graph is None:
            raise ValueError()
        else:
            Graph = import_class(graph)
            self.graph = Graph(hyper_joints, **graph_args)

        A = self.graph.A
        A_prior = self.graph.A_prior  # 获取预定义超图先验
        self.num_class = num_class
        self.num_point = num_point
        self.embedding_channels = 128

        self.data_bn = nn.BatchNorm1d(num_person * self.embedding_channels * num_point)
        self.joint_bone_embedding = JointBoneEmbedding(embed_dim=self.embedding_channels)
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_point, self.embedding_channels))
        self.tanh = nn.Tanh()

        self.l1 = TCN_GCN_unit(self.embedding_channels, self.embedding_channels, num_point, hyper_joints, A, A_prior=A_prior)
        self.l2 = TCN_GCN_unit(self.embedding_channels, self.embedding_channels, num_point, hyper_joints, A, A_prior=A_prior)
        self.l3 = TCN_GCN_unit(self.embedding_channels, self.embedding_channels, num_point, hyper_joints, A, A_prior=A_prior)
        self.l4 = TCN_GCN_unit(self.embedding_channels, self.embedding_channels * 2, num_point, hyper_joints, A, stride=2, A_prior=A_prior)
        self.l5 = TCN_GCN_unit(self.embedding_channels * 2, self.embedding_channels * 2, num_point, hyper_joints, A, A_prior=A_prior)
        self.l6 = TCN_GCN_unit(self.embedding_channels * 2, self.embedding_channels * 2, num_point, hyper_joints, A, A_prior=A_prior)
        self.l7 = TCN_GCN_unit(self.embedding_channels * 2, self.embedding_channels * 2, num_point, hyper_joints, A, stride=2, A_prior=A_prior)
        self.l8 = TCN_GCN_unit(self.embedding_channels * 2, self.embedding_channels * 2, num_point, hyper_joints, A, A_prior=A_prior)
        self.l9 = TCN_GCN_unit(self.embedding_channels * 2, self.embedding_channels * 2, num_point, hyper_joints, A, A_prior=A_prior)
        self.fc = nn.Linear(self.embedding_channels * 2, self.num_class)

        nn.init.normal_(self.fc.weight, 0, math.sqrt(2. / num_class))
        bn_init(self.data_bn, 1)
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

    def forward(self, x, epoch=None):
        N, C, T, V, M = x.size()
        x = x.permute(0, 4, 2, 3, 1).contiguous()
        x = x.view(N * M, T, V, C).permute(0, 3, 1, 2).contiguous()  # (N*M, 3, T, V)
        x = self.joint_bone_embedding(x) 
        x = self.data_bn(x)
        x += self.pos_embedding[:, :self.num_point]
        x = self.tanh(x)
        x = x.permute(0, 1, 3, 4, 2).contiguous()

        x = x.view(N, M * V * self.embedding_channels, T)
        x = self.data_bn(x)
        x = x.view(N, M, V, self.embedding_channels, T).permute(0, 1, 3, 4, 2).contiguous()

        x = x.view(N * M, self.embedding_channels, T, V)

        # 传递epoch参数到所有TCN_GCN_unit层
        x, _ = self.l1(x, epoch=epoch)
        x1 = x
        x, _ = self.l2(x, epoch=epoch)
        x, _ = self.l3(x + x1, epoch=epoch)

        x, _ = self.l4(x, epoch=epoch)
        x4 = x
        x, _ = self.l5(x, epoch=epoch)
        x, _ = self.l6(x + x4, epoch=epoch)

        x, _ = self.l7(x, epoch=epoch)
        x7 = x
        x, _ = self.l8(x, epoch=epoch)
        x, _ = self.l9(x + x7, epoch=epoch)

        # N*M,C,T,V
        c_new = x.size(1)
        x = x.view(N, M, c_new, -1)
        x = x.mean(3).mean(1)
        x = self.drop_out(x)

        return self.fc(x), []  # 不再返回hyper_joint列表


if __name__ == '__main__':
    model = Model(graph='graph.ntu_rgb_d.Graph', hyper_joints=3, graph_args={'labeling_mode':'virtual_ensemble'}).to('cuda:0')
    x = torch.randn(2, 3, 64, 25, 2).to('cuda:0')
    y = model(x)
    print()