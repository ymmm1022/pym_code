# import sys

# sys.path.extend(['../'])
# from graph import tools

# num_node = 25
# self_link = [(i, i) for i in range(num_node)]
# inward_ori_index = [(1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
#                     (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
#                     (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
#                     (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)]
# inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
# outward = [(j, i) for (i, j) in inward]
# neighbor = inward + outward

# class Graph:
#     def __init__(self, hyper_joints=0, labeling_mode='spatial'):
#         self.num_node = num_node
#         self.self_link = self_link
#         self.inward = inward
#         self.outward = outward
#         self.neighbor = neighbor
#         self.hyper_joints = hyper_joints
#         self.A = self.get_adjacency_matrix(labeling_mode)


#     def get_adjacency_matrix(self, labeling_mode=None):
#         if labeling_mode is None:
#             return self.A
#         if labeling_mode == 'spatial':
#             A = tools.get_spatial_graph(num_node, self_link, inward, outward)
#         elif labeling_mode == 'spatial_ensemble':
#             A = tools.get_spatial_graph_ensemble(num_node, self_link, inward, outward, 8)
#         elif labeling_mode == 'virtual_ensemble':
#             A = tools.get_virtual_graph_ensemble(num_node, self_link, inward, outward, self.hyper_joints, 8)
#         else:
#             raise ValueError()
#         return A


# if __name__ == '__main__':
#     g = Graph(labeling_mode='virtual_spatial')
#     print()

import sys
import torch
sys.path.extend(['../'])
from graph import tools

# ------------------------------
# 原始骨架图定义
# ------------------------------
num_node = 25
self_link = [(i, i) for i in range(num_node)]
inward_ori_index = [
    (1, 2), (2, 21), (3, 21), (4, 3), (5, 21), (6, 5), (7, 6),
    (8, 7), (9, 21), (10, 9), (11, 10), (12, 11), (13, 1),
    (14, 13), (15, 14), (16, 15), (17, 1), (18, 17), (19, 18),
    (20, 19), (22, 8), (23, 8), (24, 12), (25, 12)
]
inward = [(i - 1, j - 1) for (i, j) in inward_ori_index]
outward = [(j, i) for (i, j) in inward]
neighbor = inward + outward

# ===========================================================
# ✅ 新增：预定义人体拓扑超图分区（9个功能区域）
# ===========================================================
def build_predefined_hypergraph(num_joints=25):
    """
    构建预定义拓扑超图先验 H_prior 与 A_prior
    """
    assert num_joints == 25, "当前超图定义基于 NTU 25 关节骨架"

    # 定义各个功能区域的节点索引 (0-based)
    groups = {
        'left_hand': [21, 22, 6, 7],
        'left_arm': [4, 5, 6, 7],
        'right_hand': [23, 24, 10, 11],
        'right_arm': [8, 9, 10, 11],
        'left_foot': [14, 15],
        'left_leg': [12, 13, 14],
        'right_foot': [18, 19],
        'right_leg': [16, 17, 18],
        'torso_head': [3, 2, 20, 1, 0]
    }

    V = num_joints
    E = len(groups)
    H_prior = torch.zeros(V, E)

    for e, (name, idxs) in enumerate(groups.items()):
        H_prior[idxs, e] = 1.0

    # 计算 A_prior = H D^-1 H^T
    De_inv = torch.diag(1.0 / H_prior.sum(0))
    A_prior = H_prior @ De_inv @ H_prior.T
    A_prior = 0.5 * (A_prior + A_prior.T)

    return H_prior, A_prior


# ===========================================================
# ✅ 修改 Graph 类
# ===========================================================
class Graph:
    def __init__(self, hyper_joints=0, labeling_mode='spatial', use_hyper_prior=False):
        """
        Args:
            hyper_joints: 超节点数量（可选）
            labeling_mode: 原有空间建图方式
            use_hyper_prior: 是否生成预定义超图先验
        """
        self.num_node = num_node
        self.self_link = self_link
        self.inward = inward
        self.outward = outward
        self.neighbor = neighbor
        self.hyper_joints = hyper_joints

        # 原始邻接矩阵 A
        self.A = self.get_adjacency_matrix(labeling_mode)

        # ✅ 如果需要生成预定义超图
        if use_hyper_prior:
            self.H_prior, self.A_prior = build_predefined_hypergraph(num_joints=self.num_node)
        else:
            self.H_prior, self.A_prior = None, None


    def get_adjacency_matrix(self, labeling_mode=None):
        if labeling_mode is None:
            return self.A
        if labeling_mode == 'spatial':
            A = tools.get_spatial_graph(num_node, self_link, inward, outward)
        elif labeling_mode == 'spatial_ensemble':
            A = tools.get_spatial_graph_ensemble(num_node, self_link, inward, outward, 8)
        elif labeling_mode == 'virtual_ensemble':
            A = tools.get_virtual_graph_ensemble(num_node, self_link, inward, outward, self.hyper_joints, 8)
        else:
            raise ValueError()
        return A


if __name__ == '__main__':
    g = Graph(labeling_mode='spatial', use_hyper_prior=True)
    print("A_prior shape:", None if g.A_prior is None else g.A_prior.shape)
    print("H_prior shape:", None if g.H_prior is None else g.H_prior.shape)
