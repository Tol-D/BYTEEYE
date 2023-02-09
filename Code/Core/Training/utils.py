import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd

def encode_onehot(labels):
    # 必须在编码之前对类进行排序，以启用静态类编码。
    # 换句话说，请确保第一类始终映射到索引0
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    # 和GCN一样一行向量代表一个特征， 所以 字典hash的时候直接行hash
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    '''
        class_dict.get 先得到字典的值
        然后使用map 函数 进行映射
    '''
    return labels_onehot


def load_data(path="data/cora/", dataset="cora"):
    """Load citation network dataset (cora only for now)"""
    print('Loading {} dataset...'.format(dataset))

    # idx_features_labels.shape = (2708,1435)；
    # 第一列是节点编号，最后一列是节点类别，中间列是节点的特征
    idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
    # 提取特征并按行压缩为稀疏矩阵
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    # 将标签转换为one-hot编码
    labels = pd.get_dummies(idx_features_labels[:, -1]).values

    # build graph
    # 节点编号
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    # (节点编号：出现顺序)
    idx_map = {j: i for i, j in enumerate(idx)}
    # 边表，shape = (5429,2)
    edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
    # 使用idx_map映射edges_unordered中节点的编号
    edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
    # 将edges转换为邻接矩阵
    adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]),
                        dtype=np.float32)

    # 转换为对称矩阵
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    features = normalize_features(features)
    adj = normalize_adj(adj + sp.eye(adj.shape[0]))

    idx_train = range(140)
    idx_val = range(200, 500)
    idx_test = range(500, 1500)

    adj = torch.FloatTensor(np.array(adj.todense()))
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test


# def load_data(path="data/cora/", dataset="cora"):
#     """Load citation network dataset (cora only for now)"""
#     print('Loading {} dataset...'.format(dataset))
#
#     '''
#         一下引用的cora数据集
#         cora数据集介绍可以看https://blog.csdn.net/yeziand01/article/details/93374216
#     '''
#     idx_features_labels = np.genfromtxt("{}{}.content".format(path, dataset), dtype=np.dtype(str))
#     features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
#     labels = encode_onehot(idx_features_labels[:, -1]) #最后一列是标签
#
#     # build graph
#     idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
#     idx_map = {j: i for i, j in enumerate(idx)}
#     edges_unordered = np.genfromtxt("{}{}.cites".format(path, dataset), dtype=np.int32)
#     edges = np.array(list(map(idx_map.get, edges_unordered.flatten())), dtype=np.int32).reshape(edges_unordered.shape)
#     adj = sp.coo_matrix((np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])), shape=(labels.shape[0], labels.shape[0]), dtype=np.float32)
#
#     # build symmetric adjacency matrix
#     adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
#
#     features = normalize_features(features)
#     adj = normalize_adj(adj + sp.eye(adj.shape[0]))
#     '''
#         划分数据集
#     '''
#     idx_train = range(140)
#     idx_val = range(200, 500)
#     idx_test = range(500, 1500)
#
#     adj = torch.FloatTensor(np.array(adj.todense()))
#     features = torch.FloatTensor(np.array(features.todense()))
#     labels = torch.LongTensor(np.where(labels)[1]) # labels 再 dim=1 这个维度的索引，这里就是列
#
#     idx_train = torch.LongTensor(idx_train)
#     idx_val = torch.LongTensor(idx_val)
#     idx_test = torch.LongTensor(idx_test)
#
#     return adj, features, labels, idx_train, idx_val, idx_test
#     # 返回数据集

def normalize_adj(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt)


def normalize_features(mx):
    """Row-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    # 用对角矩阵与原始矩阵的点积起到标准化的作用，原始矩阵中每一行
    # 元素都会与对应的r_inv相乘，最终相当于除以了sum
    return mx


def accuracy(output, labels):
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)

