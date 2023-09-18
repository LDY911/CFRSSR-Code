import tensorflow as tf
import numpy as np


class TFGraphInterface(object):
    def __init__(self):
        pass

    @staticmethod
    def convert_sparse_mat_to_tensor_hsj(adj, value_weight1, value_weight2, weighted_type):
        adj = adj.tocoo()
        old_value = adj.data
        new_value = value_weight1[adj.row] * value_weight2[adj.col]  # ————注意：adj.row应该代表user，是受影响的一方；adj.col应该代表friend，是施加影响的一方
        if weighted_type == 0:  # 加权的类型是0，代表直接替换原权重
            final_value = new_value
        if weighted_type == 1:  # 加权的类型是1，代表和原来的权重取平均
            final_value = (old_value + new_value) / 2
        if weighted_type == 2:  # 加权的类型是2，代表和原来的权重相乘后再标准化一下
            final_value = old_value * new_value
            max_final_value = max(final_value)
            min_final_value = min(final_value)
            final_value = (final_value - min_final_value) / (max_final_value - min_final_value)
        indices = np.mat(list(zip(adj.row, adj.col)))
        adj_tensor = tf.SparseTensor(indices, final_value.astype(np.float32), adj.shape)
        return adj_tensor

    @staticmethod
    def convert_sparse_mat_to_tensor(adj):
        row, col = adj.nonzero()
        indices = np.array(list(zip(row, col)))
        adj_tensor = tf.SparseTensor(indices=indices, values=adj.data, dense_shape=adj.shape)
        return adj_tensor

    @staticmethod
    def convert_sparse_mat_to_tensor_inputs(X):
        coo = X.tocoo()
        indices = np.mat([coo.row, coo.col]).transpose()
        return indices, coo.data, coo.shape