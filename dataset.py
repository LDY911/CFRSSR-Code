import array
import os
import pickle
import time

import torch
import numpy as np
import pandas as pd
import itertools
from tqdm import tqdm
from torch_geometric.utils import degree
from recbole.data.dataset import SequentialDataset
from recbole.data.dataset import Dataset as RecBoleDataset
from recbole.utils import set_color, FeatureSource
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from multiprocessing import Pool
from scipy.sparse import csr_matrix, lil_matrix, coo_matrix, dok_matrix, vstack
from collections import defaultdict
from scipy.spatial.distance import cdist


class GeneralGraphDataset(RecBoleDataset):
    def __init__(self, config):
        super().__init__(config)

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.
        Construct the square matrix from the training data and normalize it
        using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized interaction matrix in Tensor.
        """

        row = self.inter_feat[self.uid_field]
        col = self.inter_feat[self.iid_field] + self.user_num
        edge_index1 = torch.stack([row, col])
        edge_index2 = torch.stack([col, row])
        edge_index = torch.cat([edge_index1, edge_index2], dim=1)

        deg = degree(edge_index[0], self.user_num + self.item_num)

        norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))
        edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]

        return edge_index, edge_weight

    def get_bipartite_inter_mat(self, row='user', row_norm=True):
        r"""Get the row-normalized bipartite interaction matrix of users and items.
        """
        if row == 'user':
            row_field, col_field = self.uid_field, self.iid_field
        else:
            row_field, col_field = self.iid_field, self.uid_field

        row = self.inter_feat[row_field]
        col = self.inter_feat[col_field]
        edge_index = torch.stack([row, col])

        if row_norm:
            deg = degree(edge_index[0], self.num(row_field))
            norm_deg = 1. / torch.where(deg == 0, torch.ones([1]), deg)
            edge_weight = norm_deg[edge_index[0]]
        else:
            row_deg = degree(edge_index[0], self.num(row_field))
            col_deg = degree(edge_index[1], self.num(col_field))

            row_norm_deg = 1. / torch.sqrt(torch.where(row_deg == 0, torch.ones([1]), row_deg))
            col_norm_deg = 1. / torch.sqrt(torch.where(col_deg == 0, torch.ones([1]), col_deg))

            edge_weight = row_norm_deg[edge_index[0]] * col_norm_deg[edge_index[1]]

        return edge_index, edge_weight

class SessionGraphDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

    def session_graph_construction(self):
        # Default session graph dataset follows the graph construction operator like SR-GNN.
        self.logger.info('Constructing session graphs.')
        item_seq = self.inter_feat[self.item_id_list_field]
        item_seq_len = self.inter_feat[self.item_list_length_field]
        x = []
        edge_index = []
        alias_inputs = []

        for i, seq in enumerate(tqdm(list(torch.chunk(item_seq, item_seq.shape[0])))):
            seq, idx = torch.unique(seq, return_inverse=True)
            x.append(seq)
            alias_seq = idx.squeeze(0)[:item_seq_len[i]]
            alias_inputs.append(alias_seq)
            # No repeat click
            edge = torch.stack([alias_seq[:-1], alias_seq[1:]]).unique(dim=-1)
            edge_index.append(edge)

        self.inter_feat.interaction['graph_idx'] = torch.arange(item_seq.shape[0])
        self.graph_objs = {
            'x': x,
            'edge_index': edge_index,
            'alias_inputs': alias_inputs
        }

    def build(self):
        datasets = super().build()
        for dataset in datasets:
            dataset.session_graph_construction()
        return datasets

class MultiBehaviorDataset(SessionGraphDataset):

    def session_graph_construction(self):
        self.logger.info('Constructing multi-behavior session graphs.')
        self.item_behavior_list_field = self.config['ITEM_BEHAVIOR_LIST_FIELD']
        self.behavior_id_field = self.config['BEHAVIOR_ID_FIELD']
        item_seq = self.inter_feat[self.item_id_list_field]
        item_seq_len = self.inter_feat[self.item_list_length_field]
        if self.item_behavior_list_field == None or self.behavior_id_field == None:
            # To be compatible with existing datasets
            item_behavior_seq = torch.tensor([0] * len(item_seq))
            self.behavior_id_field = 'behavior_id'
            self.field2id_token[self.behavior_id_field] = {0:'interaction'}
        else:
            item_behavior_seq = self.inter_feat[self.item_list_length_field]

        edge_index = []
        alias_inputs = []
        behaviors = torch.unique(item_behavior_seq)
        x = {}
        for behavior in behaviors:
            x[behavior.item()] = []

        behavior_seqs = list(torch.chunk(item_behavior_seq, item_seq.shape[0]))
        for i, seq in enumerate(tqdm(list(torch.chunk(item_seq, item_seq.shape[0])))):
            bseq = behavior_seqs[i]
            for behavior in behaviors:
                bidx = torch.where(bseq == behavior)
                subseq = torch.index_select(seq, 0, bidx[0])
                subseq, _ = torch.unique(subseq, return_inverse=True)
                x[behavior.item()].append(subseq)

            seq, idx = torch.unique(seq, return_inverse=True)
            alias_seq = idx.squeeze(0)[:item_seq_len[i]]
            alias_inputs.append(alias_seq)
            # No repeat click
            edge = torch.stack([alias_seq[:-1], alias_seq[1:]]).unique(dim=-1)
            edge_index.append(edge)

        nx = {}
        for k, v in x.items():
            behavior_name = self.id2token(self.behavior_id_field, k)
            nx[behavior_name] = v

        self.inter_feat.interaction['graph_idx'] = torch.arange(item_seq.shape[0])
        self.graph_objs = {
            'x': nx,
            'edge_index': edge_index,
            'alias_inputs': alias_inputs
        }

class LESSRDataset(SessionGraphDataset):
    def session_graph_construction(self):
        self.logger.info('Constructing LESSR session graphs.')
        item_seq = self.inter_feat[self.item_id_list_field]
        item_seq_len = self.inter_feat[self.item_list_length_field]

        empty_edge = torch.stack([torch.LongTensor([]), torch.LongTensor([])])

        x = []
        edge_index_EOP = []
        edge_index_shortcut = []
        is_last = []

        for i, seq in enumerate(tqdm(list(torch.chunk(item_seq, item_seq.shape[0])))):
            seq, idx = torch.unique(seq, return_inverse=True)
            x.append(seq)
            alias_seq = idx.squeeze(0)[:item_seq_len[i]]
            edge = torch.stack([alias_seq[:-1], alias_seq[1:]])
            edge_index_EOP.append(edge)
            last = torch.zeros_like(seq, dtype=torch.bool)
            last[alias_seq[-1]] = True
            is_last.append(last)
            sub_edges = []
            for j in range(1, item_seq_len[i]):
                sub_edges.append(torch.stack([alias_seq[:-j], alias_seq[j:]]))
            shortcut_edge = torch.cat(sub_edges, dim=-1).unique(dim=-1) if len(sub_edges) > 0 else empty_edge
            edge_index_shortcut.append(shortcut_edge)

        self.inter_feat.interaction['graph_idx'] = torch.arange(item_seq.shape[0])
        self.graph_objs = {
            'x': x,
            'edge_index_EOP': edge_index_EOP,
            'edge_index_shortcut': edge_index_shortcut,
            'is_last': is_last
        }
        self.node_attr = ['x', 'is_last']

class GCEGNNDataset(SequentialDataset):
    def __init__(self, config):
        super().__init__(config)

    def reverse_session(self):
        self.logger.info('Reversing sessions.')
        item_seq = self.inter_feat[self.item_id_list_field]
        item_seq_len = self.inter_feat[self.item_list_length_field]
        for i in tqdm(range(item_seq.shape[0])):
            item_seq[i,:item_seq_len[i]] = item_seq[i,:item_seq_len[i]].flip(dims=[0])

    def bidirectional_edge(self, edge_index):
        seq_len = edge_index.shape[1]
        ed = edge_index.T
        ed2 = edge_index.T.flip(dims=[1])
        idc = ed.unsqueeze(1).expand(-1, seq_len, 2) == ed2.unsqueeze(0).expand(seq_len, -1, 2)
        return torch.logical_and(idc[:,:,0], idc[:,:,1]).any(dim=-1)

    def session_graph_construction(self):
        self.logger.info('Constructing session graphs.')
        item_seq = self.inter_feat[self.item_id_list_field]
        item_seq_len = self.inter_feat[self.item_list_length_field]
        x = []
        edge_index = []
        edge_attr = []
        alias_inputs = []

        for i, seq in enumerate(tqdm(list(torch.chunk(item_seq, item_seq.shape[0])))):
            seq, idx = torch.unique(seq, return_inverse=True)
            x.append(seq)
            alias_seq = idx.squeeze(0)[:item_seq_len[i]]
            alias_inputs.append(alias_seq)

            edge_index_backward = torch.stack([alias_seq[:-1], alias_seq[1:]])
            edge_attr_backward = torch.where(self.bidirectional_edge(edge_index_backward), 3, 1)
            edge_backward = torch.cat([edge_index_backward, edge_attr_backward.unsqueeze(0)], dim=0)

            edge_index_forward = torch.stack([alias_seq[1:], alias_seq[:-1]])
            edge_attr_forward = torch.where(self.bidirectional_edge(edge_index_forward), 3, 2)
            edge_forward = torch.cat([edge_index_forward, edge_attr_forward.unsqueeze(0)], dim=0)

            edge_index_selfloop = torch.stack([alias_seq, alias_seq])
            edge_selfloop = torch.cat([edge_index_selfloop, torch.zeros([1, edge_index_selfloop.shape[1]])], dim=0)

            edge = torch.cat([edge_backward, edge_forward, edge_selfloop], dim=-1).long()
            edge = edge.unique(dim=-1)

            cur_edge_index = edge[:2]
            cur_edge_attr = edge[2]
            edge_index.append(cur_edge_index)
            edge_attr.append(cur_edge_attr)

        self.inter_feat.interaction['graph_idx'] = torch.arange(item_seq.shape[0])
        self.graph_objs = {
            'x': x,
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            'alias_inputs': alias_inputs
        }

    def build(self):
        datasets = super().build()
        for dataset in datasets:
            dataset.reverse_session()
            dataset.session_graph_construction()
        return datasets

class SocialDataset(GeneralGraphDataset):
    """:class:`SocialDataset` is based on :class:`~recbole_gnn.data.dataset.GeneralGraphDataset`,
    and load ``.net``.

    All users in ``.inter`` and ``.net`` are remapped into the same ID sections.
    Users that only exist in social network will be filtered.

    It also provides several interfaces to transfer ``.net`` features into coo sparse matrix,
    csr sparse matrix, :class:`DGL.Graph` or :class:`PyG.Data`.

    Attributes:
        net_src_field (str): The same as ``config['NET_SOURCE_ID_FIELD']``.

        net_tgt_field (str): The same as ``config['NET_TARGET_ID_FIELD']``.

        net_feat (pandas.DataFrame): Internal data structure stores the users' social network relations.
            It's loaded from file ``.net``.
    """
    def __init__(self, config):
        super().__init__(config)
    
    def _get_field_from_config(self):
        super()._get_field_from_config()

        self.net_src_field = self.config['NET_SOURCE_ID_FIELD']
        self.net_tgt_field = self.config['NET_TARGET_ID_FIELD']
        self.filter_net_by_inter = self.config['filter_net_by_inter']
        self.undirected_net = self.config['undirected_net']
        self._check_field('net_src_field', 'net_tgt_field')

        self.logger.debug(set_color('net_src_field', 'blue') + f': {self.net_src_field}')
        self.logger.debug(set_color('net_tgt_field', 'blue') + f': {self.net_tgt_field}')

    def _data_filtering(self):
        super()._data_filtering()
        if self.filter_net_by_inter:
            self._filter_net_by_inter()

    def _filter_net_by_inter(self):
        """Filter users in ``net_feat`` that don't occur in interactions.
        """
        inter_uids = set(self.inter_feat[self.uid_field])
        self.net_feat.drop(self.net_feat.index[~self.net_feat[self.net_src_field].isin(inter_uids)], inplace=True)
        self.net_feat.drop(self.net_feat.index[~self.net_feat[self.net_tgt_field].isin(inter_uids)], inplace=True)

    def _load_data(self, token, dataset_path):
        super()._load_data(token, dataset_path)
        self.net_feat = self._load_net(self.dataset_name, self.dataset_path)

    @property
    def net_num(self):
        """Get the number of social network records.

        Returns:
            int: Number of social network records.
        """
        return len(self.net_feat)

    def __str__(self):
        info = [
            super().__str__(),
            set_color('The number of social network relations', 'blue') + f': {self.net_num}'
        ]  # yapf: disable
        return '\n'.join(info)

    def _build_feat_name_list(self):
        feat_name_list = super()._build_feat_name_list()
        if self.net_feat is not None:
            feat_name_list.append('net_feat')
        return feat_name_list

    def _load_net(self, token, dataset_path):
        self.logger.debug(set_color(f'Loading social network from [{dataset_path}].', 'green'))
        net_path = os.path.join(dataset_path, f'{token}.net')
        if not os.path.isfile(net_path):
            raise ValueError(f'[{token}.net] not found in [{dataset_path}].')
        df = self._load_feat(net_path, FeatureSource.NET)
        if self.undirected_net:
            row = df[self.net_src_field]
            col = df[self.net_tgt_field]
            df_net_src = pd.concat([row, col], axis=0)
            df_net_tgt = pd.concat([col, row], axis=0)
            df_net_src.name = self.net_src_field
            df_net_tgt.name = self.net_tgt_field
            df = pd.concat([df_net_src, df_net_tgt], axis=1)
        self._check_net(df)
        return df

    def _check_net(self, net):
        net_warn_message = 'net data requires field [{}]'
        assert self.net_src_field in net, net_warn_message.format(self.net_src_field)
        assert self.net_tgt_field in net, net_warn_message.format(self.net_tgt_field)

    def _init_alias(self):
        """Add :attr:`alias_of_user_id`.
        """
        self._set_alias('user_id', [self.uid_field, self.net_src_field, self.net_tgt_field])
        self._set_alias('item_id', [self.iid_field])

        for alias_name_1, alias_1 in self.alias.items():
            for alias_name_2, alias_2 in self.alias.items():
                if alias_name_1 != alias_name_2:
                    intersect = np.intersect1d(alias_1, alias_2, assume_unique=True)
                    if len(intersect) > 0:
                        raise ValueError(
                            f'`alias_of_{alias_name_1}` and `alias_of_{alias_name_2}` '
                            f'should not have the same field {list(intersect)}.'
                        )

        self._rest_fields = self.token_like_fields
        for alias_name, alias in self.alias.items():
            isin = np.isin(alias, self._rest_fields, assume_unique=True)
            if isin.all() is False:
                raise ValueError(
                    f'`alias_of_{alias_name}` should not contain '
                    f'non-token-like field {list(alias[~isin])}.'
                )
            self._rest_fields = np.setdiff1d(self._rest_fields, alias, assume_unique=True)

    def get_norm_net_adj_mat(self, row_norm=False):
        r"""Get the normalized socail matrix of users and users.
        Construct the square matrix from the social network data and 
        normalize it using the laplace matrix.
        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}
        Returns:
            The normalized social network matrix in Tensor.
        """

        row = self.net_feat[self.net_src_field]
        col = self.net_feat[self.net_tgt_field]
        edge_index = torch.stack([row, col])

        deg = degree(edge_index[0], self.user_num)

        if row_norm:
            norm_deg = 1. / torch.where(deg == 0, torch.ones([1]), deg)
            edge_weight = norm_deg[edge_index[0]]
        else:
            norm_deg = 1. / torch.sqrt(torch.where(deg == 0, torch.ones([1]), deg))
            edge_weight = norm_deg[edge_index[0]] * norm_deg[edge_index[1]]

        return edge_index, edge_weight

    def net_matrix(self, form='coo', value_field=None):
        """Get sparse matrix that describe social relations between user_id and user_id.

        Sparse matrix has shape (user_num, user_num).

        Returns:
            scipy.sparse: Sparse matrix in form ``coo`` or ``csr``.
        """
        return self._create_sparse_matrix(self.net_feat, self.net_src_field, self.net_tgt_field, form, value_field)

class CIRecDataset(SocialDataset):
    def __init__(self, config):
        super().__init__(config)
        self.dataset_name = config.dataset

    def get_treatment(self, data_inter, data_trust):
        data_inter = pd.DataFrame(data_inter.cpu().numpy().T, columns=['user_id', 'item_id'])
        data_trust = pd.DataFrame(data_trust.cpu().numpy().T, columns=['friend_id', 'user_id'])
        friend_item_result = []
        friend_user_result = []
        k = 0
        for user_id in data_inter['user_id'].drop_duplicates():
            if k % 100 == 0:
                    print(k, '/', len(data_inter['user_id'].drop_duplicates()))
            user_friends, user_self_buy, user_friends_item = self.search(user_id, data_inter, data_trust)
            friend_item_result.append(user_friends_item.to_numpy().tolist())
            friend_user_result.append([user_id] * len(user_friends_item.to_numpy()))
            k += 1
        friend_user_result = list(itertools.chain.from_iterable(friend_user_result))
        friend_item_result = list(itertools.chain.from_iterable(friend_item_result))
        friend_item_result = list(itertools.chain.from_iterable(friend_item_result))
        data1 = [1 for i in range(len(friend_user_result))]
        treatment_matrix = csr_matrix((data1, (friend_user_result, friend_item_result)), shape=(self.user_num, self.item_num))
        pickle.dump(treatment_matrix, open('./'+self.dataset_name+'-t_f.pkl', 'wb'))

        return treatment_matrix

    def search(self, user_id, data_inter, data_trust):
        user_id = int(user_id)
        user_self_buy = data_inter[data_inter['user_id'] == user_id]['item_id']
        user_friends = data_trust[data_trust['user_id'] == user_id]['friend_id']
        user_friends_item_list = []
        temp_dict = {}
        for i in user_friends:
            user_friends_single = pd.DataFrame(data_inter[data_inter['user_id'] == i]['item_id'], columns=['item_id'])
            user_friends_item_list.append(user_friends_single['item_id'].tolist())
            temp_dict[i] = user_friends_single['item_id'].tolist()
        user_friends_item_list = list(itertools.chain.from_iterable(user_friends_item_list))
        user_friends_item = pd.DataFrame(user_friends_item_list, columns=['item_id']).drop_duplicates()
        return user_friends, user_self_buy, user_friends_item

    def get_cf_data_user(self, data_user_embedding, adj, T_f):
        node_embs_user = data_user_embedding.cpu().numpy()
        simi_mat_user = cdist(node_embs_user, node_embs_user, metric="euclidean").astype(np.float32)
        simi_mat_user = np.reciprocal(simi_mat_user)
        thresh = self.config.thresh
        thresh = thresh.split(', ')
        thresh_user = np.percentile(simi_mat_user, 100 - float(thresh[0]))
        np.fill_diagonal(simi_mat_user, 0)
        def replace_small_values_with_zero(matrix, threshold):
            chunk_size = 10000
            for i in range(0, matrix.shape[0], chunk_size):
                for j in range(0, matrix.shape[1], chunk_size):
                    chunk = matrix[i:i + chunk_size, j:j + chunk_size]
                    chunk[chunk < threshold] = 0
                    matrix[i:i + chunk_size, j:j + chunk_size] = chunk
            return matrix
        simi_mat_user = replace_small_values_with_zero(simi_mat_user, thresh_user)
        simi_mat_user = csr_matrix(simi_mat_user)
        node_nns_user = []
        for i in range(simi_mat_user.shape[0]):
            row_data = list(zip(simi_mat_user[i].indices, simi_mat_user[i].data))
            sorted_data = sorted(row_data, key=lambda x: x[1], reverse=True)
            node_nns_user.append([element[0] for element in sorted_data])
        a_unique = np.array(adj.nonzero()).T
        b_unique = np.array(T_f.nonzero()).T


        a_unique_pd = pd.DataFrame(a_unique, columns=['user_id', 'item_id'])
        b_unique_pd = pd.DataFrame(b_unique, columns=['user_id', 'item_id'])
        node_pairs_user = pd.merge(a_unique_pd, b_unique_pd, on=['user_id', 'item_id'], how='inner')
        node_pairs_user = node_pairs_user.to_numpy()

        return node_pairs_user, node_nns_user

    def get_cf_data_item(self, data_item_embedding, adj, T_f, chunk_num=10):
        node_embs_item = data_item_embedding.cpu().numpy()
        simi_mat_item = cdist(node_embs_item, node_embs_item, metric="euclidean").astype(np.float32)
        simi_mat_item = np.reciprocal(simi_mat_item)
        thresh = self.config.thresh
        thresh = thresh.split(', ')
        thresh_item = np.percentile(simi_mat_item, 100 - float(thresh[1]))
        np.fill_diagonal(simi_mat_item, 0)
        def replace_small_values_with_zero(matrix, threshold):
            chunk_size = 10000
            for i in range(0, matrix.shape[0], chunk_size):
                for j in range(0, matrix.shape[1], chunk_size):
                    chunk = matrix[i:i + chunk_size, j:j + chunk_size]
                    chunk[chunk < threshold] = 0
                    matrix[i:i + chunk_size, j:j + chunk_size] = chunk
            return matrix
        simi_mat_item = replace_small_values_with_zero(simi_mat_item, thresh_item)
        simi_mat_item_block = np.array_split(simi_mat_item, int(10))
        simi_mat_item = []
        for block in simi_mat_item_block:
            simi_mat_item.append(csr_matrix(block))
        simi_mat_item = vstack(simi_mat_item)
        node_nns_item = []
        for i in range(simi_mat_item.shape[0]):
                row_data = list(zip(simi_mat_item[i].indices, simi_mat_item[i].data)) 
                sorted_data = sorted(row_data, key=lambda x: x[1], reverse=True)       
                node_nns_item.append([element[0] for element in sorted_data]) 
        a_unique = np.array(adj.nonzero()).T
        b_unique = np.array(T_f.nonzero()).T


        a_unique_pd = pd.DataFrame(a_unique, columns=['user_id', 'item_id'])
        b_unique_pd = pd.DataFrame(b_unique, columns=['user_id', 'item_id'])
        node_pairs_user = pd.merge(a_unique_pd, b_unique_pd, on=['user_id', 'item_id'], how='inner')
        node_pairs_user = node_pairs_user.to_numpy()

        node_pairs_item = node_pairs_user[np.argsort(node_pairs_user[:, 1])]
        
        return node_pairs_item, node_nns_item

    def multi_thread_cal_user(self, node_pairs, node_nns, adj, T_f):
        adj_cf_item = self.get_CF_UI_user((adj, T_f, node_nns, node_pairs, True))
        adj_cf_list = np.array(adj_cf_item).reshape(-1, 2)
        return adj_cf_list

    def multi_thread_cal_item(self, node_pairs, node_nns, adj, T_f):
        adj_cf_user = self.get_CF_UI_item((adj, T_f, node_nns, node_pairs, True))
        adj_cf_list = np.array(adj_cf_user).reshape(-1, 2)
        return adj_cf_list

    def get_CF_UI_user(self, params): 
        adj, T_f, node_nns, node_pairs, verbose = params
        adj_cf_list = array.array('i')
        c = 0
        for a, b in node_pairs:
            nns_a = node_nns[a]
            for i_user in nns_a:
                if (T_f[i_user, b]==0) and (adj[i_user, b]==0): 
                    adj_cf_list.extend([a, b])
                    break
            if verbose and c % 2000 == 0:
                print(f'{c} / {len(node_pairs)} done')
            c += 1
        return adj_cf_list

    def get_CF_UI_item(self, params):
        adj, T_f, node_nns, node_pairs, verbose = params
        adj_cf_list = array.array('i')
        c = 0
        for a, b in node_pairs:
            nns_b = node_nns[b]
            for i_item in nns_b:
                if (T_f[a, i_item]==0) and (adj[a, i_item]==0):
                    adj_cf_list.extend([a, b])
                    break
            if verbose and c % 2000 == 0:
                print(f'{c} / {len(node_pairs)} done')
            c += 1
        return adj_cf_list


    def get_CF(self, params): 
        adj, simi_mat_user, simi_mat_item, node_nns_user, node_nns_item, T_f, thresh_user, thresh_item, node_pairs, verbose = params
        T_cf = np.zeros(adj.shape)
        adj_cf = np.zeros(adj.shape)
        edges_cf_t0 = []
        edges_cf_t1 = []
        c = 0
        for a, b in node_pairs:
            nns_a = node_nns_user[a]
            nns_b = node_nns_item[b]
            i, j = 0, 0
            while i < len(nns_a) - 1 and j < len(nns_b) - 1:
                if simi_mat_user[a, nns_a[i]] + simi_mat_item[b, nns_b[j]] > thresh_user + thresh_item:
                    T_cf[a, b] = T_f[a, b]
                    adj_cf[a, b] = adj[a, b]
                    break
                if T_f[nns_a[i], nns_b[j]] != T_f[a, b]:
                    T_cf[a, b] = 1 - T_f[a, b]
                    adj_cf[a, b] = adj[nns_a[i], nns_b[j]]
                    if T_cf[a, b] == 0:
                        edges_cf_t0.append([nns_a[i], nns_b[j]])
                    else:
                        edges_cf_t1.append([nns_a[i], nns_b[j]])
                    break
                if simi_mat_user[a, nns_a[i + 1]] < simi_mat_item[b, nns_b[j + 1]]:
                    i += 1
                else:
                    j += 1
            c += 1
            if verbose and c % 2000 == 0:
                print(f'{c} / {len(node_pairs)} done')
        edges_cf_t0 = np.asarray(edges_cf_t0)
        if edges_cf_t0.size == 0:
            edges_cf_t0 = edges_cf_t0.reshape(-1, 2)
        edges_cf_t1 = np.asarray(edges_cf_t1)
        if edges_cf_t1.size == 0:
            edges_cf_t1 = edges_cf_t1.reshape(-1, 2)
        return T_cf, adj_cf, edges_cf_t0, edges_cf_t1

    def get_CF_U(self, params):
        adj, simi_mat_user, simi_mat_item, node_nns_user, node_nns_item, T_f, thresh_user, thresh_item, node_pairs, verbose = params
        T_cf_item = np.zeros(adj.shape)
        adj_cf_item = np.zeros(adj.shape)
        edges_cf_t0_item = []
        edges_cf_t1_item = []
        c = 0
        for a, b in node_pairs:
            nns_b = node_nns_item[b]
            i_item = 0
            while i_item < len(nns_b) - 1:
                if simi_mat_item[b, nns_b[i_item]] > thresh_item:
                    T_cf_item[a, b] = T_f[a, b]
                    adj_cf_item[a, b] = adj[a, b]
                    break
                if T_f[a, nns_b[i_item]] != T_f[a, b]:
                    T_cf_item[a, b] = 1 - T_f[a, b]
                    adj_cf_item[a, b] = adj[a, nns_b[i_item]]
                    if T_cf_item[a, b] == 0:
                        edges_cf_t0_item.append([a, nns_b[i_item]])
                    else:
                        edges_cf_t1_item.append([a, nns_b[i_item]])
                    break
                i_item += 1
            c += 1
            if verbose and c % 2000 == 0:
                print(f'{c} / {len(node_pairs)} done')
        edges_cf_t0_item = np.asarray(edges_cf_t0_item)
        if edges_cf_t0_item.size == 0:
            edges_cf_t0_item = edges_cf_t0_item.reshape(-1, 2)
        edges_cf_t1_item = np.asarray(edges_cf_t1_item)
        if edges_cf_t1_item.size == 0:
            edges_cf_t1_item = edges_cf_t1_item.reshape(-1, 2)

        return T_cf_item, adj_cf_item, edges_cf_t0_item, edges_cf_t1_item

    def get_Feature(self, adj):
        data_user = np.array(adj.todense().sum(1)).squeeze(1)
        data_item = np.array(adj.todense().sum(0)).squeeze(0)
        data_item = data_item + 10000
        data_all_feature = self.OneHot(np.concatenate((data_user, data_item)))
        data_user_feature = data_all_feature[:data_user.shape[0], :]
        data_item_feature = data_all_feature[data_user.shape[0]:, :]
        return data_user_feature, data_item_feature

    def OneHot(self, data):
        # integer encode
        label_encoder = LabelEncoder()
        integer_encoded = label_encoder.fit_transform(data)
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        return onehot_encoded

