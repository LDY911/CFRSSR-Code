import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from recbole.utils import InputType
from recbole.model.abstract_recommender import GeneralRecommender
from recbole.model.init import xavier_normal_initialization
import pickle
from scipy.sparse import csr_matrix, coo_matrix
import scipy.sparse as sp
from geomloss import SamplesLoss
import torch.nn.functional as F
from collections import Counter
import os

class Generate(GeneralRecommender):
    input_type = InputType.POINTWISE

    def __init__(self, config, dataset):
        super(Generate, self).__init__(config, dataset)
        # load dataset info
        self.dataset_name = dataset.dataset_name
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num

        self.edge_index, self.edge_weight = dataset.get_bipartite_inter_mat(row='user')
        self.edge_index, self.edge_weight = self.edge_index.to(self.device), self.edge_weight.to(self.device)
        self.adj_f = csr_matrix((np.array([1 for i in range(self.edge_index.shape[1])]), (np.array(self.edge_index[0, :].cpu()), np.array(self.edge_index[1, :].cpu()))), shape=(dataset.user_num, dataset.item_num))

        self.net_edge_index, self.net_edge_weight = dataset.get_norm_net_adj_mat(row_norm=True)
        self.net_edge_index, self.net_edge_weight = self.net_edge_index.to(self.device), self.net_edge_weight.to(self.device)
        if os.path.exists('./' + self.dataset_name + '-t_f.pkl'):
            self.T_f = pickle.load(open('./' + self.dataset_name + '-t_f.pkl', 'rb'))
        else:
            self.T_f = dataset.get_treatment(self.edge_index, self.net_edge_index)
            pickle.dump(self.T_f, open('./' + self.dataset_name + '-t_f.pkl', 'wb'))
        self.edges_f_t1 = torch.tensor(np.asarray(self.T_f.nonzero()).T).to(self.device)
        self.edges_f_t0 = torch.tensor(np.asarray((self.T_f == 0).nonzero()).T).to(self.device)
        self.user_embedding_distance = pickle.load(open('dataset/' + self.dataset_name + '/' + self.dataset_name + '_user_all_embeddings.pkl', 'rb'))
        self.item_embedding_distance = pickle.load(open('dataset/' + self.dataset_name + '/' + self.dataset_name + '_item_all_embeddings.pkl', 'rb'))

        node_pairs_user, node_nns_user = dataset.get_cf_data_user(self.user_embedding_distance, self.adj_f, self.T_f)
        node_pairs_item, node_nns_item = dataset.get_cf_data_item(self.item_embedding_distance, self.adj_f, self.T_f)
        self.adj_cf_user = dataset.multi_thread_cal_user(node_pairs_user, node_nns_user, self.adj_f, self.T_f)
        self.adj_cf_item = dataset.multi_thread_cal_item(node_pairs_item, node_nns_item, self.adj_f, self.T_f)

        # load parameters info
        self.embedding_size = config['embedding_size']
        self.pos_weight = torch.tensor(50)
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight)
        self.emb_loss = EmbLoss()
        self.LABEL = 'label'
        self.neg_num = 1000
        self.disc_func = 'lin'
        self.reg_weight = 1e-5

        final_user = pd.DataFrame(self.adj_cf_user, columns=['user_id', 'item_id'])
        final_item = pd.DataFrame(self.adj_cf_item, columns=['user_id', 'item_id'])
        cf_core = pd.merge(final_user, final_item, on=['user_id', 'item_id'], how='inner')
        cf_core = cf_core.drop_duplicates()
        cf_core.to_csv('./' + self.dataset_name + '_' + str(config.thresh) + '_influenced_inter_data_j.csv', index=0)

        print('The refined data for the counterfactual social network has been generated')
        time.sleep(3)

        self.encoder = Encoder(config, dataset, self.adj_f, self.device)
        self.decoder = Decoder(self.device)
        # parameters initialization
        self.apply(xavier_normal_initialization)

    def forward(self):
        pass

    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        label_f = interaction[self.LABEL]
        user_all_e, item_all_e = self.encoder()
        user_e = user_all_e[user]
        item_e = item_all_e[item]

        user = user.unsqueeze(dim=1)
        item = item.unsqueeze(dim=1)
        train_edges = torch.concat([user, item], dim=1)
        T_f = self.torch_gather_nd(torch.tensor(self.T_f.todense()).to(self.device), train_edges)

        T_cf_user = self.torch_gather_nd(torch.tensor(self.T_cf_user).to(self.device), train_edges)
        label_cf_user = self.torch_gather_nd(torch.tensor(self.adj_cf_user).to(self.device), train_edges)
        T_cf_item = self.torch_gather_nd(torch.tensor(self.T_cf_item).to(self.device), train_edges)
        label_cf_item = self.torch_gather_nd(torch.tensor(self.adj_cf_item).to(self.device), train_edges)

        score_f = self.decoder(user_e, item_e, T_f)
        score_cf_user = self.decoder(user_e, item_e, T_cf_user)
        score_cf_item = self.decoder(user_e, item_e, T_cf_item)

        np_f, np_cf_user, np_cf_item = self.sample_nodepairs_ui(self.neg_num, self.edges_f_t1, self.edges_f_t0, self.edges_cf_t1_user, self.edges_cf_t0_user, self.edges_cf_t1_item, self.edges_cf_t0_item)
        loss_disc = self.calc_disc_ui(self.disc_func, torch.cat((user_all_e, item_all_e), 0), np_f, np_cf_user, np_cf_item)
        loss_f = self.bce_loss(score_f, label_f) 
        loss_cf_user = self.bce_loss(score_cf_user, label_cf_user)
        loss_cf_item = self.bce_loss(score_cf_item, label_cf_item)
        loss_emb = self.emb_loss(user_e, item_e)
        return loss_disc + loss_f + loss_cf_user + loss_cf_item + self.reg_weight * loss_emb

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        user_all_e, item_all_e = self.encoder()

        user_e = user_all_e[user]  # [batch_size, embedding_size]
        all_item_e = item_all_e  # [n_items, batch_size]
        z_user_val = torch.tensor(np.empty([0, 128])).to(self.device)
        for i in user_e:
            temp1 = i.repeat(all_item_e.shape[0], 1)
            z_user_val = torch.cat((z_user_val, temp1), 0)
        z_item_val = all_item_e.repeat(user_e.shape[0], 1)
        item = torch.tensor(np.arange(self.n_items)).to(self.device)
        list_id = []
        for i in user:
            for j in item:
                list_id.append([i, j])
        T_f_flatten = torch.tensor(self.T_f.todense()).to(self.device)[np.array(torch.tensor(list_id).cpu()).T]
        scores_val_flatten = self.decoder(z_user_val, z_item_val, T_f_flatten)
        scores = scores_val_flatten.reshape((user.shape[0], item.shape[0]))

        return scores

    def sample_nodepairs_ui(self, num_np, edges_f_t1, edges_f_t0, edges_cf_t1_user, edges_cf_t0_user, edges_cf_t1_item, edges_cf_t0_item):
        nodepairs_f = torch.concat((edges_f_t1, edges_f_t0), dim=0)
        f_idx = torch.multinomial(torch.arange(len(nodepairs_f), dtype=torch.float), num_samples=min(num_np, len(nodepairs_f)), replacement=False).to(self.device)
        np_f = nodepairs_f.index_select(0, f_idx)
        nodepairs_cf_user = torch.concat((edges_cf_t1_user, edges_cf_t0_user), dim=0)
        cf_idx = torch.multinomial(torch.arange(len(nodepairs_cf_user), dtype=torch.float), num_samples=min(num_np, len(nodepairs_cf_user)), replacement=False).to(self.device)
        np_cf_user = nodepairs_cf_user.index_select(0, cf_idx)
        nodepairs_cf_item = torch.concat((edges_cf_t1_item, edges_cf_t0_item), dim=0)
        cf_idx = torch.multinomial(torch.arange(len(nodepairs_cf_item), dtype=torch.float), num_samples=min(num_np, len(nodepairs_cf_item)), replacement=False).to(self.device)
        np_cf_item = nodepairs_cf_item.index_select(0, cf_idx)
        return np_f, np_cf_user, np_cf_item

    def calc_disc_ui(self, disc_func, z, nodepairs_f, nodepairs_cf_user, nodepairs_cf_item):
        X_f = torch.cat((z.index_select(index=nodepairs_f[:, 0], dim=0), z.index_select(index=nodepairs_f[:, 1] + self.n_users, dim=0)), dim=1).to(self.device)
        X_f = torch.cat((X_f, X_f), dim=1)
        X_cf_user = torch.cat((z.index_select(index=nodepairs_cf_user[:, 0].type(torch.int32), dim=0), z.index_select(index=nodepairs_cf_user[:, 1].type(torch.int32) + self.n_users, dim=0)), dim=1).to(self.device)
        X_cf_item = torch.cat((z.index_select(index=nodepairs_cf_item[:, 0].type(torch.int32), dim=0), z.index_select(index=nodepairs_cf_item[:, 1].type(torch.int32) + self.n_users, dim=0)), dim=1).to(self.device)
        X_cf = torch.cat((X_cf_user, X_cf_item), dim=1)
        if disc_func == 'lin':
            mean_f = X_f.mean(0)
            mean_cf = X_cf.mean(0)
            loss_disc = torch.sqrt(F.mse_loss(mean_f, mean_cf) + 1e-6)
        elif disc_func == 'kl':
            pass
        elif disc_func == 'w':
            # Wasserstein distance
            dist = SamplesLoss(loss="sinkhorn", p=2, blur=.05)
            loss_disc = dist(X_cf, X_f)
        else:
            raise Exception('unsupported distance function for discrepancy loss')
        return loss_disc

    def torch_gather_nd(self, input_data, index_data):
        index_data = index_data.type(torch.float)
        input_data = input_data.contiguous()
        inds = index_data.mv(torch.tensor(input_data.stride()).to(self.device).type(torch.float)).type(torch.int64)
        x_gather = torch.index_select(input_data.contiguous().view(-1), 0, inds)
        return x_gather

class EmbLoss(nn.Module):
    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(input=torch.norm(embedding, p=self.norm), exponent=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss


class Encoder(nn.Module):
    def __init__(self, config, dataset, adj_train, device):
        super(Encoder, self).__init__()
        self.device = device
        self.interaction_matrix = coo_matrix(adj_train).astype(np.float32)
        self.n_users = dataset.user_num
        self.n_items = dataset.item_num
        self.latent_dim = 128
        self.n_layers = 2
        self.reg_weight = 1e-05

        self.user_embedding = torch.nn.Embedding(num_embeddings=self.n_users, embedding_dim=self.latent_dim)
        self.item_embedding = torch.nn.Embedding(num_embeddings=self.n_items, embedding_dim=self.latent_dim)

        self.norm_adj_matrix = self.get_norm_adj_mat()

    def forward(self):
        all_embeddings = self.get_ego_embeddings().to() 
        embeddings_list = [all_embeddings]
        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)
        user_all_embeddings, item_all_embeddings = torch.split(lightgcn_all_embeddings, [self.n_users, self.n_items])

        return user_all_embeddings.to(self.device), item_all_embeddings.to(self.device)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape)).to(self.device)
        return SparseL

    def get_ego_embeddings(self):
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings


class Decoder(nn.Module):
    def __init__(self, device):
        super(Decoder, self).__init__()
        self.device = device
        self.dim_h = 128
        self.dim_in = self.dim_h + 1
        self.mlp_out = nn.Sequential(nn.Linear(self.dim_in, self.dim_h, bias=True), nn.ELU(), nn.Dropout(), nn.Linear(self.dim_h, 1, bias=False))

    def forward(self, z_i, z_j, T):
        z = z_i * z_j 
        h = torch.cat((z.to(self.device), T.view(-1, 1).to(self.device)), dim=1)
        h = h.to(torch.float32)
        h = self.mlp_out(h).squeeze()
        return h.to(self.device)

