import sys
import os
import math
import time
import pickle as pkl
import scipy as sp
from scipy import io
import numpy as np
import pandas as pd
import networkx as nx
import dgl
import torch
from sklearn.preprocessing import label_binarize
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
from numpy.linalg import eig, eigh
from torch_sparse import SparseTensor
from torch_geometric.utils import is_undirected, to_undirected


def generate_signal_data():
    data = io.loadmat('node_raw_data/2Dgrid.mat')
    A = data['A']
    x = data['F'].astype(np.float32)
    m = data['mask']

    A = sp.sparse.coo_matrix(A).todense()

    D_vec = np.sum(A, axis=1).A1
    D_vec_invsqrt_corr = 1 / np.sqrt(D_vec)
    D_invsqrt_corr = np.diag(D_vec_invsqrt_corr)
    L = np.eye(10000) - D_invsqrt_corr @ A @ D_invsqrt_corr

    e, u = eigh(L)

    y_low  = u @ np.diag(np.array([math.exp(-10*(ee-0)**2) for ee in e])) @ u.T @ x
    y_high = u @ np.diag(np.array([1 - math.exp(-10*(ee-0)**2) for ee in e])) @ u.T @ x
    y_band = u @ np.diag(np.array([math.exp(-10*(ee-1)**2) for ee in e])) @ u.T @ x
    y_rej  = u @ np.diag(np.array([1 - math.exp(-10*(ee-1)**2) for ee in e])) @ u.T @ x
    y_comb = u @ np.diag(np.array([abs(np.sin(ee*math.pi)) for ee in e])) @ u.T @ x

    e = torch.FloatTensor(e)
    u = torch.FloatTensor(u)
    x = torch.FloatTensor(x)
    m = torch.LongTensor(m).squeeze()
    y_low = torch.FloatTensor(y_low)
    y_high = torch.FloatTensor(y_high)
    y_band = torch.FloatTensor(y_band)
    y_rej = torch.FloatTensor(y_rej)
    y_comb = torch.FloatTensor(y_comb)

    torch.save([e, u, x, y_low, m],  'data/signal_low.pt')
    torch.save([e, u, x, y_high, m], 'data/signal_high.pt')
    torch.save([e, u, x, y_band, m], 'data/signal_band.pt')
    torch.save([e, u, x, y_rej, m],  'data/signal_rej.pt')
    torch.save([e, u, x, y_comb, m], 'data/signal_comb.pt')


def normalize_graph(g):
    g = np.array(g)
    g = g + g.T
    g[g > 0.] = 1.0
    deg = g.sum(axis=1).reshape(-1)
    deg[deg == 0.] = 1.0
    deg = np.diag(deg ** -0.5)
    adj = np.dot(np.dot(deg, g), deg)
    L = np.eye(g.shape[0]) - adj
    return L


def eigen_decompositon(g):
    "The normalized (unit “length”) eigenvectors, "
    "such that the column v[:,i] is the eigenvector corresponding to the eigenvalue w[i]."
    g = normalize_graph(g)      
    e, u = eigh(g)
    return e, u


def parse_index_file(filename):
    """Parse index file."""
    index = []
    for line in open(filename):
        index.append(int(line.strip()))
    return index


def feature_normalize(x):
    x = np.array(x)
    rowsum = x.sum(axis=1, keepdims=True)
    rowsum = np.clip(rowsum, 1, 1e10)
    return x / rowsum


def load_data(dataset_str,current_dir):
    names = ['x', 'y', 'tx', 'ty', 'allx', 'ally', 'graph']
    objects = []
    for i in range(len(names)):
        with open(f"{current_dir}/../data/{dataset_str}/raw/ind.{dataset_str}.{names[i]}", 'rb') as f:
            if sys.version_info > (3, 0):
                objects.append(pkl.load(f, encoding='latin1'))
            else:
                objects.append(pkl.load(f))

    x, y, tx, ty, allx, ally, graph = tuple(objects)
    test_idx_reorder = parse_index_file(f"{current_dir}/../data/{dataset_str}/raw/ind.{dataset_str}.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    if dataset_str == 'citeseer':
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder)+1)
        tx_extended = sp.sparse.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range-min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range-min(test_idx_range), :] = ty
        ty = ty_extended

    features = sp.sparse.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    adj = nx.adjacency_matrix(nx.from_dict_of_lists(graph))

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]

    return adj, features, labels


def eig_dgl_adj_sparse(g, sm=0, lm=0):
    A = g.adj(scipy_fmt='csr')
    deg = np.array(A.sum(axis=0)).flatten()
    D_ = sp.sparse.diags(deg ** -0.5)

    A_ = D_.dot(A.dot(D_))
    L_ = sp.sparse.eye(g.num_nodes()) - A_

    if sm > 0:
        e1, u1 = sp.sparse.linalg.eigsh(L_, k=sm, which='SM', tol=1e-5)
        e1, u1 = map(torch.FloatTensor, (e1, u1))

    if lm > 0:
        e2, u2 = sp.sparse.linalg.eigsh(L_, k=lm, which='LM', tol=1e-5)
        e2, u2 = map(torch.FloatTensor, (e2, u2))

    if sm > 0 and lm > 0:
        return torch.cat((e1, e2), dim=0), torch.cat((u1, u2), dim=1)
    elif sm > 0:
        return e1, u1
    elif lm > 0:
        return e2, u2
    else:
        pass


def load_fb100_dataset():
    mat = io.loadmat('node_raw_data/Penn94.mat')
    A = mat['A']
    metadata = mat['local_info']

    edge_index = A.nonzero()
    metadata = metadata.astype(int)
    label = metadata[:, 1] - 1  # gender label, -1 means unlabeled

    # make features into one-hot encodings
    feature_vals = np.hstack((np.expand_dims(metadata[:, 0], 1), metadata[:, 2:]))
    features = np.empty((A.shape[0], 0))
    for col in range(feature_vals.shape[1]):
        feat_col = feature_vals[:, col]
        feat_onehot = label_binarize(feat_col, classes=np.unique(feat_col))
        features = np.hstack((features, feat_onehot))

    node_feat = torch.tensor(features, dtype=torch.float)
    num_nodes = metadata.shape[0]
    label = torch.LongTensor(label)

    g = dgl.graph((edge_index[0], edge_index[1]), num_nodes=num_nodes)

    return g, node_feat, label


def generate_node_data(data_name, dataset):
    '''
    decompose dataset including:
    'cora', 'citeseer', 'pubmed'
    'chameleon', 'squirrel', 'photo', 'actor'
    '''
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if data_name in ['cora', 'citeseer', 'pubmed']:

        adj, x, y = load_data(data_name,current_dir)
        adj = adj.todense()
        x = x.todense()

        e, u = eigen_decompositon(adj)      

        e = torch.FloatTensor(e)
        u = torch.FloatTensor(u)
        x = torch.FloatTensor(x)        
        y = torch.LongTensor(dataset.data.y)

        torch.save([e, u, x, y],  f'{current_dir}/../Decomposed_data/{data_name}_decom.pt')

    elif data_name in ['arxiv']:
        g = dataset[0][0]
        g = dgl.add_reverse_edges(g)
        g = dgl.to_simple(g)
        print('adj_spaser decompose... maybe cost a long time...')
        e, u = eig_dgl_adj_sparse(g, sm=5000)
        print('success')
        x = g.ndata['feat']
        y = data[0][1]

        torch.save([e, u, x, y],  f'{current_dir}/../Decomposed_data/{data_name}_decom2.pt')

    elif data_name in ['penn']:
        g, x, y = load_fb100_dataset()
        g = dgl.add_reverse_edges(g)
        g = dgl.to_simple(g)

        e, u = eig_dgl_adj_sparse(g, sm=3000, lm=3000)

        torch.save([e, u, x, y],  f'{current_dir}/../Decomposed_data/{data_name}_decom.pt')

    elif data_name in ['chameleon', 'squirrel', 'photo','actor']:

        data = dataset.data
        adj = SparseTensor(row=data.edge_index[0], col=data.edge_index[1])
        adj = adj.to_dense()

        x = data.x
        y = data.y

        e, u = eigen_decompositon(adj)

        e = torch.FloatTensor(e)
        u = torch.FloatTensor(u)
        x = torch.FloatTensor(x)
        y = torch.LongTensor(y)

        torch.save([e, u, x, y],  f'{current_dir}/../Decomposed_data/{data_name}_decom.pt')


if __name__ == '__main__':
    # test
    generate_node_data('cora')

