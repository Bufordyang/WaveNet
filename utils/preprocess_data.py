import torch
import numpy as np
import sys
sys.path.append('./utils')
import torch_geometric.transforms as T
from torch_geometric.datasets import Planetoid, WikipediaNetwork, Amazon, Actor
from torch_geometric.data import Data
from torch_geometric.utils import is_undirected, to_undirected
from utils.decompose_graph import *
from ogb.nodeproppred.dataset_dgl import DglNodePropPredDataset
import random


class Sameple_data():
    '''
    The class of customize of datasets
        Sample_data.name: name of dataset
        Sample_data.eigenvalues: eigenvalues
        Sample_data.vectors: vectors
        Sample_data.data: node feature
        Sample_data.num_classes: num_classes
        Sample_data.num_features: num_features
    '''
    def __init__(self, root, name):
        file_path = f'{root}{name}_decom.pt'       
        tep = torch.load(file_path)
        self.eigenvalues = tep[0]      
        self.vectors = tep[1]
        self.data = Data(x=tep[2],y=tep[3])
        self.num_classes = int(self.data.y.max()+1)
        self.num_features = int(self.data.x.shape[1])
        self.name = f'{name}'

class Sample_ogb():
    def __init__(self, name, tep):
        self.eigenvalues = tep[0]      
        self.vectors = tep[1]
        self.data = Data(x=tep[2],y=tep[3])
        self.num_classes = int(self.data.y.max()+1)
        self.num_features = int(self.data.x.shape[1])
        self.name = f'{name}'

def ogb_train_graph(root, name,prefix=None):
    file_path = f'{root}{name}_decom_{prefix}.pt'       
    tep = torch.load(file_path)
    ogb_graph_list = []
    for graph in tep:
        tep = Sample_ogb(name, graph)
        ogb_graph_list.append(tep)
    return ogb_graph_list

class Toy_graph():
    '''
    Toy_graph class based on pyg Data class
    '''
    def __init__(self,name):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        tep = torch.load(f'{current_dir}/../data/{name}.pt')
        self.eigenvalues = tep.eigenvalues      
        self.vectors = tep.vectors
        self.data = Data(x=tep.x,y=tep.y,edge_index=tep.edge_index)
        self.num_classes = int(self.data.y.max()+1)
        self.num_features = int(self.data.x.shape[1])


def walk_hop(edge_index, centernode):
    indices = torch.where(edge_index[0] == centernode)
    result = torch.stack([edge_index[0][indices],edge_index[1][indices]],dim=0)
    return result

def sample_subgraph(root, name, hop=3):
    file_path = f'{root}/processed/data.pt'
    data = torch.load(file_path)
    try:
        data = data[0]
    except:
        pass
    edge_index = data.edge_index
    start_node = torch.tensor(0)
    neighbors = walk_hop(edge_index, start_node)
    walked_nodes = torch.unique(neighbors[0])
    extended_nodes = torch.unique(neighbors[1])
    print('----Sampling-----')
    for _ in range(hop):
        for center_node in extended_nodes:
            if center_node not in walked_nodes:
                tep = walk_hop(edge_index, center_node)
                extended_nodes = torch.cat((extended_nodes, tep[1]),dim=0) 
                neighbors = torch.cat([neighbors,tep], dim=1)
            walked_nodes = torch.unique(neighbors[0])
    print('----neighbors.is_undirected:-----',is_undirected(neighbors))
    if not is_undirected(neighbors):
        neighbors = to_undirected(neighbors)

    node_index = torch.unique(neighbors[0])
    node_feature = data.x[node_index]
    node_label = data.y[node_index]

    res = Data(
        x=node_feature,
        edge_index=neighbors,
        y=node_label,
        # train_percent = data.train_percent
    )
    torch.save(res,f'./testspace/subgraph_{name}.pt')
    print('----Done!-----',is_undirected(neighbors))

def decompose_matrix(name, dataset):
    print("Processing decompose matrix...")
    generate_node_data(name, dataset)
    print("Process done!")


def PreDataLoader(name,net,prefix=None):
    name = name.lower()
    if name in ['toygraph_sin', 'toygraph_esin']:
        print('loading torgraph...')
        dataset = Toy_graph(name)
        return dataset
    current_dir = os.path.dirname(os.path.abspath(__file__))
    path = f'{current_dir}/../Decomposed_data/{name}_decom.pt'
    if not os.path.exists(path):    # preprocess datasets
        data_path = f'{current_dir}/../data'
        if name in ['cora', 'citeseer', 'pubmed']:
            dataset = Planetoid(data_path, name, transform=T.NormalizeFeatures())
        elif name in ['chameleon', 'squirrel']:
            dataset = WikipediaNetwork(data_path, name, transform=T.NormalizeFeatures())
        elif name in ['computers', 'photo']:
            dataset = Amazon(data_path, name, transform=T.NormalizeFeatures())
        elif name in ['actor']:
            dataset = Actor(f'{data_path}/actor')
        decompose_matrix(name, dataset)      
    if net == 'WaveNet':
        dataset = Sameple_data(root='Decomposed_data/', name=name)
    else:
        dataset = load_data(name)
    return dataset

def load_data(name):
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = f'{current_dir}/../data'
    if name in ['cora', 'citeseer', 'pubmed']:
        dataset = Planetoid(data_path, name, transform=T.NormalizeFeatures())
    elif name in ['chameleon', 'squirrel']:
        dataset = WikipediaNetwork(data_path, name, transform=T.NormalizeFeatures())
    elif name in ['computers', 'photo']:
        dataset = Amazon(data_path, name, transform=T.NormalizeFeatures())
    elif name in ['actor']:
        dataset = Actor(f'{data_path}/actor')
    return dataset    


def generate_subgraph_ogb(data_path, name, k=3000, rep=10, mode='node', prefix=None):
    """
    k: num of sampled nodes/edges
    rep: num of sample graphs
    mode: 'node' 'edge'
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    if name in ['arxiv']:
        data = DglNodePropPredDataset(name='ogbn-arxiv',root=data_path)
    else:
        raise NameError
    g,labels = data[0]      
    split_idx = data.get_idx_split()
    train_idx, valid_idx, test_idx = split_idx['train'], split_idx['valid'], split_idx['test']
    all_graphs = []

    if mode == 'node':
        g = dgl.add_reverse_edges(g)
        g = dgl.to_simple(g)
        for i in range(rep):
            random_indices = torch.randperm(train_idx.shape[0])[:k]
            random_samples = train_idx[random_indices]
            node_sample = random_samples.tolist()
            subgraph = g.subgraph(node_sample)
            y=labels[node_sample]
            x=subgraph.ndata['feat']
            adj=subgraph.adj().to_dense()

            x = x.to_dense()
            x = feature_normalize(x)

            e, u = eigen_decompositon(adj)      

            e = torch.FloatTensor(e)
            u = torch.FloatTensor(u)
            x = torch.FloatTensor(x)
            y = torch.LongTensor(y)
            all_graphs.append([e, u, x, y])
            print(f'{i} has beed decomposed')

        torch.save(all_graphs,  f'{current_dir}/../Decomposed_data/{name}_decom_{prefix}.pt')
    elif mode == 'edge':
        train_subgraph = g.subgraph(train_idx)
        train_edges = train_subgraph.edges()
        edges_set = {(u.item(), v.item()) for u, v in zip(*train_edges)}
        for i in range(rep):
            subedges_set = set(random.sample(edges_set, k))
            edges_set -= subedges_set

            subedges_u, subedges_v = zip(*subedges_set)
            edge_ids = train_subgraph.edge_ids(torch.tensor(subedges_u), torch.tensor(subedges_v)).tolist()
            subgraph = train_subgraph.edge_subgraph(edge_ids)
            subgraph = dgl.add_reverse_edges(subgraph)
            subgraph = dgl.to_simple(subgraph)
            
            node_ids = subgraph.ndata[dgl.NID].tolist()
            y=labels[node_ids]
            x=subgraph.ndata['feat']
            adj=subgraph.adj().to_dense()

            e, u = eigen_decompositon(adj)      

            e = torch.FloatTensor(e)
            u = torch.FloatTensor(u)
            x = torch.FloatTensor(x)
            y = torch.LongTensor(y)
            all_graphs.append([e, u, x, y])
            print(f'{i} has beed decomposed')
        torch.save(all_graphs,  f'{current_dir}/../Decomposed_data/{name}_decom_{k}_{rep}_edge.pt')

    print('all saved')

