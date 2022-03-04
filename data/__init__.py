import dgl
import torch
import torch.nn as nn
from data.molecules import MoleculeDataset
from data.QM9 import QM9Dataset
from data.SBMs import SBMsDataset
from data.TSP import TSPDataset
from data.superpixels import SuperPixDataset
from data.cora import CoraDataset
from data.modelnet import ModelnetDataset
from models.networks import MLP
from utils.utils import *


class TransInput(nn.Module):

    def __init__(self, trans_fn):
        super().__init__()
        self.trans_V = trans_fn[0]
        self.trans_E = trans_fn[1]
    
    def forward(self, input):
        output = {**input}
        if self.trans_V:
            V = self.trans_V(input['V'])
            output['V'] = V
        if self.trans_E:
            E = self.trans_E(input['E'])
            output['E'] = E
        return output

class BatchNorm(nn.Module):

    def __init__(self, *args1, **args2):
        super().__init__()
        self.body = nn.BatchNorm2d(*args1, **args2)
    
    def forward(self, x):
        return self.body(x.view(*x.shape, 1, 1)).view(*x.shape)

class TransOutput(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        if args.ds.task == 'node_level': 
            dimension = args.ds.inter_channel_V 
            self.trans = nn.Sequential(
                nn.Linear(dimension, dimension // 2, bias = False), 
                BatchNorm(dimension // 2),
                nn.LeakyReLU(0.2), 
                nn.Dropout(p = args.ds.dropout), 
                nn.Linear(dimension // 2, args.ds.nb_classes), 
            )

        elif args.ds.task == 'link_level':
            if self.args.basic.edge_feature:
                dimension = args.ds.inter_channel_E 
            else:
                dimension = args.ds.inter_channel_V * 2 
            self.trans = nn.Sequential(
                nn.Linear(dimension, dimension // 2, bias = False), 
                BatchNorm(dimension // 2),
                nn.LeakyReLU(0.2), 
                nn.Dropout(p = args.ds.dropout), 
                nn.Linear(dimension // 2, args.ds.nb_classes), 
            )

        elif args.ds.task == 'graph_level':
            if args.basic.edge_feature:
                dimension = args.ds.inter_channel_V + args.ds.inter_channel_E
            else: 
                dimension = args.ds.inter_channel_V 
            self.trans = nn.Sequential(
                nn.Linear(2*dimension, dimension // 2, bias = False), 
                BatchNorm(dimension // 2),
                nn.LeakyReLU(0.2), 
                nn.Dropout(p = args.ds.dropout), 
                nn.Linear(dimension // 2, args.ds.nb_classes), 
            )
            # self.trans = nn.Sequential(
            #     nn.Linear(2*dimension, dimension , bias = False), 
            #     BatchNorm(dimension ),
            #     nn.LeakyReLU(0.2), 
            #     nn.Dropout(p = args.ds.dropout), 
            #     nn.Linear(dimension, dimension // 2), 
            #     BatchNorm(dimension // 2),
            #     nn.LeakyReLU(0.2), 
            #     nn.Dropout(p = args.ds.dropout), 
            #     nn.Linear(dimension // 2, args.ds.nb_classes), 
            # )
            
            # self.trans = nn.Sequential(
            #     MLP((2*dimension, dimension // 2), "leakyrelu", "batch", False), 
            #     nn.Dropout(p = args.ds.dropout), 
            #     nn.Linear(dimension // 2, args.ds.nb_classes), 
            # )
        else:
            raise Exception('Unknown task!')
            

    def forward(self, input):
        G, V, E = input['G'], input['V'], input['E']
        if self.args.ds.task == 'node_level':
            output = self.trans(V)
        elif self.args.ds.task == 'link_level':
            if self.args.basic.edge_feature:
                output = self.trans(E)
            else: 
                def _edge_feat(edges):
                    e = torch.cat([edges.src['V'], edges.dst['V']], dim=1)
                    return {'e': e}
                G.ndata['V'] = V
                G.apply_edges(_edge_feat)
                output = self.trans(G.edata['e'])
        elif self.args.ds.task == 'graph_level':
            G.ndata['V'] = V
            G.edata['E'] = E
            if self.args.basic.edge_feature:
                readout = torch.cat([dgl.mean_nodes(G, 'V'), dgl.max_nodes(G, 'V'), dgl.mean_edges(G, 'E'), dgl.max_edges(G, 'E')], dim = -1)
            else:
                readout = torch.cat([dgl.mean_nodes(G, 'V'), dgl.max_nodes(G, 'V')], dim = -1)
            output = self.trans(readout)
        else:
            raise Exception('Unknown task!')
        return output


def get_trans_input(args):
    if args.ds.data in ['ZINC']:
        trans_input_V = nn.Sequential(
            nn.Embedding(args.ds.in_dim_V, args.ds.node_dim),
        )
        trans_input_E = nn.Sequential(
            nn.Embedding(args.ds.in_dim_E, args.ds.edge_dim),
        )
    elif args.ds.data in ['TSP']:
        trans_input_V = nn.Sequential(
            nn.Linear(args.ds.in_dim_V, args.ds.node_dim),
        )
        trans_input_E = nn.Sequential(
            nn.Linear(args.ds.in_dim_E, args.ds.edge_dim),
        )
    elif args.ds.data in ['SBM_CLUSTER', 'SBM_PATTERN']:
        trans_input_V = nn.Sequential(
            nn.Embedding(args.ds.in_dim_V, args.ds.node_dim),
            nn.BatchNorm1d(args.ds.node_dim),
        )
        trans_input_E = nn.Sequential(
            nn.Linear(args.ds.in_dim_E, args.ds.edge_dim),
        )
    elif args.ds.data in ['CIFAR10', 'MNIST', 'Cora']:
        trans_input_V = nn.Sequential(
            nn.Linear(args.ds.in_dim_V, args.ds.node_dim),
            # nn.BatchNorm1d(args.ds.node_dim),
            # nn.LeakyReLU(),
            # nn.Linear(args.ds.node_dim, args.ds.node_dim),
            # nn.BatchNorm1d(args.ds.node_dim)
        )
        trans_input_E = nn.Sequential(
            nn.Linear(args.ds.in_dim_E, args.ds.edge_dim),
            # nn.BatchNorm1d(args.ds.edge_dim),
            # nn.LeakyReLU(),
            # nn.Linear(args.ds.edge_dim, args.ds.edge_dim),
            # nn.BatchNorm1d(args.ds.edge_dim)
        )
    elif args.ds.data in ['QM9']:
        trans_input_V = nn.Sequential(
            nn.Linear(args.ds.in_dim_V, args.ds.node_dim),
        )
        trans_input_E = nn.Sequential(
            nn.Linear(args.ds.in_dim_E, args.ds.edge_dim),
        )
    elif args.ds.data in ['modelnet40', 'modelnet10']:
        trans_input_V = nn.Sequential(
            nn.Linear(args.ds.in_dim_V, args.ds.node_dim, bias = False),
        )
        trans_input_E = nn.Sequential(
            nn.Linear(args.ds.in_dim_E, args.ds.edge_dim, bias = False),
        )
    else:
        raise Exception('Unknown dataset!')
    return (trans_input_V, trans_input_E)


def get_loss_fn(args):
    if args.ds.data in ['ZINC', 'QM9']:
        loss_fn = MoleculesCriterion()
    elif args.ds.data in ['TSP']:
        loss_fn = TSPCriterion()
    elif args.ds.data in ['SBM_CLUSTER', 'SBM_PATTERN']:
        loss_fn = SBMsCriterion(args.ds.nb_classes)
    elif args.ds.data in ['CIFAR10', 'MNIST', 'modelnet40', 'modelnet10']:
        loss_fn = SuperPixCriterion()
    elif args.ds.data in ['Cora']:
        loss_fn = CiteCriterion()
    else:
        raise Exception('Unknown dataset!')
    return loss_fn


def load_data(args):
    if args.ds.data in ['ZINC']:
        return MoleculeDataset(args.ds.data)
    elif args.ds.data in ['QM9']:
        return QM9Dataset(args.ds.data, args.extra)
    elif args.ds.data in ['TSP']:
        return TSPDataset(args.ds.data)
    elif args.ds.data in ['MNIST', 'CIFAR10']:
        return SuperPixDataset(args.ds.data) 
    elif args.ds.data in ['SBM_CLUSTER', 'SBM_PATTERN']: 
        return SBMsDataset(args.ds.data)
    elif args.ds.data in ['Cora']:
        return CoraDataset(args.ds.data)
    elif args.ds.data in ['modelnet40', 'modelnet10']:
        return ModelnetDataset(args.ds.data, num_points = args.ds.num_points, k = args.ds.k)
    else:
        raise Exception('Unknown dataset!')


def load_metric(args):
    if args.ds.data in ['ZINC', 'QM9']:
        return MAE
    elif args.ds.data in ['TSP']:
        return binary_f1_score
    elif args.ds.data in ['MNIST', 'CIFAR10']:
        return accuracy_MNIST_CIFAR
    elif args.ds.data in ['SBM_CLUSTER', 'SBM_PATTERN']:
        return accuracy_SBM
    elif args.ds.data in ['Cora']:
        return CoraAccuracy
    elif args.ds.data in ['modelnet10', 'modelnet40']:
        return overall
    else:
        raise Exception('Unknown dataset!')
