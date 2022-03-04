import torch
import dgl
import dgl.function as fn
import torch.nn as nn
import numpy as np
from models.networks import MLP
from utils.visualize import *


OPS = {
    'V_None' : lambda args: V_None(args),
    'V_I'    : lambda args: V_I(args),
    'V_Max'  : lambda args: V_Max(args),
    'V_Mean' : lambda args: V_Mean(args),
    'V_Sum'  : lambda args: V_Sum(args),
    'V_Std'  : lambda args: V_Std(args),
    'V_Min'  : lambda args: V_Min(args),
    'V_Gem2' : lambda args: V_Gem2(args),
    'V_Gem3' : lambda args: V_Gem3(args),
    'E_None' : lambda args: E_None(args),
    'E_I'    : lambda args: E_I(args),
    'E_Sub'  : lambda args: E_Sub(args),
    'E_Gauss': lambda args: E_Gauss(args),
    'E_Max'  : lambda args: E_Max(args),
    'E_Sum'  : lambda args: E_Sum(args),
    'E_Mean' : lambda args: E_Mean(args),
    'E_Had'  : lambda args: E_Had(args),
    'E_Src'  : lambda args: E_Src(args),
    'E_Dst'  : lambda args: E_Dst(args),
    # 'E_Cat'  : lambda args: E_Cat(args),
    # 'E_GRU'  : lambda args: E_GRU(args),
    # 'E_FiLM' : lambda args: E_FiLM(args),
}


V_OPS = ['V_None', 'V_I', 'V_Mean', 'V_Sum', 'V_Max', 'V_Std', 'V_Gem2', 'V_Gem3']
E_OPS = ['E_None', 'E_I', 'E_Sub', 'E_Gauss', 'E_Max', 'E_Sum', 'E_Mean', 'E_Had']


def get_OPS(type):
    if type == 'V':
        return V_OPS
    elif type == 'E':
        return E_OPS
    else: 
        raise Exception('Unknown OPS!')


class V_Package(nn.Module):

    def __init__(self, args, operation, operation_name, affine = True):
        
        super().__init__()
        self.args = args
        self.operation = operation
        self.operation_name = operation_name
        if operation_name in ['V_None', 'V_I']:
            self.seq = None
        else:
            self.seq = nn.Sequential()
            self.seq.add_module('fc_bn', nn.Linear(args.ds.node_dim, args.ds.node_dim, bias = True))
            self.seq.add_module('bn', nn.BatchNorm1d(self.args.ds.node_dim, affine=affine))

    def forward(self, input):
        V = self.operation(input)
        if self.seq:
            V = self.seq(V)
        return V 


class E_Package(nn.Module):

    def __init__(self, args, operation, operation_name, affine = True):

        super().__init__()
        self.args = args
        self.operation = operation
        self.operation_name = operation_name
        if operation_name in ['E_None', 'E_I']:
            self.seq = None
        else:
            self.seq = nn.Sequential()
            self.seq.add_module('fc_bn', nn.Linear(args.ds.edge_dim, args.ds.edge_dim, bias = True))
            self.seq.add_module('bn', nn.BatchNorm1d(self.args.ds.edge_dim, affine=affine))

    def forward(self, input):
        E = self.operation(input)
        if self.seq:
            E = self.seq(E)
        return E


def get_package(type):

    if type == 'V': 
        return V_Package
    elif type == 'E': 
        return E_Package
    else:
        raise Exception('Unknown operation type!')


class GenMessage(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args        = args
        self.A           = MLP((self.args.ds.edge_dim, ) * (self.args.basic.nb_mlp_layer + 1))
        # self.A           = MLP((self.args.ds.edge_dim, ) * (self.args.basic.nb_mlp_layer + 1), "leakyrelu", "batch")
        self.B           = nn.Linear(args.ds.edge_dim, args.ds.node_dim, bias = True)
        self.C           = nn.Linear(args.ds.edge_dim, args.ds.node_dim, bias = True)

    def forward(self, Vs, Vt, Ei):
        if self.args.basic.edge_feature:
            x     = self.A(Ei)
            scale = torch.sigmoid(self.B(x))
            shift = self.C(x)
            msg   = scale * Vs + shift
            return msg
        else:
            return Vs

# class GenMessage(nn.Module):

#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         if self.args.basic.edge_feature:
#             self.C1 = nn.Linear(args.ds.node_dim*2, args.ds.node_dim, bias = False)
#             self.C2 = nn.Linear(args.ds.node_dim*2, args.ds.node_dim, bias = False)
#             self.K  = nn.Linear(args.ds.edge_dim, args.ds.node_dim, bias = False)
#         self.B = nn.Linear(args.ds.edge_dim, args.ds.node_dim, bias = False)

#     def forward(self, Vs, Vt, Ei):
#         nodes = torch.cat([Vs, Vt], dim = -1)
#         if self.args.basic.edge_feature:
#             message = torch.sigmoid(self.K(Ei)) * self.C1(nodes) + self.B(Ei) + self.C2(nodes)
#         else:
#             message = torch.B(nodes)
#         return message

# class GenMessage(nn.Module):
    
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         inter_dim = args.ds.edge_dim + args.ds.node_dim
#         self.U = nn.Linear(args.ds.node_dim * 3, inter_dim, bias = True)
#         self.V = nn.Linear(args.ds.edge_dim, inter_dim, bias = True)
#         self.P = nn.Linear(inter_dim, args.ds.node_dim, bias = True)

#     def forward(self, Vs, Vt, Ei):
#         message = torch.cat([Vs, Vt, Vs - Vt], dim = -1)
#         if self.args.basic.edge_feature:
#             message = self.P(torch.tanh(self.U(message) * self.V(Ei)))
#         else:
#             message = self.P(torch.tanh(self.U(message)))
#         return message

class NodePooling(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.A   = nn.Linear(args.ds.node_dim, args.ds.node_dim)
        self.B   = nn.Linear(args.ds.node_dim, args.ds.node_dim)
        self.act = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, V):
        V = self.A(V)
        V = self.act(V)
        V = self.B(V)
        return V


class V_None(nn.Module):

    def __init__(self, args):
        super().__init__()
    
    def forward(self, input):
        G, V, E = input
        return V.mul(0.)


class V_I(nn.Module):
    
    def __init__(self, args):
        super().__init__()
    
    def forward(self, input):
        G, V, E = input
        return V


class V_Basic(nn.Module):
    
    def __init__(self, args, fn_agg):
        super().__init__()
        self.args       = args
        self.fn_agg     = fn_agg
        self.pooling    = NodePooling(args)
        self.message_fn = GenMessage(args)
        self.act        = nn.LeakyReLU(negative_slope=0.2)

    def messages(self, edges):
        M = self.message_fn(edges.src['V'], edges.dst['V'], edges.data['E'])
        return {'M' : M}

    def forward(self, input):
        G, V, E = input
        V = self.act(V)
        G.ndata['V']  = self.pooling(V)
        G.edata['E']  = E
        G.update_all(self.messages, self.fn_agg('M', 'V'))
        return G.ndata['V']


class V_Max(V_Basic):

    def __init__(self, args):
        super().__init__(args, fn.max)


class V_Mean(V_Basic):
    
    def __init__(self, args):
        super().__init__(args, fn.mean)


class V_Sum(V_Basic):

    def __init__(self, args):
        super().__init__(args, fn.sum)

class V_Min(V_Basic):
    
    def __init__(self, args):
        super().__init__(args, fn.min)

class V_Std(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.pooling = NodePooling(args)
        self.message_fn = GenMessage(args)
        self.act        = nn.LeakyReLU(negative_slope=0.2)

    def messages(self, edges):
        M1 = self.message_fn(edges.src['V'], edges.dst['V'], edges.data['E'])
        M2 = M1 * M1
        return {'M1': M1, 'M2': M2}
    
    def placeholder1(self, edges):
        return {'M': edges.data['M1']}

    def placeholder2(self, edges):
        return {'M': edges.data['M2']}

    def forward(self, input):
        G, V, E = input
        V = self.act(V)
        G.ndata['V'] = self.pooling(V)
        G.edata['E'] = E
        G.apply_edges(self.messages)
        G.update_all(self.placeholder1, fn.mean('M', 'V1'))
        G.update_all(self.placeholder2, fn.mean('M', 'V2'))
        return torch.sqrt(torch.relu(G.ndata['V2'] - G.ndata['V1']) + 1e-5)


class V_Gem(nn.Module):

    def __init__(self, args, alpha):
        super().__init__()
        self.alpha = alpha
        self.pooling = NodePooling(args)
        self.message_fn = GenMessage(args)
        self.act        = nn.LeakyReLU(negative_slope=0.2)

    def messages(self, edges, eps = 1e-5):
        M = self.message_fn(edges.src['V'], edges.dst['V'], edges.data['E'])
        M = torch.pow(M.clamp(min = eps), self.alpha)
        return {'M' : M}

    def forward(self, input):
        G, V, E = input
        V = self.act(V)
        G.ndata['V'] = self.pooling(V)
        G.edata['E'] = E
        G.update_all(self.messages, fn.mean('M', 'V'))
        return torch.pow(G.ndata['V'], 1./self.alpha)


class V_Gem2(V_Gem):

    def __init__(self, args):
        super().__init__(args, 2)


class V_Gem3(V_Gem):
    
    def __init__(self, args):
        super().__init__(args, 3)


class E_None(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def forward(self, input):
        G, V, E = input
        return E.mul(0.)

class E_I(nn.Module):
    
    def __init__(self, args):
        super().__init__()
        self.args = args
    
    def forward(self, input):
        G, V, E = input
        return E


class GenRelation(nn.Module):

    def __init__(self, args, in_channel):
        super().__init__()
        self.args        = args
        channel_sequence = (in_channel, ) + (self.args.ds.edge_dim, ) * self.args.basic.nb_mlp_layer
        self.A = MLP(channel_sequence)
        self.B = nn.Linear(args.ds.edge_dim, args.ds.edge_dim, bias = True)
        self.C = nn.Linear(args.ds.edge_dim, args.ds.edge_dim, bias = True)

    def forward(self, Ein, Et):
        Et    = self.A(Et)
        scale = torch.sigmoid(self.B(Et))
        shift = self.C(Et)
        Eout  = scale * Ein + shift
        return Eout


class E_Basic(nn.Module):

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.act  = nn.LeakyReLU(negative_slope=0.2)
        self.relation_fn = GenRelation(args, in_channel = args.ds.node_dim)

    def trans_edges(self, edges):
        raise NotImplementedError

    def forward(self, input): 
        G, V, E = input
        E = self.act(E)
        G.ndata['V'] = V
        G.edata['E'] = E
        G.apply_edges(self.trans_edges)
        return G.edata['E']


class E_Gauss(E_Basic):

    def __init__(self, args):
        super().__init__(args)

    def trans_edges(self, edges):
        src = edges.src['V']
        dst = edges.dst['V']
        distance = ((src - dst) * (src - dst))
        sigma = 1
        Et = torch.exp(-distance / (2 * sigma))
        # print("d", distance)
        # print("e", Et)
        E = self.relation_fn(Ein = edges.data['E'], Et = Et)

        return {'E': E}


class E_Sub(E_Basic):

    def __init__(self, args):
        super().__init__(args)

    def trans_edges(self, edges):
        src = edges.src['V']
        dst = edges.dst['V']
        Et  = torch.abs(src - dst)
        E   = self.relation_fn(Ein = edges.data['E'], Et = Et)
        return {'E': E}


class E_Had(E_Basic):

    def __init__(self, args):
        super().__init__(args)

    def trans_edges(self, edges):
        src = edges.src['V']
        dst = edges.dst['V']
        Et  = src * dst
        E   = self.relation_fn(Ein = edges.data['E'], Et = Et)
        return {'E': E}

class E_Src(E_Basic):
    
    def __init__(self, args):
        super().__init__(args)

    def trans_edges(self, edges):
        Et = edges.src['V']
        E  = self.relation_fn(Ein = edges.data['E'], Et = Et)
        return {'E': E}


class E_Dst(E_Basic):
    
    def __init__(self, args):
        super().__init__(args)

    def trans_edges(self, edges):
        Et = edges.dst['V']
        E  = self.relation_fn(Ein = edges.data['E'], Et = Et)
        return {'E': E}


class E_Sum(E_Basic):
    
    def __init__(self, args):
        super().__init__(args)

    def trans_edges(self, edges):
        Et = edges.src['V'] + edges.dst['V']
        E  = self.relation_fn(Ein = edges.data['E'], Et = Et)
        return {'E': E}


class E_Mean(E_Basic):
    
    def __init__(self, args):
        super().__init__(args)

    def trans_edges(self, edges):
        Et = (edges.src['V'] + edges.dst['V']) / 2
        E  = self.relation_fn(Ein = edges.data['E'], Et = Et)
        return {'E': E}


class E_Max(E_Basic):
    
    def __init__(self, args):
        super().__init__(args)

    def trans_edges(self, edges):
        Et = torch.max(edges.src['V'], edges.dst['V'])
        E  = self.relation_fn(Ein = edges.data['E'], Et = Et)
        return {'E': E}


# class EdgePooling(nn.Module):
    
#     def __init__(self, args):
#         super().__init__()
#         self.args = args
#         self.U = nn.Linear(args.ds.node_dim, args.ds.edge_dim, bias = False)
#         self.V = nn.Linear(args.ds.node_dim, args.ds.edge_dim, bias = False)
#         self.P = nn.Linear(args.ds.edge_dim, args.ds.edge_dim, bias = True)

#     def forward(self, E1, E2):
#         E = self.U(E1) * self.V(E2)
#         E = self.P(torch.tanh(E))
#         return E

# class E_Cat(nn.Module):
    
#     def __init__(self, args):
#         super().__init__()
#         self.args        = args
#         channel_sequence = (2*args.ds.edge_dim, ) + (args.ds.edge_dim, ) * args.basic.nb_mlp_layer
#         self.A           = MLP(channel_sequence)
#         self.pooling     = EdgePooling(args)
#         self.act         = nn.LeakyReLU()
    
#     def trans_edges(self, edges):
#         E = torch.cat([self.pooling(edges.src['V'], edges.dst['V']), edges.data['E']], dim = -1)
#         E = self.A(E)
#         return {'E' : E}

#     def forward(self, input):
#         G, V, E = input
#         E = self.act(E)
#         G.ndata['V'] = V
#         G.edata['E'] = E
#         G.apply_edges(self.trans_edges)
#         return G.edata['E']


# class E_GRU(nn.Module):

#     def __init__(self, args):
#         super().__init__()
#         self.args    = args
#         self.gru     = torch.nn.GRUCell(args.ds.edge_dim, args.ds.edge_dim)
#         self.pooling = EdgePooling(args)
#         self.act     = nn.LeakyReLU()
    
#     def trans_edges(self, edges):
#         E = self.gru(self.pooling(edges.src['V'], edges.dst['V']), edges.data['E'])
#         return {'E' : E}

#     def forward(self, input):
#         G, V, E = input
#         E = self.act(E)
#         G.ndata['V'] = V
#         G.edata['E'] = E
#         G.apply_edges(self.trans_edges)
#         return G.edata['E']


# class E_FiLM(nn.Module):

#     def __init__(self, args):
#         super().__init__()
#         self.args    = args
#         self.B       = nn.Linear(args.ds.edge_dim, args.ds.edge_dim, bias = True)
#         self.C       = nn.Linear(args.ds.edge_dim, args.ds.edge_dim, bias = True)
#         self.pooling = EdgePooling(args)
#         self.act     = nn.LeakyReLU()

#     def trans_edges(self, edges):
#         x     = self.pooling(edges.src['V'], edges.dst['V'])
#         scale = torch.sigmoid(self.B(x))
#         shift = self.C(x)
#         E     = scale * edges.data['E'] + shift
#         return {'E' : E}

#     def forward(self, input):
#         G, V, E = input
#         E = self.act(E)
#         G.ndata['V'] = V
#         G.edata['E'] = E
#         G.apply_edges(self.trans_edges)
#         return G.edata['E']

if __name__ == '__main__':
    print("test")