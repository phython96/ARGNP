import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.operations import get_package, get_OPS, OPS
from models.networks import MLP


class Mixed(nn.Module):
    
    def __init__(self, args, type):
        super().__init__()
        self.args       = args
        self.type       = type
        self._ops       = get_OPS(type)
        self.candidates = nn.ModuleDict({
            name: get_package(type)(args, OPS[name](args), name, affine = False)
            for name in self._ops
        })
    

    def forward(self, input: tuple, weight: dict, selected_idx: int = None):
        if selected_idx is None or selected_idx == -1:
            weight = weight.softmax(0)
            return sum( weight[i] * self.candidates[name](input) for i, name in enumerate(self._ops) )
        else:
            selected_op = self._ops[selected_idx]
            return self.candidates[selected_op](input)


class Cell(nn.Module):

    def __init__(self, args, cell_arch, last = False):
        super().__init__()
        self.args      = args
        self.cell_arch = cell_arch
        self.nb_nodes  = max(cell_arch['V'].keys()) - 1
        total_nodes    = self.nb_nodes + 2
        self.trans_output_V = nn.Sequential( MLP((total_nodes*args.ds.node_dim, args.ds.inter_channel_V), "leakyrelu", "batch", False) )
        self.trans_output_E = nn.Sequential( MLP((total_nodes*args.ds.edge_dim, args.ds.inter_channel_E), "leakyrelu", "batch", False) )
        if not last:
            self.trans_output_V.add_module( "fin_V", nn.Linear(args.ds.inter_channel_V, args.ds.node_dim, bias=False) )
            self.trans_output_E.add_module( "fin_E", nn.Linear(args.ds.inter_channel_E, args.ds.edge_dim, bias=False) )
        self.activate = nn.LeakyReLU(negative_slope=0.2)

        # self.trans_concat_V = nn.Linear(self.nb_nodes * args.ds.node_dim, args.ds.node_dim, bias = True)
        # self.trans_concat_E = nn.Linear(self.nb_nodes * args.ds.edge_dim, args.ds.edge_dim, bias = True)
        # self.batchnorm_V    = nn.BatchNorm1d(args.ds.node_dim)
        # self.batchnorm_E    = nn.BatchNorm1d(args.ds.edge_dim)
        # self.activate       = nn.LeakyReLU(negative_slope=0.2)

        self.load_arch()


    def load_arch(self):
        for type in ['V', 'E']: 
            link_para = {}
            for Si, incomings in self.cell_arch[type].items():
                for edge in incomings:
                    Vj, Ek, selected = edge['Vj'], edge['Ek'], edge['selected']
                    link_para[str((Si, Vj, Ek))] = Mixed(self.args, type)
            setattr(self, f'link_para_{type}', nn.ModuleDict(link_para))


    def forward(self, input):

        V0, V1, E0, E1 = input['V0'], input['V1'], input['E0'], input['E1']
        G, arch_para, cell_topo = input['G'], input['arch_para'], input['cell_topo']
        Vin, Ein = V1, E1   

        states = {
            'V': [V0, V1], 
            'E': [E0, E1],
        }

        for Si in range(2, self.nb_nodes + 2): 
            for type in ['V', 'E']: 
                tmp_state = []
                incomings = cell_topo[type][Si]
                for edge in incomings:
                    Vj, Ek, weight_order, selected = edge['Vj'], edge['Ek'], edge['weight_order'], edge['selected']
                    input_v = states['V'][Vj]
                    input_e = states['E'][Ek]
                    func    = getattr(self, f'link_para_{type}')[str((Si, Vj, Ek))]
                    output  = func((G, input_v, input_e), arch_para[weight_order], selected)
                    tmp_state.append(output)
                states[type].append(sum(tmp_state))
        
        for type in ['V', 'E']: 
            states[type] = [self.activate(f) for f in states[type]]

        V = self.trans_output_V(torch.cat(states['V'], dim = 1))
        E = self.trans_output_E(torch.cat(states['E'], dim = 1))

        # V = self.trans_concat_V(torch.cat(states['V'][2:], dim = 1))
        # E = self.trans_concat_E(torch.cat(states['E'][2:], dim = 1))
        # V = self.batchnorm_V(V)
        # E = self.batchnorm_E(E) 
        # V = self.activate(V)
        # E = self.activate(E)
        # V = F.dropout(V, self.args.ds.dropout, training = self.training)
        # E = F.dropout(E, self.args.ds.dropout, training = self.training)
        # V = V + Vin
        # E = E + Ein
        output = {**input}
        output.update({'V0': V1, 'V1': V, 'E0': E1, 'E1': E})
        return output



