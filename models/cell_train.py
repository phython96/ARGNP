import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.operations import get_package, OPS
from models.networks import MLP

class Cell(nn.Module):

    def __init__(self, args, genotype, last = False):

        super().__init__()
        self.args     = args
        self.nb_nodes = args.nb_nodes
        self.genotype = genotype
        total_nodes   = self.nb_nodes + 2
        self.trans_output_V = nn.Sequential( MLP((total_nodes*args.ds.node_dim, args.ds.inter_channel_V), "leakyrelu", "batch", False) )
        self.trans_output_E = nn.Sequential( MLP((total_nodes*args.ds.edge_dim, args.ds.inter_channel_E), "leakyrelu", "batch", False) )
        if not last:
            self.trans_output_V.add_module( "fin_V", nn.Linear(args.ds.inter_channel_V, args.ds.node_dim, bias=False) )
            self.trans_output_E.add_module( "fin_E", nn.Linear(args.ds.inter_channel_E, args.ds.edge_dim, bias=False) )
        self.activate = nn.LeakyReLU(negative_slope=0.2)
        self.load_genotype('V')
        self.load_genotype('E')
    

    def load_genotype(self, type):
        if type == 'V': 
            genotype = self.genotype.V
        elif type == 'E': 
            genotype = self.genotype.E
        else: 
            raise Exception('Unknown genotype type!')

        link_dict   = {}
        module_dict = {}
        for Si, Vj, Ek, Op in genotype:

            Si = f'{Si}'

            if Si not in link_dict:
                link_dict[Si] = []
            link_dict[Si].append((Vj, Ek))

            if Si not in module_dict:
                module_dict[Si] = nn.ModuleList([])
            module_dict[Si].append(get_package(type)(self.args, OPS[Op](self.args), Op, affine=True)) 
        setattr(self, f'module_dict_{type}', nn.ModuleDict(module_dict))
        setattr(self, f'link_dict_{type}', link_dict)
    

    def forward(self, input):
    
        G, V0, V1, E0, E1 = input['G'], input['V0'], input['V1'], input['E0'], input['E1']
        states_V, states_E = [V0, V1], [E0, E1]
        for Si in range(2, self.nb_nodes + 2):
            Si = f'{Si}'
            agg = []
            for i, (Vj, Ek) in enumerate(self.link_dict_V[Si]):
                hs = self.module_dict_V[Si][i]((G, states_V[Vj], states_E[Ek]))
                agg.append(hs)
            states_V.append(sum(agg))

            agg = []
            for i, (Vj, Ek) in enumerate(self.link_dict_E[Si]):
                hs = self.module_dict_E[Si][i]((G, states_V[Vj], states_E[Ek]))
                agg.append(hs)
            states_E.append(sum(agg))

        states_V = [self.activate(f) for f in states_V]
        states_E = [self.activate(f) for f in states_E]
        # record
        self.Es = [x.cpu() for x in states_E]
        self.Vs = [x.cpu() for x in states_V]

        V = self.trans_output_V(torch.cat(states_V, dim = 1))
        E = self.trans_output_E(torch.cat(states_E, dim = 1))

        output = {**input}
        output.update({'V0': V1, 'V1': V, 'E0': E1, 'E1': E})
        return output


