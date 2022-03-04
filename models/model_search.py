import os
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from collections import OrderedDict
from collections import namedtuple
from models.cell_search import Cell
from models.operations import V_OPS, E_OPS, get_OPS
from models.networks import MLP
from data import TransInput, TransOutput, get_trans_input
from hydra.utils import get_original_cwd, to_absolute_path

Genotype = namedtuple('Genotype', 'V E')

def expand_genotype(src_genotypes):
    nb_layers = len(src_genotypes)
    nb_nodes  = max( x[0] for x in src_genotypes[0].V) - 1
    topology = []
    weight_config = []
    weight_order = 0
    for cell_id in range(nb_layers):

        src_type_dict = {'V': {}, 'E': {}}
        for type in ['V', 'E']:
            src_type = getattr(src_genotypes[cell_id],type)
            for Si, Vj, Ek, Op, _ in src_type:
                if Si not in src_type_dict[type]:
                    src_type_dict[type][Si] = []
                src_type_dict[type][Si].append((Vj, Ek, Op))

        cell_topo = {'V': {}, 'E': {}}
        for Si in range(2, nb_nodes + 2):
            
            for type in ['V', 'E']:
                ops = get_OPS(type)
                cell_topo[type][(Si-1)*2] = []
                cell_topo[type][(Si-1)*2+1] = []
                for Vj, Ek, Op in src_type_dict[type][Si]:
                    
                    nSi = (Si-1) * 2
                    nVj = Vj if Vj < 2 else (Vj - 1) * 2 + 1 
                    nEk = Ek if Ek < 2 else (Ek - 1) * 2 + 1
                    edge = {
                        'Vj': nVj,
                        'Ek': nEk,
                        'weight_order': -1, 
                        'selected': ops.index(Op), 
                    }
                    cell_topo[type][nSi].append(edge)
                    
                    nSi = (Si-1)*2 + 1
                    nVj = Vj if Vj < 2 else (Vj - 1) * 2 + 1 
                    nEk = Ek if Ek < 2 else (Ek - 1) * 2 + 1
                    edge = {
                        'Vj': nVj, 
                        'Ek': nEk, 
                        'weight_order': weight_order, 
                        'selected': -1,
                    }
                    weight_order += 1
                    weight_config.append(len(ops))
                    cell_topo[type][nSi].append(edge)
            
                nSi = (Si-1)*2 + 1
                nVj = nEk = (Si-1)*2
                edge = {
                    'Vj': nVj,
                    'Ek': nEk,
                    'weight_order': weight_order,
                    'selected': -1,
                }
                weight_order += 1
                weight_config.append(len(ops))
                cell_topo[type][nSi].append(edge)
        
        topology.append(cell_topo)
    return topology, weight_config


# if __name__ == '__main__':
#     with open("archs/new_sgas/ZINC/50/cell_geno.txt", "r") as f:
#         src_genotypes = eval(f.read())
#     r1, r2 = expand_genotype(src_genotypes)
#     print(r1)
#     print(r2)
#     import ipdb; ipdb.set_trace()


def load_arch_basic(nb_nodes, nb_layers):
    #! 获取纯结构的拓扑字典
    topology = []
    weight_config = []
    weight_order = 0
    for cell_id in range(nb_layers):
        cell_topo = {}

        for type in ['V', 'E']: 
            type_topo = {}
            ops = get_OPS(type)

            for Si in range(2, nb_nodes + 2):
                incomings = []
                for Vj in range(Si):
                    edge = {
                        'Vj': Vj, 
                        'Ek': Vj, 
                        'weight_order': weight_order,
                        'selected': -1, 
                    }
                    weight_config.append(len(ops))
                    incomings.append(edge)
                    weight_order += 1
                type_topo[Si] = incomings
            
            cell_topo[type] = type_topo
        
        topology.append(cell_topo)
    return topology, weight_config


class Model_Search(nn.Module):

    def __init__(self, args, trans_input_fn, loss_fn):
        super().__init__()
        self.args = args
        self.nb_layers = args.basic.nb_layers
        if args.ds.expand_from == "":
            self.arch_topo, para = load_arch_basic(args.basic.nb_nodes, self.nb_layers)
        else:
            path = os.path.join(get_original_cwd(), args.ds.expand_from)
            with open(path, "r") as f: 
                src_genotypes = eval(f.read())
                self.arch_topo, para = expand_genotype(src_genotypes)

        self.arch_para      = self.init_arch_para(para) 
        self.cells          = nn.ModuleList([Cell(args, self.arch_topo[i], last = (i==self.nb_layers-1)) for i in range(self.nb_layers)])
        self.loss_fn        = loss_fn
        self.trans_input_fn = trans_input_fn
        self.trans_input    = TransInput(trans_input_fn)
        self.trans_output   = TransOutput(args)
        if args.ds.pos_encode > 0:
            self.position_encoding = nn.Linear(args.ds.pos_encode, args.ds.node_dim)


    def forward(self, input):

        input = self.trans_input(input)
        G, V, E   = input['G'], input['V'], input['E']
        if "arch_topo" not in input or input['arch_topo'] is None: 
            arch_topo = self.arch_topo
        else: 
            arch_topo = input['arch_topo']

        if self.args.ds.pos_encode > 0:
            V = V + self.position_encoding(G.ndata['pos_enc'].float().cuda())
        
        output = {'G': G, 'V0': V, 'V1': V, 'E0': E, 'E1': E, 'arch_para': self.arch_para}
        for i, cell in enumerate(self.cells):
            output['cell_topo'] = arch_topo[i]
            output = cell(output)
        
        output.update({'V': output['V1'], 'E': output['E1']})
        output = self.trans_output(output)
        return output


    def init_arch_para(self, para):
        
        arch_para = []
        for plen in para:
            arch_para.append(Variable(1e-3 * torch.rand(plen).cuda(), requires_grad = True))
        return arch_para
    
        # total = len(para)
        # arch_para = []
        # if 'V' in self.args.optimizer.fix:
        #     requires_grad = False
        #     eps = 1
        # else:
        #     requires_grad = True
        #     eps = 1e-3
        
        # for plen in para[:total//2]:
        #     arch_para.append(Variable(eps * torch.rand(plen).cuda(), requires_grad = requires_grad))

        # if 'E' in self.args.optimizer.fix:
        #     requires_grad = False
        #     eps = 1
        # else:
        #     requires_grad = True
        #     eps = 1e-3
        
        # for plen in para[total//2:]:
        #     arch_para.append(Variable(eps * torch.rand(plen).cuda(), requires_grad = requires_grad))
        # return arch_para


    def new(self):
        model_new = Model_Search(self.args, get_trans_input(self.args), self.loss_fn).cuda()
        for x, y in zip(model_new.arch_parameters(), self.arch_parameters()):
            x.data.copy_(y.data)
        return model_new
    

    def load_alpha(self, alphas):
        for x, y in zip(self.arch_parameters(), alphas):
            x.data.copy_(y.data)


    def arch_parameters(self):
        return self.arch_para


    def _loss(self, input, targets):
        scores = self.forward(input)
        return self.loss_fn(scores, targets)