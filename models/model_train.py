import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from models.cell_train import Cell
from models.operations import V_OPS, E_OPS
from models.networks import MLP
from data import TransInput, TransOutput

class Model_Train(nn.Module):
    
    def __init__(self, args, genotypes, trans_input_fn, loss_fn):
        super().__init__()
        self.args           = args
        self.nb_layers      = args.nb_layers
        self.genotypes      = genotypes
        self.cells          = nn.ModuleList([Cell(args, genotypes[i], last = (i==self.nb_layers-1)) for i in range(self.nb_layers)])
        self.loss_fn        = loss_fn
        self.trans_input_fn = trans_input_fn
        self.trans_input    = TransInput(trans_input_fn)
        self.trans_output   = TransOutput(args)

        if args.ds.pos_encode > 0:
            self.position_encoding = nn.Linear(args.ds.pos_encode, args.ds.node_dim)

    
    def forward(self, input):
        input = self.trans_input(input)
        G, V, E = input['G'], input['V'], input['E']
        if self.args.ds.pos_encode > 0:
            V = V + self.position_encoding(G.ndata['pos_enc'].float().to("cuda"))
        output = {'G': G, 'V0': V, 'V1': V, 'E0': E, 'E1': E}
        for i, cell in enumerate(self.cells):
            output = cell(output)
        
        output.update({'V': output['V1'], 'E': output['E1']})
        output = self.trans_output(output)
        return output
    
    
    def _loss(self, input, targets):
        scores = self.forward(input)
        return self.loss_fn(scores, targets)