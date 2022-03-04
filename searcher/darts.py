import dgl
import torch
import numpy as np
from tqdm import tqdm
from rich.console import Console
from collections import namedtuple
from models.operations import V_OPS, E_OPS, get_OPS
from searcher.architect import Architect
Genotype = namedtuple('Genotype', 'V E')


class DARTS:

    def __init__(self, args_dict): 
        self.max_nb_edges  = 2
        self.args          = args_dict['args']
        self.model_search  = args_dict['model_search']
        self.arch_queue    = args_dict['arch_queue']
        self.para_queue    = args_dict['para_queue']
        self.loss_fn       = args_dict['loss_fn']
        self.metric        = args_dict['metric']
        self.optimizer     = args_dict['optimizer']
        self.architect     = Architect(self.model_search, self.args)
        self.console       = Console()
        self.arch_topo     = None
    
    def search(self, args_dict):

        lr = args_dict['lr']

        self.model_search.train()
        epoch_loss   = 0
        epoch_metric = 0
        device       = torch.device('cuda')
        with tqdm(self.para_queue, desc = '=> searching by darts <=', leave = False) as t:
            for i_step, (batch_graphs, batch_targets) in enumerate(t):
                #! 1. 准备训练集数据
                G = batch_graphs.to(device)
                V = batch_graphs.ndata['feat'].to(device)
                E = batch_graphs.edata['feat'].to(device)
                batch_targets = batch_targets.to(device)
                #! 2. 准备验证集数据
                batch_graphs_search, batch_targets_search = next(iter(self.arch_queue))
                GS = batch_graphs_search.to(device)
                VS = batch_graphs_search.ndata['feat'].to(device)
                ES = batch_graphs_search.edata['feat'].to(device)
                batch_targets_search = batch_targets_search.to(device)
                #! 3. 优化结构参数
                self.architect.step(
                    input_train       = {'G': G, 'V': V, 'E': E, 'arch_topo': self.arch_topo},
                    target_train      = batch_targets,
                    input_valid       = {'G': GS, 'V': VS, 'E': ES, 'arch_topo': self.arch_topo},
                    target_valid      = batch_targets_search,
                    eta               = lr,
                    network_optimizer = self.optimizer,
                    unrolled          = self.args.optimizer.unrolled
                )
                #! 4. 优化模型参数
                self.optimizer.zero_grad()
                batch_scores  = self.model_search({'G': G, 'V': V, 'E': E, 'arch_topo': self.arch_topo})
                loss          = self.loss_fn(batch_scores, batch_targets)
                loss.backward()
                self.optimizer.step()

                epoch_loss   += loss.detach().item()
                epoch_metric += self.metric(batch_scores, batch_targets)
                t.set_postfix(lr         = lr,
                              loss       = epoch_loss / (i_step + 1),
                              metric     = epoch_metric / (i_step + 1))

        return {'loss'   : epoch_loss / (i_step + 1), 
                'metric' : epoch_metric / (i_step + 1)}


    def genotypes(self, show = True):
        genotypes = []
        arch_para = self.model_search.arch_parameters()
        arch_topo = self.model_search.arch_topo
        arch_para = [alpha.softmax(0) for alpha in arch_para]

        for i, cell_topo in enumerate(arch_topo):

            if show:
                self.console.log(f"[red]Cell => {i}")

            result = {}
            for type in ['V', 'E']: 

                if show:
                    self.console.log(f"Type => {type}")
                
                result[type] = []
                ops = get_OPS(type)
                
                for Si, incomings in cell_topo[type].items():

                    if show: 
                        self.console.log(f'Node => {Si}, incoming edges: ')
                        for edge in incomings:
                            Vj, Ek, order = edge['Vj'], edge['Ek'], edge['weight_order']
                            grad = arch_para[order].requires_grad
                            para = list(arch_para[order].detach().cpu().numpy())
                            para = [f"{x:.4f}" for x in para]
                            self.console.log(f"Vj: {Vj}, Ek: {Ek}, arch: {para}, order: {order}, grad: {grad}")

                    fix_incomings = [edge for edge in incomings if edge['selected'] != -1]
                    for edge in fix_incomings:
                        result[type].append((Si, edge['Vj'], edge['Ek'], ops[edge['selected']]))
                    
                    act_incomings  = [edge for edge in incomings if edge['selected'] == -1]
                    best_incomings = sorted(
                        act_incomings, 
                        key = lambda edge : -max(arch_para[edge["weight_order"]][1:])
                    )[:self.max_nb_edges]
                
                    for edge in best_incomings:
                        op = torch.argmax(arch_para[edge['weight_order']][1:]) + 1
                        result[type].append((Si, edge['Vj'], edge['Ek'], ops[op]))

            genotypes.append( Genotype(V = result['V'], E = result['E']) )

        return genotypes

