import dgl
import math
import torch
import numpy as np
import torch.distributions.categorical as cate
from tqdm import tqdm
from collections import namedtuple
from models.operations import V_OPS, E_OPS, get_OPS
from searcher.architect import Architect
from rich.console import Console

Genotype = namedtuple('Genotype', 'V E')

def normalize(v):
    min_v = torch.min(v)
    range_v = torch.max(v) - min_v
    if range_v > 0:
        normalized_v = (v - min_v) / range_v
    else:
        normalized_v = torch.zeros(v.size()).cuda()

    return normalized_v


class SGAS: 

    def __init__(self, args_dict): 
        self.max_nb_edges = 2
        self.args         = args_dict['args']
        self.model_search = args_dict['model_search']
        self.arch_queue   = args_dict['arch_queue']
        self.para_queue   = args_dict['para_queue']
        self.loss_fn      = args_dict['loss_fn']
        self.metric       = args_dict['metric']
        self.optimizer    = args_dict['optimizer']
        self.architect    = Architect(self.model_search, self.args)
        # self.init_sgas_paras()
        self.console   = Console()
        self.cell_id   = 0
        self.arch_topo = self.model_search.arch_topo
        

    def search(self, args_dict):

        lr = args_dict['lr']

        self.model_search.train()
        epoch_loss   = 0
        epoch_metric = 0
        device       = torch.device('cuda')
        with tqdm(self.para_queue, desc = '=> searching by sgas <=', leave = False) as t:
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

        epoch = args_dict['epoch']
        self.edge_decision(epoch)

        return {'loss'   : epoch_loss / (i_step + 1), 
                'metric' : epoch_metric / (i_step + 1)}


    def edge_decision(self, epoch, show = False): 
        
        if self.cell_id >= self.args.basic.nb_layers:
            self.console.log('Decision have been made completely!')
            return False
        
        if show:
            self.console.log(f"[red]Current Cell: {self.cell_id}, Cell Numbers: {self.args.basic.nb_layers}")
        
        arch_topo = self.arch_topo
        cell_topo = arch_topo[self.cell_id]
        arch_para = self.model_search.arch_para

        for type in ['V', 'E']: 
            
            if show:
                self.console.log(f"[red]Type => {type}")

            type_topo = cell_topo[type]
            inverse = []
            alpha_collection = []
            for Si, incomings in type_topo.items():

                if show: 
                    self.console.log(f'Node => {Si}, incoming edges: ')
                    for edge in incomings:
                        Vj, Ek, order = edge['Vj'], edge['Ek'], edge['weight_order']
                        grad = arch_para[order].requires_grad
                        para = list(arch_para[order].detach().softmax(0).cpu().numpy())
                        para = [f"{x:.4f}" for x in para]
                        self.console.log(f"Vj: {Vj}, Ek: {Ek}, arch: {para}, order: {order}, grad: {grad}")

                for id, edge in enumerate(incomings):
                    if edge['selected'] == -1:
                        inverse.append({'Si': Si, 'edge': edge})
                        alpha_collection.append(arch_para[edge['weight_order']])

            stack_alpha_collection = torch.stack(alpha_collection).detach()
            mat = torch.softmax(stack_alpha_collection, dim = -1)

            importance = torch.sum(mat[:, 1:], dim = -1)
            probs = mat[:, 1:] / importance[:, None]
            entropy = cate.Categorical(probs=probs).entropy() / math.log(probs.size()[1])
            score = normalize(importance) * normalize(1 - entropy)

            candidate_flags = torch.tensor([ (value['edge']['selected'] == -1) for value in inverse ])
            
            if epoch >= self.args.optimizer.warmup_dec_epoch and \
               (epoch - self.args.optimizer.warmup_dec_epoch) % self.args.optimizer.decision_freq == 0 and \
               sum(candidate_flags) > 0:
               
                masked_score = torch.min(score, ((2 * candidate_flags.float() - 1) * np.inf).to("cuda"))
                selected_edge_idx = torch.argmax(masked_score)
                selected_op_idx = torch.argmax(probs[selected_edge_idx]) + 1

                edge = inverse[selected_edge_idx]['edge']
                edge['selected'] = selected_op_idx.item()
                arch_para[edge['weight_order']].requires_grad = False
                self.console.log(f"[red][bold]selected_edge_idx: {edge['weight_order']}, selected_op: {selected_op_idx}")

            if self.check_edges(type_topo, arch_para):
                self.cell_id += 1

        return self.cell_id < self.args.basic.nb_layers


    def check_edges(self, type_topo, arch_para):
        for Si, incomings in type_topo.items():
            num_selected_edges = 0
            for edge in incomings:
                if edge['selected'] > -1:
                    num_selected_edges += 1
            if num_selected_edges >= self.max_nb_edges:
                for edge in incomings:
                    if edge['selected'] == -1:
                        edge['selected'] = 0
                        arch_para[edge['weight_order']].requires_grad = False
                    else:
                        pass
        
        for Si, incomings in type_topo.items():
            for edge in incomings:
                if edge['selected'] == -1:
                    return False
    
        return True


    def genotypes(self, show = True):
        genotypes = []
        arch_topo = self.arch_topo
        arch_para = self.model_search.arch_para
        arch_para = [alpha.softmax(0) for alpha in arch_para]

        for i, cell_topo in enumerate(arch_topo):

            if show:
                self.console.log(f"[red]Cell => {i}")

            result = {}
            for type in ['V', 'E']:

                if show:
                    self.console.log(f"Type => {type}")

                type_topo = cell_topo[type]
                ops = get_OPS(type)
                result[type] = []

                for Si, incomings in type_topo.items():

                    if show: 
                        self.console.log(f'Node => {Si}, incoming edges: ')
                        for edge in incomings:
                            Vj, Ek, order = edge['Vj'], edge['Ek'], edge['weight_order']
                            if order > -1:
                                para = list(arch_para[order].detach().cpu().numpy())
                                grad = arch_para[order].requires_grad
                            else:
                                para = list(arch_para[order].mul(0.).detach().cpu().numpy())
                                grad = "Fixed Edge"
                            para = [f"{x:.4f}" for x in para]
                            self.console.log(f"Vj: {Vj}, Ek: {Ek}, arch: {para}, order: {order}, grad: {grad}, selected: {edge['selected']}")

                    inverse = []
                    alpha_collection = []
                    num_selected_edges = 0
                    for id, edge in enumerate(incomings):
                        if edge['selected'] > 0:
                            num_selected_edges += 1
                            result[type].append( (Si, edge['Vj'], edge['Ek'], ops[edge['selected']]) )
                        else:
                            inverse.append({'Si': Si, 'edge': edge})
                            alpha_collection.append(arch_para[edge['weight_order']])
                    
                    num_edges_to_select = self.max_nb_edges - num_selected_edges
                    if num_edges_to_select > 0:
                        stack_alpha_collection = torch.stack(alpha_collection).detach()
                        # mat = torch.softmax(stack_alpha_collection, dim = -1)
                        mat = stack_alpha_collection
                        # self.console.log(mat)
                        importance = torch.sum(mat[:, 1:], dim = -1)
                        post_select_edges = torch.topk(importance, k = num_edges_to_select).indices
                        for j in post_select_edges:
                            edge = inverse[j]['edge']
                            selected_op = torch.argmax(mat[j][1:]) + 1
                            result[type].append( (Si, edge['Vj'], edge['Ek'], ops[selected_op]) )

            genotypes.append( Genotype(V = result['V'], E = result['E']) )
        return genotypes

               
