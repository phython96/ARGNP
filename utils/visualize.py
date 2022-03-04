import os
import sys
import torch
import numpy as np
# import pygraphviz as pgv
import networkx as nx
from pathlib import Path

# def add_edge(g, src, dst, label = '', color = 'red', penwidth = 1):
#     g.add_edge(
#         src, dst, 
#         color     = color,
#         style     = 'solid',
#         label     = label,
#         fontname  = 'Times-Roman',
#         fontsize  = 14,
#         fontcolor = "#7F01FF",
#         arrowsize = 1,
#         arrowhead = 'normal',
#         arrowtail = 'normal',
#         dir       = 'forward',
#         penwidth  = penwidth,
#     )

# def add_node(g, node, label = None, color = 'black', shape = 'rect', penwidth = 1):
#     g.add_node(
#         n         = node,
#         label     = label,
#         color     = color,
#         fontname  = 'Times-Roman',
#         fontsize  = 14,
#         shape     = shape,
#         style     = 'solid',
#         fontcolor = 'black',
#         fixedsize = False,
#         width     = 1,
#         penwidth  = penwidth,
#         #height    = 1
#     )
    

# def plot_genotype(g, genotype, i_layer):
#     V_color         = 'dodgerblue'
#     E_color         = 'crimson'
#     V_link_color    = 'dodgerblue'
#     E_link_color    = 'crimson'
#     V_op_link_color = 'dodgerblue'
#     E_op_link_color = 'crimson'
#     V_op_color      = 'dodgerblue'
#     E_op_color      = 'crimson'

#     add_node(g, f'V0_{i_layer}', label = 'V0', color = V_color, penwidth = 2)
#     add_node(g, f'Vo_{i_layer}', label = 'Vo', color = V_color, penwidth = 2)

#     add_node(g, f'E0_{i_layer}', label = 'E0', color = E_color, penwidth = 2)
#     add_node(g, f'Eo_{i_layer}', label = 'Eo', color = E_color, penwidth = 2)

#     nums = 0
#     for Vi, Vj, Ek, Op in genotype.V:
#         if Vi > nums:
#             nums = Vi 
#         op_node = f'V{Vi}_V{Vj}_E{Ek}_O{Op}_{i_layer}'
#         add_node(g, op_node, label = Op, shape = 'hexagon', color = V_color)

#     for Ei, Vj, Ek, Op in genotype.E:
#         op_node = f'E{Ei}_V{Vj}_E{Ek}_O{Op}_{i_layer}'
#         add_node(g, op_node, label = Op, shape = 'hexagon', color = E_color)

#     for Si in range(1, nums + 1):
#         add_node(g, f'V{Si}_{i_layer}', label = f'V{Si}', color = V_color)
#         add_node(g, f'E{Si}_{i_layer}', label = f'E{Si}', color = E_color)
#         add_edge(g, f'V{Si}_{i_layer}', f'Vo_{i_layer}', color = V_link_color)
#         add_edge(g, f'E{Si}_{i_layer}', f'Eo_{i_layer}', color = E_link_color)

#     for Vi, Vj, Ek, Op in genotype.V:
#         op_node = f'V{Vi}_V{Vj}_E{Ek}_O{Op}_{i_layer}'
#         add_edge(g, f'V{Vj}_{i_layer}', op_node, color = V_link_color)
#         add_edge(g, f'E{Ek}_{i_layer}', op_node, color = E_link_color)
#         add_edge(g, op_node, f'V{Vi}_{i_layer}', color = V_op_link_color)


#     for Ei, Vj, Ek, Op in genotype.E: 
#         op_node = f'E{Ei}_V{Vj}_E{Ek}_O{Op}_{i_layer}'
#         add_edge(g, f'V{Vj}_{i_layer}', op_node, color = V_link_color)
#         add_edge(g, f'E{Ek}_{i_layer}', op_node, color = E_link_color)
#         add_edge(g, op_node, f'E{Ei}_{i_layer}', color = E_op_link_color)


# def plot_genotypes(args, i_epoch, genotypes):
#     g = pgv.AGraph(
#         directed    = True,
#         strict      = True,
#         nodesep     = 0.5,
#         rankdir     = 'TB',
#         splines     = 'spline',
#         concentrate = True,
#         bgcolor     = 'white',
#         compound    = True,
#         normalize   = False,
#         encoding    = 'UTF-8'
#     )

#     for i_layer, genotype in enumerate(genotypes):
#         plot_genotype(g, genotype, i_layer)

#     dir_path     = f'{args.ds.arch_save}/{args.ds.data}'
#     pdf_dir_path = f'{dir_path}/imgs'
#     txt_dir_path = f'{dir_path}/txts'
#     if not os.path.isdir(dir_path):
#         os.mkdir(dir_path)
#     if not os.path.exists(pdf_dir_path):
#         os.mkdir(pdf_dir_path)
#     if not os.path.exists(txt_dir_path):
#         os.mkdir(txt_dir_path)

#     g.layout('dot')
#     g.draw(f'{pdf_dir_path}/{i_epoch}.png')
#     with open(f'{txt_dir_path}/{i_epoch}.txt', 'w') as f:
#         f.write(str(genotypes))
#     f.close()


# def weight(e):
#     # 模长
#     e = torch.norm(e, dim = 1)
#     e = (e - torch.min(e)) / (torch.max(e) - torch.min(e)+ 1e-6)
#     return e

def weight(e):
    pca = PCA(1)
    pca.fit(e)
    e = pca.transform(e).squeeze(1)
    e = (e - torch.min(e)) / (torch.max(e) - torch.min(e) + 1e-6)
    return e


import matplotlib.pyplot as plt
def plot_graphs(args, G, Vs, Es, epoch_id, example_id, V = None, L = None):
    
    nb_cell = len(Es)
    nb_node = len(Es[0])
    plt.figure(figsize=(nb_node*4, 2*4), dpi = 200)
    
    nG = G.to_networkx()

    node_size  = 1
    edge_size  = 1
    node_color = "grey"
    node_cmap  = None
    edge_cmap  = None
    pos = nx.spring_layout(nG)


    if args.ds.data in [ 'modelnet40', 'modelnet10' ]: 
        pos = {}
        # A = torch.rand(3, 2)
        A = torch.Tensor([[1, -2], [-2, 3], [1, 3]])
        V = V.mm(A)
        for i in range(V.shape[0]):
            pos[i] = V[i].numpy()
        node_size = 1
        edge_size = 1
        node_color = None
        node_cmap = plt.cm.gist_rainbow
        edge_cmap = plt.cm.hsv

    elif args.ds.data in ['MNIST']:
        pos = {}
        A = torch.Tensor([[0, -1], [1, 0]])
        P = V[:, 1:].mm(A)
        for i in range(V.shape[0]):
            pos[i] = P[i].numpy()
        V = V[:, 0]
        V = (V - torch.min(V)) / (torch.max(V) - torch.min(V)) * 100 + 20
        node_color = list(V)
        node_size = node_color
        node_cmap = plt.cm.YlGn
        edge_cmap = plt.cm.hsv
    
    elif args.ds.data in ['ZINC' ]:
        # node_color = V
        node_size = 125
        edge_size = 5
        node_cmap = plt.cm.brg
        edge_cmap = plt.cm.coolwarm

    elif args.ds.data in ['SBM_CLUSTER']:
        node_color = L
        node_size = 20
        edge_size = 1
        node_cmap = plt.cm.gist_rainbow
        edge_cmap = plt.cm.binary
        
        R = 20
        pos = {}
        for i in range(L.shape[0]):
            theta = L[i].numpy() / 6 * 2 * 3.1415926
            x = R * np.cos(theta)
            y = R * np.sin(theta)
            pos[i] = np.random.normal([x, y], 8)



    for j in range(nb_node):
        
        # ax = plt.subplot(1, nb_node, j + 1)
        ax = plt.subplot(2, nb_node, j + 1)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        #! old start
        E = Es[0][j]
        E = weight(E)
        edge_width = []
        wmp = {}
        for u, v, data in nG.edges(data = True):
            if (u, v) not in wmp:
                wmp[(u, v)] = E[data['id']].item()
            else: 
                wmp[(u, v)] += E[data['id']].item()

        mi = 1000.
        mx = -1000.
        for u, v, data in nG.edges(data = True):
            ew  = wmp[(u, v)] / 2
            data['weight'] = ew 
            edge_width.append(ew)
            mx = max(mx, ew)
            mi = min(mi, ew)
        #! old end


        #! start
        # E = Es[i][j]
        # edge_width = []
        # pre = None
        # save = []
        # edge_size = []
        # for u, v, data in  nG.edges(data = True):

        #     if L[u] == L[v]:
        #         edge_size.append(4)
        #     else:
        #         edge_size.append(1)

        #     if pre is not None and u != pre: 
        #         # deal
        #         save = torch.stack(save)
        #         save = weight(save)
        #         save = list(save)
        #         save = [x.item() for x in save]
        #         edge_width += save
        #         save = []
        #     pre = u
        #     save.append(E[data['id']])

        # save = torch.stack(save)
        # save = weight(save)
        # save = list(save)
        # save = [x.item() for x in save]
        # edge_width += save

        # mi = 0.
        # mx = 1.
        #! end
        node_color = None
        nx.draw_networkx_nodes(nG, pos, node_size = node_size, node_color = node_color, cmap = node_cmap)
        nx.draw_networkx_edges(nG, pos, width = edge_size, edge_color = edge_width, 
                               edge_vmin = mi, edge_vmax = mx, edge_cmap = edge_cmap, arrows = False)

        # import ipdb; ipdb.set_trace()
        #! start draw node features
        ax = plt.subplot(2, nb_node, nb_node + j + 1)
        
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        
        V = Vs[0][j]
        node_color = weight(V)
        nx.draw_networkx_nodes(nG, pos, node_size = node_size*20, node_color = node_color, cmap = node_cmap)
        #! end draw edge features
        

    prefix = os.path.splitext(args.ds.load_genotypes)[0]
    plt.subplots_adjust(left=0.00, bottom=0.00, right=1.0, top=1.0,wspace=0.00, hspace=0.00)

    path = prefix + "_graphs"
    try:
        os.mkdir(path)
    except:
        pass
    path = os.path.join(path, str(epoch_id))
    try:
        os.mkdir(path)
    except:
        pass
    
    Path(path).mkdir(parents = True, exist_ok = True)
    plt.savefig(os.path.join(path, f'{example_id}.pdf'))



class PCA(object):
    def __init__(self, n_components=2):
        self.n_components = n_components

    def fit(self, X):
        n = X.shape[0]
        self.mean = torch.mean(X, axis=0)
        X = X - self.mean
        covariance_matrix = 1 / n * torch.matmul(X.T, X)
        eigenvalues, eigenvectors = torch.eig(covariance_matrix, eigenvectors=True)
        eigenvalues = torch.norm(eigenvalues, dim=1)
        idx = torch.argsort(-eigenvalues)
        eigenvectors = eigenvectors[:, idx]
        self.proj_mat = eigenvectors[:, 0:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return X.matmul(self.proj_mat)
