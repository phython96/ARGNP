import os
import dgl
import time
import torch
import torch.nn as nn
import pickle 
from torch.utils.data import Dataset
import numpy as np
import h5py
import glob
from scipy.spatial.distance import cdist
from sklearn.model_selection import StratifiedShuffleSplit
from scipy import sparse as sp
from hydra.utils import get_original_cwd, to_absolute_path


def _pairwise_distance(x):
    """
    Compute pairwise distance of a point cloud.
    Args:
        x: tensor (batch_size, num_points, num_dims)
    Returns:
        pairwise distance: (batch_size, num_points, num_points)
    """
    x_inner = -2*torch.matmul(x, x.transpose(2, 1))
    x_square = torch.sum(torch.mul(x, x), dim=-1, keepdim=True)
    return x_square + x_inner + x_square.transpose(2, 1)


def _knn_matrix(x, k=16, self_loop=True):
    """Get KNN based on the pairwise distance.
    Args:
        x: (batch_size, num_dims, num_points, 1)
        k: int
    Returns:
        nearest neighbors: (batch_size, num_points ,k) (batch_size, num_points, k)
    """
    x = x.transpose(2, 1).squeeze(-1)
    batch_size, n_points, n_dims = x.shape
    if self_loop:
        _, nn_idx = torch.topk(-_pairwise_distance(x.detach()), k=k)
    else:
        _, nn_idx = torch.topk(-_pairwise_distance(x.detach()), k=k+1)
        nn_idx = nn_idx[:, :, 1:]
    center_idx = torch.arange(0, n_points, device=x.device).repeat(batch_size, k, 1).transpose(2, 1)
    return torch.stack((nn_idx, center_idx), dim=0)


class Dilated2d(nn.Module):
    """
    Find dilated neighbor from neighbor list

    edge_index: (2, batch_size, num_points, k)
    """
    def __init__(self, k=9, dilation=1, stochastic=False, epsilon=0.0):
        super(Dilated2d, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k

    def forward(self, edge_index):
        if self.stochastic:
            raise NotImplementedError('stochastic currently is not supported')
            # if torch.rand(1) < self.epsilon and self.training:
            #     num = self.k * self.dilation
            #     randnum = torch.randperm(num)[:self.k]
            #     edge_index = edge_index[:, :, :, randnum]
            # else:
            #     edge_index = edge_index[:, :, :, ::self.dilation]
        else:
            edge_index = edge_index[:, :, :, ::self.dilation]
        return edge_index


class DilatedKnn2d(nn.Module):
    """
    Find the neighbors' indices based on dilated knn
    """
    def __init__(self, k=9, dilation=1, self_loop=False, stochastic=False, epsilon=0.0):
        super(DilatedKnn2d, self).__init__()
        self.dilation = dilation
        self.stochastic = stochastic
        self.epsilon = epsilon
        self.k = k
        self.self_loop = self_loop
        self._dilated = Dilated2d(k, dilation, stochastic, epsilon)
        self.knn = _knn_matrix

    def forward(self, x):
        edge_index = self.knn(x, self.k * self.dilation, self.self_loop)
        return self._dilated(edge_index)


def remove_self_loops(edge_index):
    if edge_index[0, 0, 0, 0] == 0:
        edge_index = edge_index[:, :, :, 1:]
    return edge_index


def add_self_loops(edge_index):
    if edge_index[0, 0, 0, 0] != 0:
        self_loops = torch.arange(0, edge_index.shape[2]).repeat(2, edge_index.shape[1], 1).unsqueeze(-1)
        edge_index = torch.cat((self_loops.to(edge_index.device), edge_index[:, :, :, 1:]), dim=-1)
    return edge_index


def translate_pointcloud(pointcloud):
    xyz1 = torch.Tensor(3)
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def normalize_pointcloud(pointcloud):
    pmin = np.min(pointcloud, axis = 0)
    pmax = np.max(pointcloud, axis = 0)
    pointcloud -= (pmin + pmax) / 2
    scale = np.max(np.linalg.norm(pointcloud, axis = 1))
    pointcloud *= 1.0 / scale
    return pointcloud
 

def load_data(partition, name = 'modelnet10'):
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(get_original_cwd(), "data/modelnet/data", name, '*%s*.h5'%partition)):
        with h5py.File(os.path.join(get_original_cwd(), h5_name), 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


class Modelnet(Dataset):

    def __init__(self, name, num_points, partition='train'):
        self.data, self.label = load_data(partition, name)
        self.num_points = num_points
        self.partition = partition        


    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        # pointcloud = normalize_pointcloud(pointcloud)
        return pointcloud, label


    def __len__(self):
        return self.data.shape[0]


class ModelnetDGL(torch.utils.data.Dataset):

    def __init__(self, name, num_points, split, nb_sample = None, k = 9, d = 1):
        super().__init__()
        self.split = split
        self.num_points = num_points
        self.nb_sample = nb_sample
        self.data = Modelnet(name, num_points, split)
        self.dilated_knn_graph = DilatedKnn2d(k=k, dilation=d)


    def __getitem__(self, idx):
        node, label = self.data[idx]
        node = torch.tensor(node).cuda()
        _node = node.transpose(0, 1).unsqueeze(0).unsqueeze(3)
        edge_index = self.dilated_knn_graph(_node)
        _, nb_batch, nb_node, nb_edge = edge_index.shape
        edge_index = edge_index.view(2, -1)
        
        g = dgl.graph(data = (edge_index[0], edge_index[1]))
        g.ndata['feat'] = node
        X, Y = node.index_select(0, edge_index[0]), node.index_select(0, edge_index[1])
        # g.edata['feat'] = torch.cat([X-Y, X, Y], dim = 1)
        g.edata['feat'] = torch.cat([X-Y, X, Y], dim = 1) * 0.
        # import ipdb; ipdb.set_trace()
        # g.edata['feat'] = torch.ones(nb_node * nb_edge, 1)
        return g, label


    def __len__(self):
        return len(self.data)



class ModelnetDataset(torch.utils.data.Dataset):

    def __init__(self, name, num_points = 1024, k = 9, d = 1):

        start = time.time()
        print("[I] Loading dataset %s..." % (name))
        self.name = name

        self.train = ModelnetDGL(name=name, num_points=num_points, split='train', k=k, d=d)
        self.test  = ModelnetDGL(name=name, num_points=num_points, split='test', k=k, d=d)
        # self.test  = self.train
        self.val   = None
        print('train, test :',len(self.train),len(self.test))
        print("[I] Finished loading.")
        print("[I] Data load time: {:.4f}s".format(time.time()-start))


    def collate(self, samples):
        graphs, labels = map(list, zip(*samples))
        labels = torch.tensor(labels)
        for idx, graph in enumerate(graphs):
            graphs[idx].ndata['feat'] = graph.ndata['feat'].float()
            graphs[idx].edata['feat'] = graph.edata['feat'].float()
        batched_graph = dgl.batch(graphs)
        return batched_graph, labels.squeeze(1)




if __name__ == '__main__':
    # ds = ModelnetDataset('modelnet40')
    # import ipdb; ipdb.set_trace()
    # print(ds.train[0])
    dataset = ModelnetDataset("modelnet40")
    import ipdb; ipdb.set_trace()