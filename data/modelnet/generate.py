import os
import sys
import glob
import time
import torch
import h5py
import pickle
import dgl
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
from torch_edge import DilatedKnn2d

def download(data_dir):
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    # DATA_DIR = os.path.join(BASE_DIR, 'data')
    DATA_DIR = data_dir
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    print(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048'))
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition, data_dir='data'):
    download(data_dir)
    # BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = data_dir
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5'%partition)):
        with h5py.File(h5_name, 'r') as f:
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    return all_data, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', data_dir='data'):
        self.data, self.label = load_data(partition, data_dir)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        if self.partition == 'train':
            pointcloud = translate_pointcloud(pointcloud)
            np.random.shuffle(pointcloud)
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]


class ModelnetDGL(torch.utils.data.Dataset):

    def __init__(self, num_points, split, nb_sample = None, data_dir = 'data'):
        super().__init__()
        self.split = split
        self.num_points = num_points
        self.nb_sample = nb_sample
        self.data = ModelNet40(num_points, split, data_dir)
        self._prepare()


    def __getitem__(self, idx):
        return self.graph_lists[idx], self.graph_labels[idx]


    def __len__(self):
        return len(self.graph_labels)


    def _prepare(self):
        dataset = self.data
        self.graph_lists = []
        self.graph_labels = []
        knn = DilatedKnn2d()
        example = dataset[0][0]
        example = torch.tensor(example)
        example = example.transpose(0, 1).unsqueeze(0).unsqueeze(3)
        example = knn(example)
        print(example.shape)
        _, nb_batch, nb_node, nb_edge = example.shape

        if self.nb_sample is None: 
            self.nb_sample = len(dataset)
        for i in tqdm(range(self.nb_sample)): 
            x, l = dataset[i]
            x = torch.tensor(x)
            l = torch.tensor(l)
            e = x.transpose(0, 1).unsqueeze(0).unsqueeze(3)
            e = knn(e)
            g = dgl.DGLGraph()
            g.add_nodes(nb_node)
            for j in range(nb_node):
                for k in range(nb_edge):
                    te = e[:, 0, j, k]
                    s = te[0].item()
                    t = te[1].item()
                    g.add_edge(s, t)
            g.ndata['feat'] = x
            g.edata['feat'] = torch.ones(nb_node * nb_edge, 1)
            self.graph_lists.append(g)
            self.graph_labels.append(l)
        


if __name__ == '__main__':
    # start = time.time()
    # with open('data/molecules/modelnet40.pkl','wb') as f:
    #     pickle.dump([dataset.train,dataset.val,dataset.test,num_atom_type,num_bond_type],f)
    # print('Time (sec):',time.time() - start)

    train = ModelnetDGL(1024, 'train')
    val = None
    test = ModelnetDGL(1024, 'test')

    start = time.time()
    with open('modelnet40_1024.pkl','wb') as f:
        pickle.dump([train, val, test, 3, 1],f)
    print('Time (sec):',time.time() - start)

    # import ipdb; ipdb.set_trace()