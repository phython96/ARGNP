<h1 align="center">
Automatic Relation-aware Graph Network Proliferation 
</h1>

<div align="center">

[![](https://img.shields.io/badge/paper-pink?style=plastic&logo=GitBook)](https://readpaper.com/paper/651316002870833152)
[![](https://img.shields.io/badge/-github-grey?style=plastic&logo=github)](https://github.com/phython96/ARGNP) 
[![](https://img.shields.io/badge/video-red?style=plastic&logo=airplayvideo)]() 
[![](https://img.shields.io/badge/project-informational?style=plastic&logo=producthunt)]()
</div>


## What's new? 🔥

1. **We devise a RELATION-AWARE GRAPH SEARCH SPACE that comprises both node and relation learning operations.**
So, the ARGNP can leverage the edge attributes in some datasets, such as ZINC. 
It significantly improves the graph representative capability. 
Interestingly, we also observe the performance improvement even if there is no available edge attributes in some datasets. 

2. **We design a NETWORK PROLIFERATION SEARCH PARADIGM to progressively determine the GNN architectures by iteratively performing network division and differentiation.**
The network proliferation search paradigm decomposes the training of global supernet into sequential local supernets optimization, which alleviates the interference among child graph neural architectures. It reduces the spatial-time complexity from quadratic to linear and enables the search to thoroughly free from the cell-sharing trick. 

3. **The framework is suitable for solving node-level, edge-level, and graph-level tasks.**


## Relation-aware Graph Search Space
<img src="assets/space.png" width="800" />

## Network Proliferation Search Paradigm

<!-- ![](assets/proliferation.gif) -->
<img src="assets/proliferation.gif" width="800" />


## Environment
Try the following command for installation. 
```sh
# Install Graphviz
sudo apt install graphviz
# Install Python Environment
conda env create -f environment.yml
conda activate gnas
```

## Download datasets
Some datasets (CLUSTER, TSP, ZINC, and CIFAR10) are provided by project [benchmarking-gnns](https://github.com/graphdeeplearning/benchmarking-gnns), the others (ModelNet10 and ModelNet40) are provided in [Princeton ModelNet](http://modelnet.cs.princeton.edu/). 
|DATASET|TYPE|URL|
|---|---|---|
|CLUSTER|node|[click here](https://data.dgl.ai/dataset/benchmarking-gnns/SBM_CLUSTER.pkl)|
|TSP|edge|[click here](https://data.dgl.ai/dataset/benchmarking-gnns/TSP.pkl)|
|ZINC|graph|[click here](https://data.dgl.ai/dataset/benchmarking-gnns/ZINC.pkl)|
|CIFAR10|graph|[click here](https://data.dgl.ai/dataset/benchmarking-gnns/CIFAR10.pkl)|
|ModelNet10|graph|[click here](http://modelnet.cs.princeton.edu/)|
|ModelNet40|graph|[click here](http://modelnet.cs.princeton.edu/)|



## Searching
We have provided scripts for easily searching graph neural networks on six datasets. 
```shell
python start.py gpu=0 repeats=4 data=ZINC save='archs/start'
```

## Training
We provided scripts for easily training graph neural networks searched by ARGNP.
```python
CUDA_VISIBLE_DEVICES=0 python train.py ds=ZINC  optimizer=train_optimizer ds.load_genotypes='archs2/start/repeat3/ZINC/45/cell_geno.txt'
```


## Visualization
We provided a python script for easily visualize searched architectures. 
Just run the following command. 
```python
python utils/plot_genotype.py
```

