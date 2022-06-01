<h1 align="center">
Automatic Relation-aware Graph Network Proliferation 
</h1>

<div align="center">

[![](https://img.shields.io/badge/paper-pink?style=plastic&logo=GitBook)](https://arxiv.org/abs/2205.15678)
[![](https://img.shields.io/badge/oral-video-red?style=plastic&logo=airplayvideo)](https://0633e92166c0a27ea1aa-ab47878a9e45eb9e2f15be38a59f867e.ssl.cf1.rackcdn.com/PJNEQWFQ-2100498-1663000-Upload-1652882468.mp4) 
[![](https://img.shields.io/badge/poster-horizontal-informational?style=plastic&logo=Microsoft%20PowerPoint)](https://www.conferenceharvester.com/uploads/harvester/presentations/PJNEQWFQ/PJNEQWFQ-PDF-2100498-1663000-1-PDF(1).pdf)
[![](https://img.shields.io/badge/poster-vertical-informational?style=plastic&logo=Microsoft%20PowerPoint)](https://0633e92166c0a27ea1aa-ab47878a9e45eb9e2f15be38a59f867e.ssl.cf1.rackcdn.com/PJNEQWFQ-2100498-1663000-Upload-1652884373.pdf)
</div>


<div align="center">

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/automatic-relation-aware-graph-network/link-prediction-on-tsp-hcp-benchmark-set)](https://paperswithcode.com/sota/link-prediction-on-tsp-hcp-benchmark-set?p=automatic-relation-aware-graph-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/automatic-relation-aware-graph-network/graph-classification-on-cifar10-100k)](https://paperswithcode.com/sota/graph-classification-on-cifar10-100k?p=automatic-relation-aware-graph-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/automatic-relation-aware-graph-network/graph-regression-on-zinc-100k)](https://paperswithcode.com/sota/graph-regression-on-zinc-100k?p=automatic-relation-aware-graph-network)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/automatic-relation-aware-graph-network/node-classification-on-cluster)](https://paperswithcode.com/sota/node-classification-on-cluster?p=automatic-relation-aware-graph-network)
</div>
      
## Contributions 

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

## Citation

```bib
@misc{cai2022automatic,
      title={Automatic Relation-aware Graph Network Proliferation}, 
      author={Shaofei Cai and Liang Li and Xinzhe Han and Jiebo Luo and Zheng-Jun Zha and Qingming Huang},
      year={2022},
      eprint={2205.15678},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
