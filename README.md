# Multi-View Graph Convolutional Networks with Attention Mechanism (PyTorch)

**Implementation of**:  
> Kaixuan Yao, Jiye Liang, Jianqing Liang, Ming Li, Feilong Cao.  
> *Multi-view graph convolutional networks with attention mechanism*.  
> Artificial Intelligence, Volume 307, 2022, 103708.  
> [https://doi.org/10.1016/j.artint.2022.103708](https://doi.org/10.1016/j.artint.2022.103708)

---

## Overview

This repository provides a **PyTorch implementation** of the Multi-View Graph Convolutional Network with Attention (MV-GCN-Attention) as proposed by Yao et al. (2022).  

The method integrates **multi-view learning** and **graph neural networks** by:
- Representing data from multiple graph views
- Aggregating features via **graph convolutional networks**
- Applying an **attention mechanism** to learn optimal view weights
- Combining them for node classification on citation network datasets

This implementation is based on **[PyTorch Geometric](https://pytorch-geometric.readthedocs.io)** and runs directly through the provided Jupyter Notebook.

---

## Features

- **Multi-view graph construction** from dataset features
- **Cosine similarity**-based adjacency generation
- **Sparse graph representation** for efficiency
- **Attention-based view aggregation** to improve representation learning
- **Node classification** on benchmark citation datasets

---

## Supported Datasets

The following citation network datasets are supported and tested:

- **Cora**
- **Citeseer**
- **Pubmed**

Datasets are automatically downloaded via `torch_geometric.datasets.Planetoid`.

---

## Tech Stack

- [PyTorch](https://pytorch.org)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io)
- NumPy
- scikit-learn
- NetworkX

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/itsatefe/Multi-View-GCN-Attention.git
cd Multi-View-GCN-Attention
pip install torch torchvision torchaudio
pip install torch_geometric
pip install numpy scikit-learn networkx
````

---

## ▶Usage

Run the provided Jupyter Notebook:

```bash
jupyter notebook multi-view-gcn-attention.ipynb
```

The notebook contains:

1. Dataset loading & preprocessing
2. Multi-view adjacency matrix construction
3. MV-GCN-Attention model definition
4. Training and evaluation scripts

---

## Methodology

The **MV-GCN-Attention** architecture follows these steps:

1. **Multi-view graph construction**

   * Each view captures a different similarity perspective of the dataset (e.g., feature similarity via cosine distance).
2. **Graph convolution per view**

   * GCN layers extract local neighborhood representations for each view.
3. **Attention-based view weighting**

   * Attention mechanism learns to emphasize more informative views.
4. **Feature aggregation and classification**

   * Weighted sum of all views is fed into a classifier for node classification.

---

## Citation

If you use this code, please cite:

```bibtex
@article{yao2022multi,
  title={Multi-view graph convolutional networks with attention mechanism},
  author={Yao, Kaixuan and Liang, Jiye and Liang, Jianqing and Li, Ming and Cao, Feilong},
  journal={Artificial Intelligence},
  volume={307},
  pages={103708},
  year={2022},
  publisher={Elsevier}
}
```

