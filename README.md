# Self-Pruning Neural Network

## Overview
This project implements a self-pruning neural network where each weight has a learnable gate that determines whether it should be active or pruned during training.

## Features
- Custom PrunableLinear layer
- Sparsity regularization using L1 loss
- Training on CIFAR-10 dataset
- Analysis of sparsity vs accuracy trade-off

## Results

| Lambda | Accuracy (%) | Sparsity (%) |
|--------|-------------|--------------|
| 0.001  | 29.52       | 3.95         |
| 0.01   | 30.73       | 21.51        |
| 0.1    | 31.10       | 49.01        |

## How to Run

```bash
pip install torch torchvision matplotlib
python train.py