import torch, torchvision
import torchvision.datasets as datasets
import numpy as np


def get_mnist_data():
    return datasets.MNIST(root='./data', train=True,
                          download=True, transform=None)

def construct_dataset_from_features(features):
    """
    Returns:
      dataset: matrix with n rows: (context, label)
      opt_vals: vector of expected optimal (reward, action) for each context
    """
    pass

def get_raw_features():
    """
    Returns list of (context, label) pairs
    """
    res = []
    mnist = get_mnist_data()
    for im, label in mnist:
        tensor = torchvision.transforms.ToTensor()(im)
        context_vec = tensor.numpy().flatten()
        res.append((context_vec, label))
    return res


def get_vae_features():
    """
    TODO
    """
    pass
