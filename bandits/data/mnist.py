import torch, torchvision
import torchvision.datasets as datasets
import numpy as np
import random


def get_mnist_data():
    return datasets.MNIST(root='./data', train=True,
                          download=True, transform=None)

def construct_dataset_from_features(data,
                                    r_correct=1,
                                    r_incorrect=0,
                                    shuffle_rows=True):
    """
    Returns:
      features: matrix with feature vectors as rows
      rewards: matrix with rows (r_0, r_1, ..., r_9)
      opt_vals: vector of expected optimal (reward, action) for each context
    """
    num_contexts = len(data)
    if shuffle_rows:
        random.shuffle(data)

    features, labels = map(np.array, zip(*data))

    # normalize
    sstd = safe_std(np.std(features, axis=0, keepdims=True)[0, :])
    features= ((features - np.mean(features, axis=0, keepdims=True)) / sstd)

    rewards = np.zeros((num_contexts, 10))
    rewards[np.arange(num_contexts), labels] = 1.0

    return features, rewards, (np.ones(num_contexts), labels)

def safe_std(values):
  """Remove zero std values for ones."""
  return np.array([val if val != 0.0 else 1.0 for val in values])

def get_raw_features():
    """
    Returns:
      features
      labels
    """
    res = []
    mnist = get_mnist_data()
    for im, label in mnist:
        tensor = torchvision.transforms.ToTensor()(im)
        feature_vec = tensor.numpy().flatten()
        res.append((feature_vec, label))
    return res


def get_vae_features():
    """
    TODO
    """
    pass
