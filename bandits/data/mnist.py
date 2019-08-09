import torch, torchvision
import torchvision.datasets as datasets
import numpy as np
import cv2
import sklearn
from sklearn.decomposition import PCA
def get_mnist_data():
    return datasets.MNIST(root='./data', train=True,
                          download=True, transform=None)

def construct_dataset_from_features(features):
    """
    Returns:
      dataset: matrix with n rows: (context, label)
      opt_vals: vector of expected optimal (reward, action) for each context
    """

def get_raw_features():
    res = []
    mnist = get_mnist_data()
    for im, label in mnist:
        tensor = torchvision.transforms.ToTensor()(im)
        context_vec = tensor.numpy().flatten()
        res.append((context_vec, label))
    return res

def get_PCA_features():
    """
    TODO
    :return:
    """
    res = get_raw_features()
    dataset = [feats for feats,label in res]
    scaler = sklearn.preprocessing.StandardScaler()
    scaler.fit(dataset)
    dataset = scaler.transform(dataset)
    # .40 yields 23 components
    pca = PCA(n_components=20)
    pca.fit(dataset)
    dataset_pca = pca.transform(dataset)
    print(pca.n_components_)
    return dataset_pca

def get_bovw_features():
    """
    TODO
    :return:
    """
    def get_SIFT_feature(pic):
        sift = cv2.xfeatures2d.SIFT_create()
        pic = (pic-np.min(pic))*255/np.max(pic)
        pic = pic.astype(np.uint8)
        kp, desc = sift.detectAndCompute(pic,None)
        return kp,desc
    """
    dataset= []
    mnist = get_mnist_data()
    for im, label in mnist:
        tensor = torchvision.transforms.ToTensor()(im)
        imgarray = tensor.numpy()
        imgarray = imgarray[0]
        img_kp, img_desc = get_SIFT_feature(imgarray)
        print(img_kp[0])
        print(img_desc)
        dataset.append(img_kp,label)
    """
def get_vae_features():
    """
    TODO
    """
    pass

get_PCA_features()