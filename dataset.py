import numpy as np
import torch
from scipy.io import loadmat
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale, normalize, minmax_scale
from torch.utils.data import Dataset

from utils.superpixel_utils import HSI_to_superpixels, create_association_mat, create_spixel_graph, show_superpixel,show_superpixel_gt
from Toolbox.Preprocessing import Processor
from sklearn.neighbors import kneighbors_graph

from scipy import sparse
from sklearn.datasets import load_iris
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import normalize, scale


class Dataset(Dataset):
    def __init__(self, path_to_data, path_to_gt,  num_superpixel=200, n_neighbors=10, is_pca=True, in_channel=4, is_labeled=True, is_superpixel=True):
        self.p = Processor()
        self.img, self.gt = self.p.prepare_data(path_to_data, path_to_gt)
        m, n, b = self.img.shape
        self.x = scale(self.img.reshape(-1, b))
        self.x_ori = self.x
        self.y = self.gt.reshape(-1)
        pixel_num = 0
        for i in np.unique(self.y):
            class_pixel_num = np.nonzero(self.y == i)[0].shape[0]
            pixel_num += int(class_pixel_num)
            print("pixel_class_num:", class_pixel_num)
        print("pixel num:", pixel_num)
        if not is_labeled:
            self.n_classes = np.unique(self.y).shape[0] - 1
        else:
            self.n_classes = np.unique(self.y).shape[0]

        self.n_samples, self.n_bands = self.x.shape
        print(f"data shape: {self.x.shape}, class number: {self.n_classes}")
        x_pca = self.img
        if is_pca:
            pca = PCA(n_components=in_channel)
            x_pca = pca.fit_transform(self.x).reshape(m, n, -1)
            self.x_pca, _= self.p.get_HSI_patches_rw(x_pca, self.gt, (10, 10), is_indix=False, is_labeled=is_labeled)
            self.x_pca = scale(self.x_pca.reshape((m*n, -1)))
        if is_superpixel: 
            self.sp_labels = HSI_to_superpixels(x_pca, num_superpixel=num_superpixel, is_pca=False,
                                                    is_show_superpixel=False)
            show_superpixel(self.sp_labels, x_pca[:,:,:3])
            #show_superpixel(self.sp_labels, self.img[:,:,:3])
            self.association_mat = create_association_mat(self.sp_labels)
            x_sp = np.dot(self.association_mat.T, self.x)
            x_sum = self.association_mat.T.sum(1).reshape((-1, 1))
            x_sp = x_sp / x_sum
            self.x = x_sp

