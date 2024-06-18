import os
import argparse
from utils import yaml_config_hook
from methods import *
from sklearn.preprocessing import minmax_scale
from dataset import Dataset
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    config = yaml_config_hook("config_pau.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    args = parser.parse_args()
    if not os.path.exists(args.model_path):
        os.makedirs(args.model_path)

    # random
    np.random.seed(args.seed)

    # prepare data
    root = args.dataset_root
    if args.dataset == "InP":
        im_, gt_ = 'Indian_pines_corrected', 'Indian_pines_gt'
    elif args.dataset == "Sal":
        im_, gt_ = 'Salinas_corrected', 'Salinas_gt'
    elif args.dataset == "PaU":
        im_, gt_ = 'PaviaU', 'PaviaU_gt'
    elif args.dataset == "Hou":
        im_, gt_ = 'Houston2013', 'Houston2013_gt'
    else:
        raise NotImplementedError


    img_path = root + im_ + '.mat'
    gt_path = root + gt_ + '.mat'

    print(f"Processing: {img_path}")
    dataset = Dataset(img_path, gt_path, args.num_superpixel, args.n_neighbors, args.is_pca, args.in_channel, is_labeled=False, is_superpixel=True)
    x = dataset.x
    """
    dataset.p.show_class_map_color(dataset.gt, save = args.dataset+"_gt.pdf") 
    img = dataset.img[:,:, [39, 29, 19]]
    n1, n2, n3 = img.shape
    img =  minmax_scale(img.reshape((-1, n3))).reshape((n1,n2,n3))
    dataset.p.show_img( img / img.max(), save = args.dataset+"_img.pdf") 

    label_idx = np.unique(dataset.gt)
    dataset.p.show_color_label(args.label.split(" "), label_idx, args.dataset+"_label.pdf")
    """
    
    #####    Dual SC
    time_start = time.perf_counter()
    C, R =  dualsc_with_zero(x, alpha = 0.0000001, beta = 0.0001, rho=1, max_iter=100)
    y_pred = cal_sp_sc(C, dataset.association_mat, 0.6, dataset.n_classes)
    oa, kappa, nmi, ca = get_acc(y_pred, dataset.gt)
    run_time = round(time.perf_counter() - time_start, 3)
    print(f"dualsc acc: {round(oa, 4), round(kappa, 4), round(nmi, 4)}, time: {run_time}")
    print(ca)
    #oa, kappa, nmi, ca, class_map_full = get_acc_map(y_pred, dataset.gt)
    #dataset.p.show_class_map_color(class_map_full.reshape(dataset.gt.shape), save = args.dataset+"_dualsc.pdf")
