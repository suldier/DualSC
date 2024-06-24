import heapq
import numpy as np
from utils.evaluation import cluster_accuracy as calacc, get_parameter_number
import os
#from evaluate import cluster_accuracy as acc

import fnmatch
from sklearn.cluster import KMeans
from utils.postprocess import spixel_to_pixel_labels, affinity_to_pixellabels

from munkres import Munkres
from scipy.sparse.linalg import svds
from sklearn.cluster import SpectralClustering
from sklearn.metrics import normalized_mutual_info_score, cohen_kappa_score, accuracy_score
import scipy as sp
from sklearn.preprocessing import normalize,minmax_scale
import ot
import time
import signal

import matplotlib.pyplot as plt
from sklearn.neighbors import kneighbors_graph
class FuncTimeoutException(Exception):
  pass

def handler(signum, _):
  raise FuncTimeoutException('Time exceeded!')
def func_timeout(times=0):
  def decorator(func):
    if not times:
      return func
    def wraps(*args, **kwargs):
      signal.alarm(times)
      result = func(*args, **kwargs)
      signal.alarm(0)
      return result
    return wraps
  return decorator
signal.signal(signal.SIGALRM, handler)

def find_files(root_dir, query="*.wav", include_root_dir=True):
     """Find files recursively.
     Args:
     root_dir (str): Root root_dir to find.
         query (str): Query to find.
         include_root_dir (bool): If False, root_dir name is not included.
     Returns:
         list: List of found filenames.
     """
     files = []
     for root, _, filenames in os.walk(root_dir, followlinks=True):
         for filename in fnmatch.filter(filenames, query):
             files.append(os.path.join(root, filename))
     if not include_root_dir:
         files = [file_.replace(root_dir + "/", "") for file_ in files]
 
     return files

def soft_threshold(A, lamb):
        B = np.zeros((A.shape))
        idx = np.where(A > lamb)
        B[idx] = A[idx] - lamb
        idx = np.where(A < -lamb)
        B[idx] = A[idx] + lamb
        return B

def dualsc_with_zero(X, alpha = 1, beta = 1, rho=1000, max_iter=100):
        """
        Sovle Co-subspace clustering problem using ADMM
        min_{R, C} 1/2||RX - XC||_2^F + \alpha||R||_1 + \beta ||C||_1
        s.t. diag(R) = 0, diag(C) = 0
        """
        m, n = X.shape
        R = np.zeros((m, m))
        #R = np.identity(m)
        C = np.zeros((n, n))
        #C = np.identity(n)
        Delta_A = np.zeros((m, m))
        Delta_B = np.zeros((n, n))
        A_pre = np.zeros((m, m))
        B_pre = np.zeros((n, n))
        B = np.identity(n)
        I_m = np.identity(m)
        I_n = np.identity(n)
        err1, err2 = 1e-3, 1e-3
        i = 0
        while i < max_iter:
            A = np.dot(np.dot(X, np.dot(B, X.T)) + rho * (R - np.diag(np.diag(R))) - Delta_A,
                np.linalg.inv(np.dot(X, X.T) + rho * I_m))
            J_A = soft_threshold(A + Delta_A / rho, alpha / rho)
            R = J_A - np.diag(np.diag(J_A))
            Delta_A += rho * (A - (R - np.diag(np.diag(R))))

            B = np.dot(np.linalg.inv(np.dot(X.T, X) + rho * I_n),
                np.dot(X.T, np.dot(A, X)) + rho * (C - np.diag(np.diag(C))) - Delta_B)
            J_B = soft_threshold(B + Delta_B / rho, beta / rho)
            C = J_B - np.diag(np.diag(J_B))
            Delta_B += rho * (B - (C - np.diag(np.diag(C))))

            err_ar = np.linalg.norm(A - R, np.inf)
            err_aa = np.linalg.norm(A - A_pre, np.inf)
            err_bc = np.linalg.norm(B - C, np.inf)
            err_bb = np.linalg.norm(B - B_pre, np.inf)
            A_pre = A
            B_pre = B
            i +=1
            print('iter: %d, err_aa: %f, err_bb: %f, err_ac: %f, err_bc: %f' % (i, err_aa, err_bb, err_ar, err_bc))
            if (err_ar < err1 or err_aa < err1) and (err_bc < err2 or err_bb <err2):
                break
        return R, C

def dualsc_with_zero_plot_loss_acc(X, dataset, name, scro,  alpha = 1, beta = 1, rho=1000, max_iter=100):
        """
        Sovle Co-subspace clustering problem using ADMM
        min_{R, C} 1/2||RX - XC||_2^F + \alpha||R||_1 + \beta ||C||_1
        s.t. diag(R) = 0, diag(C) = 0
        """

        loss_acc = np.zeros((max_iter+1, 5))

        m, n = X.shape
        R = np.zeros((m, m))
        #R = np.identity(m)
        C = np.zeros((n, n))
        #C = np.identity(n)
        Delta_A = np.zeros((m, m))
        Delta_B = np.zeros((n, n))
        A_pre = np.zeros((m, m))
        B_pre = np.zeros((n, n))
        B = np.identity(n)
        I_m = np.identity(m)
        I_n = np.identity(n)
        err1, err2 = 1e-3, 1e-3
        i = 0
        while i < max_iter:
            A = np.dot(np.dot(X, np.dot(B, X.T)) + rho * (R - np.diag(np.diag(R))) - Delta_A,
                np.linalg.inv(np.dot(X, X.T) + rho * I_m))
            J_A = soft_threshold(A + Delta_A / rho, alpha / rho)
            R = J_A - np.diag(np.diag(J_A))
            Delta_A += rho * (A - (R - np.diag(np.diag(R))))

            B = np.dot(np.linalg.inv(np.dot(X.T, X) + rho * I_n),
                np.dot(X.T, np.dot(A, X)) + rho * (C - np.diag(np.diag(C))) - Delta_B)
            J_B = soft_threshold(B + Delta_B / rho, beta / rho)
            C = J_B - np.diag(np.diag(J_B))
            Delta_B += rho * (B - (C - np.diag(np.diag(C))))

            err_ar = np.linalg.norm(A - R, np.inf)
            err_aa = np.linalg.norm(A - A_pre, np.inf)
            err_bc = np.linalg.norm(B - C, np.inf)
            err_bb = np.linalg.norm(B - B_pre, np.inf)
            A_pre = A
            B_pre = B
            i +=1
 
            y_pred = cal_sp_sc(R, dataset.association_mat, scro, dataset.n_classes)
            oa, kappa, nmi = get_acc(y_pred, dataset.gt)
            loss_acc[i, :] = [err_aa, err_bb, err_ar, err_bc, oa] 

            print('iter: %d, err_aa: %f, err_bb: %f, err_ac: %f, err_bc: %f' % (i, err_aa, err_bb, err_ar, err_bc))
            if (err_ar < err1 or err_aa < err1) and (err_bc < err2 or err_bb <err2):
                break
        np.save(name, loss_acc[:i+1, :])
        return R, C

def thrC(C, ro):
        if ro < 1:
            N = C.shape[1]
            Cp = np.zeros((N, N))
            S = np.abs(np.sort(-np.abs(C), axis=0))
            Ind = np.argsort(-np.abs(C), axis=0)
            for i in range(N):
                cL1 = np.sum(S[:, i]).astype(float)
                stop = False
                csum = 0
                t = 0
                while (stop == False):
                    csum = csum + S[t, i]
                    if csum > ro * cL1:
                        stop = True
                        Cp[Ind[0:t + 1, i], i] = C[Ind[0:t + 1, i], i]
                    t = t + 1
        else:
            Cp = C
        return Cp

@func_timeout(1200)
def post_proC(C, K, d, alpha):
        # C: coefficient matrix, K: number of clusters, d: dimension of each subspace
        n = C.shape[0]
        C = 0.5 * (C + C.T)
        r = d * K + 1
        if r > n:
            r = int(n-1)
        U, S, _ = svds(C, r, v0=np.ones(C.shape[0]))
        U = U[:, ::-1]
        S = np.sqrt(S[::-1])
        S = np.diag(S)
        U = U.dot(S)
        U = normalize(U, norm='l2', axis=1)
        Z = U.dot(U.T)
        Z = Z * (Z > 0)
        L = np.abs(Z ** alpha)
        L = L / L.max()
        L = 0.5 * (L + L.T)
        spectral = SpectralClustering(n_clusters=K, eigen_solver='arpack', affinity='precomputed',
                                      assign_labels='discretize', random_state=42)
        spectral.fit(L)
        grp = spectral.fit_predict(L) # + 1
        return grp, L

def class_acc(y_true, y_pre):
        """
        calculate each class's acc
        :param y_true:
        :param y_pre:
        :return:
        """
        ca = []
        for c in np.unique(y_true):
            y_c = y_true[np.nonzero(y_true == c)]  # find indices of each classes
            y_c_p = y_pre[np.nonzero(y_true == c)]
            acurracy = accuracy_score(y_c, y_c_p)
            ca.append(acurracy)
        ca = np.array(ca)
        return ca

def spixel2pixel_labels(y_pred_sp, association_mat):
        y_pred = spixel_to_pixel_labels(y_pred_sp, association_mat)
        return y_pred.astype('int')

def cal_sp_sc(C, association_mat, ro, n_clusters):
        C = thrC(C, ro)
        y_pred_sp, C_final = post_proC(C, n_clusters, 8, 18)
        y_pred = spixel_to_pixel_labels(y_pred_sp, association_mat)
        return y_pred.astype('int')

def cal_sp_sc_nopos(C, association_mat, n_clusters):
        y_pred_sp = cal_pure_sc(C, n_clusters)
        y_pred = spixel2pixel_labels(y_pred_sp, association_mat)
        return y_pred.astype('int')


def cal_pixel_sc(C, ro, n_clusters):
        C = thrC(C, ro)
        y_pred, C_final = post_proC(C, n_clusters, 8, 18)
        return y_pred.astype('int')

def cal_pure_sc(affinity_mat, n_clusters):
        affinity_mat = 0.5 * (np.abs(affinity_mat) + np.abs(affinity_mat.T))
        spectral = SpectralClustering(n_clusters=n_clusters, eigen_solver='arpack', affinity='precomputed',
                                          assign_labels='discretize', random_state=42)
        spectral.fit(affinity_mat)
        y_pre = spectral.fit_predict(affinity_mat)
        y_pre = y_pre.astype('int')
        return y_pre

def get_acc_map_no_ground(y_pred, gt):
        class_map_full = np.zeros(gt.shape, dtype=np.int8)
        class_map = y_pred.reshape(gt.shape)
        indx_nozero = np.where(gt != 0)
        indx_zero = np.where(gt == 0)
        y_target = gt[indx_nozero]
        y_predict = class_map[indx_nozero] + 1
        res = calacc(y_target, y_predict)
        class_map_part = res[0]
        class_map_full[indx_nozero] = class_map_part
        oa = res[1]
        kappa = res[2]
        nmi = res[3]
        ca = res[6]
        return oa, kappa, nmi, ca, class_map_full
 

def get_acc_map(y_pred, gt):
        class_map_full = np.zeros(gt.shape, dtype=np.int8)
        class_map = y_pred.reshape(gt.shape)
        indx_nozero = np.where(gt != 0)
        indx_zero = np.where(gt == 0)
        y_target = gt[indx_nozero]
        y_predict = class_map[indx_nozero] + 1
        res = calacc(y_target, y_predict)
        class_map_part = res[0]
        class_map_full[indx_nozero] = class_map_part
        for i, j in zip(indx_zero[0], indx_zero[1]):
            for k, l in zip(indx_nozero[0], indx_nozero[1]):
                if class_map[i, j] == class_map[k, l]:
                    class_map_full[i, j] = class_map_full[k, l]
                    break
        oa = res[1]
        kappa = res[2]
        nmi = res[3]
        ca = res[6]
        return oa, kappa, nmi, ca, class_map_full
 
def get_acc(y_pred, gt):
        class_map = y_pred.reshape(gt.shape)
        indx_nozero = np.where(gt != 0)
        y_target = gt[indx_nozero]
        y_predict = class_map[indx_nozero] + 1
        res = calacc(y_target, y_predict)
        oa = res[1]
        kappa = res[2]
        nmi = res[3]
        ca = res[6]
        return oa, kappa, nmi,ca 
 
def kmean(x, k):
    kmeans = KMeans(n_clusters=k)
    y = kmeans.fit_predict(x)
    return y

def lrsc_noiseless(A, tau=0.1):
    #tau = 100/np.linalg.norm(A)**2
    _, S, V = np.linalg.svd(A)
    V = V.T
    lam = S
    r = max(np.sum(lam > 1/np.sqrt(tau)), 1)
    temp = np.identity(r) - np.diag(1/lam[:r]**2/tau)
    C = np.dot(np.dot(V[:, :r], temp), V[:, :r].T) 
    return C
def OT(X, eps):
    n = X.shape[0]
    a, b = np.ones((n,)) / n, np.ones((n,)) / n
    M = ot.dist(X, X)
    M /= M.max()
    C = ot.sinkhorn(a, b, M, eps, numItermax=100,  verbose=True)
    return C
def adjacent_mat(x, n_neighbors=10):
        """
        Construct normlized adjacent matrix, N.B. consider only connection of k-nearest graph
        :param x: array like: n_sample * n_feature
        :return:
        """
        A = kneighbors_graph(x, n_neighbors=n_neighbors, include_self=True).toarray()
        A = A * np.transpose(A)
        D = np.diag(np.reshape(np.sum(A, axis=1) ** -0.5, -1))
        normlized_A = np.dot(np.dot(D, A), D)
        return normlized_A

def gcsc(X, lam):
    A = adjacent_mat(X)
    X_ = np.transpose(X)  # shape: n_dim * n_samples
    X_embedding = np.dot(X_, A)
    I = np.eye(X.shape[0])
    inv = np.linalg.inv(np.dot(np.transpose(X_embedding), X_embedding) + lam* I)
    C = np.dot(np.dot(inv, np.transpose(X_embedding)), X_)
    return C

def ssc(X, lamb, rho=1):
        n = X.shape[0]
        C = np.zeros((n, n))
        Delta = np.zeros((n, n))
        A_pre = np.zeros((n, n))
        I = np.identity(n)
        XX = np.dot(X, X.T)
        err1, err2 = 1e-3, 1e-3
        i, max_iter = 0, 100
        while i < max_iter:
            A = np.dot(XX + rho * (C - np.diag(np.diag(C))) - Delta,
                    np.linalg.inv(XX + rho *  I))
            J = soft_threshold(A + Delta / rho, lamb / rho)
            C = J - np.diag(np.diag(J))
            Delta = Delta + rho * (A - (C - np.diag(np.diag(C))))
            err_ac = np.linalg.norm(A - C, np.inf)
            err_aa = np.linalg.norm(A - A_pre, np.inf)
            A_pre = A
            i +=1
            print('iter: %d, err_ac: %f, err_aa: %f' % (i, err_ac, err_aa))
            if err_ac < err1 and err_aa < err1:
                break
        return C
