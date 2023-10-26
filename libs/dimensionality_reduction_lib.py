import scipy.linalg
import numpy as np

def mcol(array):
    return array.reshape((array.shape[0], 1))

def compute_mean(D):
    return mcol(D.mean(1))

def vcol(row):
    return row.reshape((row.size,1))

def compute_covariance_matrix(D):
    Dc = D - compute_mean(D)
    return np.dot(Dc, Dc.T) / D.shape[1]

def center_data(D):
    mu = vcol(D.mean(1))
    return D - mu

def between_class_covariance_matrix(D, L):
    N = D.shape[1]
    mu = compute_mean(D)
    k = len(set(L))
    Sb = np.zeros((int(D.shape[0]), int(D.shape[0])))
    for c in range(k):
        Dc = D[:, L == c]
        nc = Dc.shape[1]
        muc = compute_mean(Dc)
        Sb = Sb + nc * np.dot((muc - mu), (muc - mu).T)
    Sb = Sb / N
    return Sb

def within_class_covariance_matrix(D, L):
    N = D.shape[1]
    k = len(set(L))
    Sw = np.zeros((int(D.shape[0]), int(D.shape[0])))
    for c in range(k):
        Dc = D[:, L == c]
        nc = Dc.shape[1]
        Swc = compute_covariance_matrix(Dc)
        Sw = Sw + nc*Swc
    Sw = Sw / N
    return Sw

def pca_projection(D, m):
    C = compute_covariance_matrix(D)
    U, s, Vh = np.linalg.svd(C)
    P = U[:, 0:m]
    DP = np.dot(P.T, D)
    return DP

def lda_projection(D,L,m):
    SB = between_class_covariance_matrix(D,L)
    SW = within_class_covariance_matrix(D,L)
    s, U = scipy.linalg.eigh(SB, SW)
    W = U[:, ::-1][:, 0:m]
    DP = np.dot(W.T, D)
    return DP, W

def PCA_impl(D, m):
    DC = center_data(D)
    C = compute_covariance_matrix(DC)
    s, U = np.linalg.eigh(C)
    P = U[:, ::-1][:, 0:m]
    DP = np.dot(P.T, D)
    
    return DP,P