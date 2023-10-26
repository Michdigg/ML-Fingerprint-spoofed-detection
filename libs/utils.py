import numpy as np
import matplotlib.pyplot as plt
import sys
from libs.evaluation import DCF_min_impl
from libs.dimensionality_reduction_lib import PCA_impl
from libs.gaussian_classification import loglikelihoods
sys.path.insert(0, 'plots')

def load(train_path, test_path):
    fT=open(train_path,'r')
    fE=open(test_path,'r')
    DTR=[]
    DTE=[]
    LTE=[]
    LTR=[]
    for line in fT:
        splitted=line.split(',')
        DTR.append([float(i) for i in splitted[:-1]])
        LTR.append(int(splitted[-1]))
    DTR=np.array(DTR)
    LTR=np.array(LTR)
    for line in fE:
        splitted=line.split(',')
        DTE.append([float(i) for i in splitted[:-1]])
        LTE.append(int(splitted[-1]))
    DTE=np.array(DTE)
    LTE=np.array(LTE)
    fT.close()
    fE.close()
    return (DTR, LTR),(DTE,LTE)

def center_data(D):
    mu = vcol(D.mean(1))
    return D - mu

def compute_mean(D):
    return mcol(D.mean(1))

def mcol(array):
    return array.reshape((array.shape[0], 1))

def mrow(array):
    return array.reshape((1, array.shape[0]))

def vrow(col):
    return col.reshape((1,col.size))

def vcol(row):
    return row.reshape((row.size,1))

def compute_covariance_matrix(D):
    Dc = D - compute_mean(D)
    return np.dot(Dc, Dc.T) / D.shape[1]

def plot_feature(D, L, path):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    for i in range(D.shape[0]):
        plt.figure()
        plt.xlabel("Feature " + str(i + 1))
        plt.ylabel("Number of elements")
        plt.hist(D0[i, :], bins=60,density=True, alpha=0.7, label="Spoofed fingerprint")
        plt.hist(D1[i, :], bins=60, density=True, alpha=0.7, label="Authentic fingerprint")
        plt.legend()
        plt.savefig(path + (str(i+1)))
        plt.close()

def plot_cross_feature(D, L, path):
    D0 = D[:, L == 0]
    D1 = D[:, L == 1]

    for i in range(D.shape[0]):
        for j in range(D.shape[0]):
            if i == j:
                continue
            plt.figure()
            plt.xlabel("Feature " + str(i + 1))
            plt.ylabel("Feature " + str(j + 1))
            plt.scatter(D0[i, :], D0[j, :], label="Spoofed fingerprint")
            plt.scatter(D1[i, :], D1[j, :], label="Authentic fingerprint")
            plt.legend()
            plt.savefig(path + (str(i+1)) + ("_") + str((j+1)))
            plt.close()

def component_variance_PCA_plot(D, path):
    # Calcola i valori propri della matrice di covarianza
    DC = center_data(D)
    C = compute_covariance_matrix(DC)
    eigenvalues, _ = np.linalg.eigh(C)
    
    # Ordina i valori propri in ordine decrescente
    eigenvalues = eigenvalues[::-1]
    
    # Calcola la varianza spiegata per ogni componente principale
    explained_variance = eigenvalues / np.sum(eigenvalues)
    y_min, y_max = plt.ylim()
    y_values = np.linspace(y_min, y_max, 20)
    plt.yticks(y_values)
    plt.xlim(right=9)
    # Creare un grafico della varianza spiegata
    plt.plot(np.cumsum(explained_variance))
    plt.xlabel('Components number')
    plt.ylabel('Cumulative variance')
    plt.grid()
    plt.savefig(path)
    plt.close()

def normalize_zscore(D, mu=[], sigma=[]):
    if mu == [] or sigma == []:
        mu = np.mean(D, axis=1)
        sigma = np.std(D, axis=1)
    ZD = D
    ZD = ZD - mcol(mu)
    ZD = ZD / mcol(sigma)
    return ZD, mu, sigma

def kfold(D, L, param_estimator, options):
        
        K = options["K"]
        pca = options["pca"]
        pi = options["pi"]
        (cfn, cfp) = options["costs"]
        znorm = options["znorm"]
        
        samplesNumber = D.shape[1]
        N = int(samplesNumber / K)
        
        np.random.seed(seed=0)
        indexes = np.random.permutation(D.shape[1])
        
        scores = np.array([])
        labels = np.array([])
       
        
        for i in range(K):
            idxTest = indexes[i*N:(i+1)*N]
            
            idxTrainLeft = indexes[0:i*N]
            idxTrainRight = indexes[(i+1)*N:]
            idxTrain = np.hstack([idxTrainLeft, idxTrainRight])
            
            DTR = D[:, idxTrain]
            LTR = L[idxTrain]   
            DTE = D[:, idxTest]
            LTE = L[idxTest]
            
            if znorm == True:
                DTR,mu,sigma= normalize_zscore(DTR)
                DTE,_,_ = normalize_zscore(DTE,mu,sigma)
            
            if pca is not None:
                DTR, P = PCA_impl(DTR, pca)
                DTE = np.dot(P.T, DTE)
                
            means, covariances = param_estimator(DTR, LTR)
            scores_i = loglikelihoods(DTE, means, covariances, [1-pi, pi])
            scores = np.append(scores, scores_i)
            labels = np.append(labels, LTE)
            
            
        labels = np.array(labels,dtype=int)
        min_DCF,_,_ = DCF_min_impl(scores, labels, pi, cfp, cfn)
     
        return min_DCF, scores, labels

def dimension_DCF_plot_gaussian(modelName, DCFList):
    path = "plots/multivariate_gaussian_models/" + modelName + "/dimension_DCF_plot"
    plt.xlabel("PCA dimensions")
    plt.ylabel("DCF_min")
    plt.title(modelName)
    plt.plot(np.linspace(6,10,5),DCFList)
    plt.savefig(path)
    plt.close()