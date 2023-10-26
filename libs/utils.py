import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sys
from libs.evaluation import DCF_min_impl
from libs.dimensionality_reduction_lib import PCA_impl
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
    DC = center_data(D)
    C = compute_covariance_matrix(DC)
    eigenvalues, _ = np.linalg.eigh(C)
    eigenvalues = eigenvalues[::-1]
    explained_variance = eigenvalues / np.sum(eigenvalues)
    y_min, y_max = plt.ylim()
    y_values = np.linspace(y_min, y_max, 20)
    plt.yticks(y_values)
    plt.xlim(right=9)
    plt.plot(np.cumsum(explained_variance))
    plt.xlabel('Components number')
    plt.ylabel('Cumulative variance')
    plt.grid()
    plt.savefig(path)
    plt.close()

def plot_log_reg(lrsPCA, lrsPCAZnorm, path):
    lamb = np.logspace(-7, 2, num=9)

    plt.semilogx(lamb,lrsPCA["6"], label = "PCA 6")
    plt.semilogx(lamb,lrsPCA["7"], label = "PCA 7")
    plt.semilogx(lamb,lrsPCA["8"], label = "PCA 8")
    plt.semilogx(lamb,lrsPCA["9"], label = "PCA 9")
    plt.semilogx(lamb,lrsPCA["None"], label = "No PCA Znorm")
    plt.semilogx(lamb,lrsPCAZnorm["6"], label = "PCA 6 Znorm")
    plt.semilogx(lamb,lrsPCAZnorm["7"], label = "PCA 7 Znorm")
    plt.semilogx(lamb,lrsPCAZnorm["8"], label = "PCA 8 Znorm")
    plt.semilogx(lamb,lrsPCAZnorm["9"], label = "PCA 9 Znorm")
    plt.semilogx(lamb,lrsPCAZnorm["None"], label = "No PCA Znorm")
   
    plt.xlabel("Lambda")
    plt.ylabel("DCF_min")
    plt.legend()
    plt.savefig(path)
    plt.close()

def plot_svm(lrsPCA, lrsPCAZnorm, path):
    C_values = np.logspace(-5, 2, num=8)

    plt.semilogx(C_values,lrsPCA["6"], label = "PCA 7")
    plt.semilogx(C_values,lrsPCA["8"], label = "PCA 8")
    plt.semilogx(C_values,lrsPCA["None"], label = "No PCA Znorm")
    plt.semilogx(C_values,lrsPCAZnorm["6"], label = "PCA 7 Znorm")
    plt.semilogx(C_values,lrsPCAZnorm["8"], label = "PCA 8 Znorm")
    plt.semilogx(C_values,lrsPCAZnorm["None"], label = "No PCA Znorm")
   
    plt.xlabel("C")
    plt.ylabel("DCF_min")
    plt.legend()
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

def kfold(D, L, model, options):
        
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
                
            model.train(DTR, LTR)
            scores_i = model.compute_scores(DTE)
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

def compute_correlation(X, Y):
    x_sum = np.sum(X)
    y_sum = np.sum(Y)

    x2_sum = np.sum(X ** 2)
    y2_sum = np.sum(Y ** 2)

    sum_cross_prod = np.sum(X * Y.T)

    n = X.shape[0]
    numerator = n * sum_cross_prod - x_sum * y_sum
    denominator = np.sqrt((n * x2_sum - x_sum ** 2) * (n * y2_sum - y_sum ** 2))

    corr = numerator / denominator
    return corr

def plot_correlations(DTR, path, cmap="Greys"):
    corr = np.zeros((10, 10))
    for x in range(10):
        for y in range(10):
            X = DTR[x, :]
            Y = DTR[y, :]
            pearson_elem = compute_correlation(X, Y)
            corr[x][y] = pearson_elem

    sns.set()
    heatmap = sns.heatmap(np.abs(corr), linewidth=0.2, cmap=cmap, square=True, cbar=False)
    fig = heatmap.get_figure()
    fig.savefig(path + ".svg")