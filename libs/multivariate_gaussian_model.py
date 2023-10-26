import numpy as np

def logpdf_GAU_ND(X, mu, C):
    M = X.shape[0]
    sign, logDetSigma = np.linalg.slogdet(C)
    logpdfx = []
    for x in X.T:
        xc = np.array(x).reshape((X.shape[0],1)) - mu
        cInv = np.linalg.inv(C)
        logpdfxi = -M/2*np.log(2*np.pi) - 1/2*logDetSigma - 1/2*np.dot(np.dot(xc.T, cInv), xc).ravel()
        logpdfx.append(logpdfxi)
    return np.array(logpdfx).ravel()

def loglilelihood(X, mu, C):
    return logpdf_GAU_ND(X, mu, C).sum()