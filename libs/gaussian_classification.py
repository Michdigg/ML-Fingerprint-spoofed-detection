import numpy
import scipy.special
from libs.multivariate_gaussian_model import logpdf_GAU_ND
from libs.dimensionality_reduction_lib import within_class_covariance_matrix


def mcol(array):
    return array.reshape((array.shape[0], 1))

def compute_mean(D):
    return mcol(D.mean(1))

def vcol(row):
    return row.reshape((row.size,1))

def vrow(col):
    return col.reshape((1,col.size))

def compute_covariance_matrix(D):
    Dc = D - compute_mean(D)
    return numpy.dot(Dc, Dc.T) / D.shape[1]

def center_data(D):
    mu = vcol(D.mean(1))
    return D - mu

def gaussian_params_stimator(DT, LT):
    k = len(set(LT))
    meansVector = []
    covariancesVector = []
    for c in range(k):
        Dc = DT[:, LT == c]
        muc = compute_mean(Dc)
        Cc = compute_covariance_matrix(Dc)
        meansVector.append(muc)
        covariancesVector.append(Cc)
    return meansVector, covariancesVector

def naive_bayes_gaussian_params_estimator(DT, LT):
    k = len(set(LT))
    meansVector = []
    covariancesVector = []
    for c in range(k):
        Dc = DT[:, LT == c]
        muc = compute_mean(Dc)
        Cc = numpy.diag(numpy.diag(compute_covariance_matrix(Dc)))
        meansVector.append(muc)
        covariancesVector.append(Cc)
    return meansVector, covariancesVector

def tied_gaussian_params_estimator(DT, LT):
    k = len(set(LT))
    meansVector = []
    S = within_class_covariance_matrix(DT, LT)
    covariancesVector = []
    for c in range(k):
        Dc = DT[:, LT == c]
        muc = compute_mean(Dc)
        meansVector.append(muc)
        covariancesVector.append(S)
    return meansVector, covariancesVector

def tied_naive_bayes_gaussian_params_estimator(DT, LT):
    k = len(set(LT))
    meansVector = []
    S = numpy.diag(numpy.diag(within_class_covariance_matrix(DT, LT)))
    covariancesVector = []
    for c in range(k):
        Dc = DT[:, LT == c]
        muc = compute_mean(Dc)
        meansVector.append(muc)
        covariancesVector.append(S)
    return meansVector, covariancesVector

def gaussian_score_evaluator(DE, means, covariances, Pc = []):
    S = []
    k = len(means)
    for c in range(k):
        pdf = logpdf_GAU_ND(DE, means[c], covariances[c])
        S.append(pdf)
    Sjoint = []
    if len(Pc) == 0:
        P = 1 / k
        Sjoint = numpy.exp(numpy.array(S)) * P
    else:
        #todo
        print('Prior probability case to implement')
    
    return Sjoint

def opt_gaussian_score_evaluator(DE, means, covariances, Pc = []):
    S = []
    k = len(means)
    for c in range(k):
        pdf = logpdf_GAU_ND(DE, means[c], covariances[c])
        S.append(pdf)
    logSJoint = []
    if len(Pc) == 0:
        P = 1 / k
        logSJoint = numpy.array(S) + numpy.log(P)
    else:
        #todo
        print('Prior probability case to implement')
    
    return logSJoint

def opt_tied_gaussian_score_evaluator(DE, means, tiedCovariance, Pc = []):
    S = []
    k = len(means)
    for c in range(k):
        pdf = logpdf_GAU_ND(DE, means[c], tiedCovariance)
        S.append(pdf)
    logSJoint = []
    if len(Pc) == 0:
        P = 1 / k
        logSJoint = numpy.array(S) + numpy.log(P)
    else:
        #todo
        print('Prior probability case to implement')
    
    return logSJoint

def gaussian_label_predict(DE, means, covariances, Pc = []):
    SJoint = gaussian_score_evaluator(DE, means, covariances)
    SMarginal = vrow(SJoint.sum(0))
    SPost = SJoint / SMarginal
    return numpy.argmax(SPost, axis=0)

def optimized_gaussian_label_predict(DE, means, covariances, Pc = []):
    logSJoint = opt_gaussian_score_evaluator(DE, means, covariances)
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = numpy.exp(logSPost)
    return numpy.argmax(SPost, axis=0), SPost

def optimized_tied_gaussian_label_predict(DE, means, tiedCovariance, Pc = []):
    logSJoint = opt_tied_gaussian_score_evaluator(DE, means, tiedCovariance)
    logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = numpy.exp(logSPost)
    return numpy.argmax(SPost, axis=0)

class GaussianClassificator:
    def __init__(self, type, prior):
        self.type = type
        self.prior = prior

    def train(self, DTR, LTR):
        means, covariances = (None, None)
        if self.type == "MVG":
            means, covariances = gaussian_params_stimator(DTR, LTR)
        if self.type == "Naive Bayes":
            means, covariances = naive_bayes_gaussian_params_estimator(DTR, LTR)
        if self.type == "Tied":
            means, covariances = tied_gaussian_params_estimator(DTR, LTR)
        if self.type == "Tied Naive Bayes":
            means, covariances = tied_naive_bayes_gaussian_params_estimator(DTR, LTR)
        
        self.means = means
        self.covariances = covariances

        return self.means, self.covariances
    
    def compute_scores(self, DTE):
        likelihoods=[]
        logSJoint=[]
        logSMarginal=0
        for i in range(2):
            mu=self.means[i]
            c=self.covariances[i]
            ll= logpdf_GAU_ND(DTE, mu, c)
            likelihoods.append(ll)
            logSJoint.append(ll+numpy.log(self.prior[i]))
            
        logSMarginal = vrow(scipy.special.logsumexp(logSJoint, axis=0))
        logSPost = logSJoint - logSMarginal
        llr = logSPost[1,:] - logSPost[0,:] - numpy.log(self.prior[1]/self.prior[0])
        
        return llr