import numpy
import scipy.optimize

def mcol(v):
    return v.reshape((v.size,1))
def mRow(v):
    return v.reshape((1,v.size))

class SVMClassificator:
    
    def __init__(self,K,C,piT):
        self.K = K
        self.C = C
        self.piT = piT
        self.Z = []
        self.D_ = []
        self.w_hat_star = []
        self.f = []
    
    def __compute_lagrangian_wrapper(self,H):
        def compute_lagrangian(alpha):
            alpha = alpha.reshape(-1, 1)
            Ld_alpha = 0.5 * alpha.T @ H @ alpha - numpy.sum(alpha)
            gradient = H @ alpha - 1
            return Ld_alpha.item(), gradient.flatten()
        return compute_lagrangian
    
    def __compute_H(self,D_,LTR):
        Z = numpy.where(LTR == 0, -1, 1).reshape(-1, 1)
        Gb = D_.T @ D_
        Hc = Z @ Z.T * Gb
        return Z, Hc
    
    def train(self,DTR,LTR):
        nf = DTR[:,LTR==0].shape[1]
        nt = DTR[:,LTR==1].shape[1]
        emp_prior_f = nf/ DTR.shape[1]
        emp_prior_t = nt/ DTR.shape[1]
        Cf = self.C * self.piT / emp_prior_f
        Ct = self.C * self.piT / emp_prior_t
        
        K_row = numpy.ones((1, DTR.shape[1])) * self.K
        D_ = numpy.vstack((DTR, K_row))
        self.D_ = D_
        self.Z,H_=self.__compute_H(D_,LTR)
        compute_lag=self.__compute_lagrangian_wrapper(H_)
        bound_list=[(-1,-1)]*DTR.shape[1]
        
        for i in range(DTR.shape[1]):
            if LTR[i] == 0:
                bound_list[i] = (0,Cf)
            else:
                bound_list[i] = (0,Ct)
        
        (alfa,self.f,d)=scipy.optimize.fmin_l_bfgs_b(compute_lag,x0=numpy.zeros(LTR.size),approx_grad=False,factr=1.0,bounds=bound_list)
        w_hat_star = (mcol(alfa)* self.Z * D_.T).sum(axis=0)
        self.w_hat_star = w_hat_star
        
    def compute_scores(self,DTE):
        K_row2 = numpy.ones((1, DTE.shape[1])) * self.K
        D2_ = numpy.vstack((DTE, K_row2))
        score = numpy.dot(self.w_hat_star, D2_)
        
        return score
    
def polynomial_kernel_with_bias(x1, x2,xi,ci):
     d=2
     return ((numpy.dot(x1.T, x2) + ci) ** d) + xi

def rbf_kernel_with_bias(x1, x2,xi, gamma):
     return numpy.exp(-gamma * numpy.sum((x1 - x2) ** 2)) + xi

class SVMKernelClassificator:
    
    def __init__(self,K,C,piT,mode,ci):
        self.K = K
        self.C = C
        self.piT = piT
        self.mode = mode
        self.alfa = []
        self.ci  = ci               
        self.DTR = []
        self.LTR = []
        
        
        if mode == "poly":
            self.kernel_func = polynomial_kernel_with_bias
        elif mode == "rbf":
            self.kernel_func = rbf_kernel_with_bias
        else:
            self.kernel_func = polynomial_kernel_with_bias
            
    
    def __compute_kernel_score(self,alpha, DTR, L, kernel_func, x,xi,ci):
         Z = numpy.zeros(L.shape)
         Z[L == 1] = 1
         Z[L == 0] = -1
         score = 0
         for i in range(alpha.shape[0]):
             if alpha[i] > 0:
                 score += alpha[i]*Z[i]* kernel_func(DTR[:, i],x,xi,ci)
         return score
    
    
    def __compute_lagrangian_wrapper(self,H):
        def compute_lagrangian(alpha):
            alpha = alpha.reshape(-1, 1)
            Ld_alpha = 0.5 * alpha.T @ H @ alpha - numpy.sum(alpha)
            gradient = H @ alpha - 1
            return Ld_alpha.item(), gradient.flatten()
        return compute_lagrangian
    
    def __compute_H(self,DTR,LTR,kernel_func,xi,ci):
         n_samples = DTR.shape[1]
         Hc = numpy.zeros((n_samples, n_samples))
         Z = numpy.where(LTR == 0, -1, 1)
         for i in range(n_samples):
             for j in range(n_samples):
                 Hc[i, j] = Z[i]*Z[j]* kernel_func(DTR[:, i], DTR[:, j],xi,ci)
         return Hc

    
    def train(self,DTR,LTR):
        self.DTR = DTR
        self.LTR = LTR
        
        nf = DTR[:,LTR==0].shape[1]
        nt = DTR[:,LTR==1].shape[1]
        emp_prior_f = nf/ DTR.shape[1]
        emp_prior_t = nt/ DTR.shape[1]
        Cf = self.C * self.piT / emp_prior_f
        Ct = self.C * self.piT / emp_prior_t
       
        xi = self.K * self.K
        H_=self.__compute_H(DTR,LTR,self.kernel_func,xi,self.ci)
        compute_lag=self.__compute_lagrangian_wrapper(H_)
        bound_list=[(-1,-1)]*DTR.shape[1]
        
        for i in range(DTR.shape[1]):
            if LTR[i] == 0:
                bound_list[i] = (0,Cf)
            else:
                bound_list[i] = (0,Ct)

        (alfa,f,d)=scipy.optimize.fmin_l_bfgs_b(compute_lag,x0=numpy.zeros(LTR.size),approx_grad=False,factr=1.0,bounds=bound_list)
        self.alfa = alfa
        
    def compute_scores(self,DTE):
        score=numpy.array([self.__compute_kernel_score(self.alfa,self.DTR,self.LTR,self.kernel_func,x,self.K*self.K,self.ci) for x in DTE.T])
        return score