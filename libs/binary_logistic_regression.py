import scipy.optimize
import numpy
from utils import vcol

class LogRegClassificator:
    def __init__(self,l,piT):
        self.l = l
        self.piT = piT
    
    def logreg_obj(self,v):
        loss = 0
        loss_c0 = 0
        loss_c1 = 0
        w,b = v[0:-1],v[-1]
        w = vcol(w)
        n = self.DTR.shape[1]
        regularization = (self.l / 2) * numpy.sum(w ** 2) 
        for i in range(n):
            if (self.LTR[i:i+1] == 1):
                zi = 1
                loss_c1 += numpy.logaddexp(0,-zi * (numpy.dot(w.T,self.DTR[:,i:i+1]) + b))
            else:
                zi=-1
                loss_c0 += numpy.logaddexp(0,-zi * (numpy.dot(w.T,self.DTR[:,i:i+1]) + b))
        J = regularization + (self.piT / self.nT) * loss_c1 + (1-self.piT)/self.nF * loss_c0
        return J
   

    
    def train(self,DTR,LTR):
        self.DTR  = DTR
        self.LTR = LTR
        x0 = numpy.zeros(DTR.shape[0] + 1)
        self.nT = len(numpy.where(LTR == 1)[0])
        self.nF = len(numpy.where(LTR == 0)[0])
        params,_,_ = scipy.optimize.fmin_l_bfgs_b(self.logreg_obj, x0,approx_grad=True)
        self.b = params[-1]
        self.w = numpy.array(params[0:-1])
        self.S = []
        return self.b,self.w
    
    def compute_scores(self,DTE):
        S = self.S
        for i in range(DTE.shape[1]):
            x = DTE[:,i:i+1]
            x = numpy.array(x)
            x = x.reshape((x.shape[0],1))
            self.S.append(numpy.dot(self.w.T,x) + self.b)
        
        S = [1 if x > 0 else 0 for x in S]
        
        llr = numpy.dot(self.w.T, DTE) + self.b
        
        return llr
    
class QuadLogRegClassificator:
    def __init__(self,l,piT):
        self.l = l
        self.piT = piT

    def __gradient_test(self,DTR, LTR, l, pt,nt,nf):
        z=numpy.empty((LTR.shape[0]))    
        z=2*LTR-1
        def gradient(v):
            w, b = v[0:-1], v[-1]
            second_term=0        
            third_term = 0
            first_term = l*w
            for i in range(DTR.shape[1]):            
                S=numpy.dot(w.T,DTR[:,i])+b            
                ziSi = numpy.dot(z[i], S)
                if LTR[i] == 1:
                    internal_term = numpy.dot(numpy.exp(-ziSi),(numpy.dot(-z[i],DTR[:,i])))/(1+numpy.exp(-ziSi))                #print(1+np.exp(-ziSi))
                    second_term += internal_term            
                else :
                    internal_term_2 = numpy.dot(numpy.exp(-ziSi),(numpy.dot(-z[i],DTR[:,i])))/(1+numpy.exp(-ziSi))                
                    third_term += internal_term_2
                    
            derivative_w= first_term + (pt/nt)*second_term + (1-pt)/(nf) * third_term
            
            first_term = 0                   
            second_term=0
    
            for i in range(DTR.shape[1]):            
                S=numpy.dot(w.T,DTR[:,i])+b
                ziSi = numpy.dot(z[i], S)
                if LTR[i] == 1:                
                    internal_term = (numpy.exp(-ziSi) * (-z[i]))/(1+numpy.exp(-ziSi))
                    first_term += internal_term            
                else :
                    internal_term_2 = (numpy.exp(-ziSi) * (-z[i]))/(1+numpy.exp(-ziSi))                
                    second_term += internal_term_2
        
            derivative_b= (pt/nt)*first_term + (1-pt)/(nf) * second_term        
            grad = numpy.hstack((derivative_w,derivative_b))
            return grad
        return gradient
        
    def quad_logreg_obj(self,v):
        loss = 0
        loss_c0 = 0
        loss_c1 = 0        
        
        w,b = v[0:-1],v[-1]
        w = vcol(w)
        n = self.fi_x.shape[1]
        
        regularization = (self.l / 2) * numpy.sum(w ** 2) 
        
        for i in range(n):
            
            if (self.LTR[i:i+1] == 1):
                zi = 1
                loss_c1 += numpy.logaddexp(0,-zi * (numpy.dot(w.T,self.fi_x[:,i:i+1]) + b))
            else:
                zi=-1
                loss_c0 += numpy.logaddexp(0,-zi * (numpy.dot(w.T,self.fi_x[:,i:i+1]) + b))
        
        J = regularization + (self.piT / self.nT) * loss_c1 + (1-self.piT)/self.nF * loss_c0
        grad = self.grad_funct(v)
        return J,grad
    
    def train(self,DTR,LTR):
        self.DTR  = DTR
        self.LTR = LTR
        self.nt = DTR[:, LTR == 1].shape[1]
        self.nf = DTR.shape[1]-self.nt
        
        def vecxxT(x):
            x = x[:, None]
            xxT = x.dot(x.T).reshape(x.size ** 2, order='F')
            return xxT
        
        expanded_DTR = numpy.apply_along_axis(vecxxT, 0, DTR)
        self.fi_x = numpy.vstack([expanded_DTR, DTR])
        
        x0 = numpy.zeros(self.fi_x.shape[0] + 1)
        
        self.nT = len(numpy.where(LTR == 1)[0])
        self.nF = len(numpy.where(LTR == 0)[0])
        self.grad_funct = self.__gradient_test(self.fi_x, self.LTR, self.l, self.piT,self.nt,self.nf)

        params,_,_ = scipy.optimize.fmin_l_bfgs_b(self.quad_logreg_obj, x0)
        
        self.b = params[-1]
        self.w = numpy.array(params[0:-1])
        self.S = []

        return self.b,self.w
    
    def compute_scores(self,DTE):
        S = self.S
       
        for i in range(DTE.shape[1]):
            x = vcol(DTE[:,i:i+1])
            mat_x = numpy.dot(x,x.T)
            vec_x= vcol(numpy.hstack(mat_x))
            fi_x = numpy.vstack((vec_x,x))
            self.S.append(numpy.dot(self.w.T,fi_x) + self.b)
            
        pred = [1 if x > 0 else 0 for x in S]
        
        return S