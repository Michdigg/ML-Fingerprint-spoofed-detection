import numpy as np
import matplotlib.pyplot as plt

def confusion_matrix(pred,LTE):
    nclasses = int(np.max(LTE))+1
    matrix = np.zeros((nclasses,nclasses))
    
    
    for i in range(len(pred)):
        matrix[pred[i],LTE[i]] += 1
        
    return matrix

def binary_posterior_prob(llr,prior,Cfn,Cfp):
    new_llr = np.zeros(llr.shape);
    for i in range(len(llr)):
        new_llr[i] = llr[i] + np.log(prior*Cfn/((1-prior)*Cfp))
        
    return new_llr

def binary_DCFu(prior,Cfn,Cfp,cm):
    FNR = cm[0,1]/(cm[0,1]+cm[1,1])
    FPR = cm[1,0]/(cm[1,0]+cm[0,0])
    
    DCFu = prior*Cfn*FNR + (1-prior)*Cfp*FPR
    
    return DCFu


def ROC_plot(thresholds,post_prob,LTE):
    FNR=[]
    FPR=[]
    for t in thresholds:
        pred = [1 if x >= t else 0 for x in post_prob]
        cm = confusion_matrix(pred, LTE)
            
        FNR.append(cm[0,1]/(cm[0,1]+cm[1,1]))
        FPR.append(cm[1,0]/(cm[1,0]+cm[0,0]))
        
    plt.figure()
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    TPR=1-np.array(FNR)
    plt.plot(FPR,TPR,scalex=False,scaley=False)
    plt.show()
    
    return FPR,TPR
    
def Bayes_plot(llr,LTE):
    effPriorLogOdds = np.linspace(-3, 3,21)
    DCF_effPrior = {}
    DCF_effPrior_min = {}
    print("In bayes_plot")
    for p in effPriorLogOdds:
       
        effPrior = 1/(1+np.exp(-p))

        post_prob = binary_posterior_prob(llr,effPrior,1,1)
        pred = [1 if x >=0 else 0 for x in post_prob]
        cm = confusion_matrix(pred, LTE)
        
        dummy = min(effPrior,(1-effPrior))
        
        DCF_effPrior[p] = (binary_DCFu(effPrior, 1, 1, cm)/dummy)
        
        thresholds = np.sort(post_prob)
        tmp_DCF = []
        for t in thresholds:
            pred = [1 if x >=t else 0 for x in post_prob]
            cm = confusion_matrix(pred, LTE)
            tmp_DCF.append((binary_DCFu(effPrior, 1, 1, cm)/dummy))
        
        DCF_effPrior_min[p] = (np.min(tmp_DCF)) 
        
    plt.plot(effPriorLogOdds, DCF_effPrior.values(), label='DCF', color='r')
    plt.plot(effPriorLogOdds, DCF_effPrior_min.values(), label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.legend()
    plt.show()
    
    return DCF_effPrior,DCF_effPrior_min

def DCF_norm_impl(llr,label,prior,Cfp,Cfn):
        
        post_prob = binary_posterior_prob(llr,prior,Cfn,Cfp)
        pred = [1 if x > 0 else 0 for x in post_prob]
        cm = confusion_matrix(pred, label)
        
        DCFu = binary_DCFu(prior,Cfn,Cfp,cm)
        
        dummy_DCFu = min(prior*Cfn,(1-prior)*Cfp)
        DCF_norm = DCFu/dummy_DCFu
        
        return DCF_norm
    
def DCF_min_impl(llr,label,prior,Cfp,Cfn):
    post_prob = binary_posterior_prob(llr,prior,Cfn,Cfp)
    thresholds = np.sort(post_prob)
    DCF_tresh = []
    dummy_DCFu = min(prior*Cfn,(1-prior)*Cfp)
    
    for t in thresholds:
        
        pred = [1 if x >= t else 0 for x in post_prob]
        cm = confusion_matrix(pred, label)
        DCF_tresh.append(binary_DCFu(prior, Cfn, Cfp, cm)/dummy_DCFu)
        
    min_DCF = min(DCF_tresh)
    t_min = thresholds[np.argmin(DCF_tresh)]
        
    return min_DCF,t_min,thresholds
