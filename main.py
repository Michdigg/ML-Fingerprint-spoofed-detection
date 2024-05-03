import sys
import numpy as np
import itertools
sys.path.insert(0, 'libs')
sys.path.insert(1, 'dataset')
from libs.utils import *
from libs.dimensionality_reduction_lib import pca_projection, lda_projection
from libs.gaussian_classification import *
from libs.binary_logistic_regression import *
from libs.svm import *
from libs.gaussian_mixture_models import *
from libs.calibration import *
from libs.evaluation import *

if __name__== "__main__":
    
    (DTR,LTR), (DTE,LTE)=load('dataset/Train.txt','dataset/Test.txt')
    DTR = DTR.T
    LTR = LTR.T
    DTE = DTE.T

    prior,Cfp,Cfn = (0.5,10,1)
    optionsPca6 =   {"K":5,"pca":6,"pi":0.5,"costs":(1,10),"znorm" : False}
    optionsPca7 =   {"K":5,"pca":7,"pi":0.5,"costs":(1,10),"znorm" : False}
    optionsPca8 =   {"K":5,"pca":8,"pi":0.5,"costs":(1,10),"znorm" : False}
    optionsPca9 =   {"K":5,"pca":9,"pi":0.5,"costs":(1,10),"znorm" : False}
    optionsNoPca =   {"K":5,"pca":None,"pi":0.5,"costs":(1,10),"znorm" : False}
    
    optionsList = [optionsPca6, optionsPca7, optionsPca8, optionsPca9, optionsNoPca]

    optionsPca6ZNorm =   {"K":5,"pca":6,"pi":0.5,"costs":(1,10),"znorm" : True}
    optionsPca7ZNorm =   {"K":5,"pca":7,"pi":0.5,"costs":(1,10),"znorm" : True}
    optionsPca8ZNorm =   {"K":5,"pca":8,"pi":0.5,"costs":(1,10),"znorm" : True}
    optionsPca9ZNorm =   {"K":5,"pca":9,"pi":0.5,"costs":(1,10),"znorm" : True}
    optionsNoPcaZNorm =   {"K":5,"pca":None,"pi":0.5,"costs":(1,10),"znorm" : True}

    optionsListZNorm = [optionsPca6ZNorm, optionsPca7ZNorm, optionsPca8ZNorm, optionsPca9ZNorm, optionsNoPcaZNorm]

    """ 1. PLOT FEATURES """

    DC = center_data(DTR)
    plot_feature(DC, LTR, "plots/feature_display/single_feature/feature_")
    plot_cross_feature(DC, LTR, "plots/feature_display/cross_feature/feature_")

    """ 2. DIMENSIONALITY REDUCTION """

    m = 6
    ## PCA
    DP_PCA = pca_projection(DC, m)
    plot_feature(DP_PCA, LTR, "plots/dimensionality_reduction/PCA/m"+str(m)+"/single_feature/feature_")
    plot_cross_feature(DP_PCA, LTR, "plots/dimensionality_reduction/PCA/m"+str(m)+"/cross_feature/feature_")
    
    # LDA
    DP_LDA, _ = lda_projection(DC, LTR, m)
    plot_feature(DP_LDA, LTR, "plots/dimensionality_reduction/LDA/m"+str(m)+"/single_feature/feature_")
    plot_cross_feature(DP_LDA, LTR, "plots/dimensionality_reduction/LDA/m"+str(m)+"/cross_feature/feature_")

    #LDA + PCA
    _ , W = lda_projection(DC, LTR, m)
    DP_LDAPCA = np.dot(W, DP_PCA)
    plot_feature(DP_LDAPCA, LTR, "plots/dimensionality_reduction/LDA-PCA/m"+str(m)+"/single_feature/feature_")
    plot_cross_feature(DP_LDAPCA, LTR, "plots/dimensionality_reduction/LDA-PCA/m"+str(m)+"/cross_feature/feature_")

    # PCA and variance plot
    component_variance_PCA_plot(DTR, "plots/dimensionality_reduction/PCA/component_variance_plot")

    #Pearson coefficients
    plot_correlations(DTR, "plots/feature_display/pearson_correlation_whole_dataset")
    plot_correlations(DTR[:, LTR == 0], "plots/feature_display/pearson_correlation_authentic_dataset", cmap="Blues")
    plot_correlations(DTR[:, LTR == 1], "plots/feature_display/pearson_correlation_spoofed_dataset", cmap="Reds")

    " ------------------------------------------- TRAINING ------------------------------------------- "

    """ 3. Gaussian Classificator """

    #MVG
    gaussianModelMVG = GaussianClassificator("MVG", [prior, 1-prior])
    mvgMinDCF = []
    for i, options in enumerate(optionsList):
        min_DCF, scores, labels = kfold(DTR, LTR, gaussianModelMVG, options)
        mvgMinDCF.append(min_DCF)

    dimension_DCF_plot_gaussian("MVG", mvgMinDCF)

    #Naive Bayes
    gaussianModelNB = GaussianClassificator("Naive Bayes", [prior, 1-prior])
    mvgNBMinDCF = []
    for i, options in enumerate(optionsList):
        min_DCF, scores, labels = kfold(DTR, LTR, gaussianModelNB, options)
        mvgNBMinDCF.append(min_DCF)

    dimension_DCF_plot_gaussian("naive_bayes", mvgNBMinDCF)
    
    #Tied
    gaussianModelTied = GaussianClassificator("Tied", [prior, 1-prior])
    mvgTiedMinDCF = []
    for i, options in enumerate(optionsList):
        min_DCF, scores, labels = kfold(DTR, LTR, gaussianModelTied, options)
        mvgTiedMinDCF.append(min_DCF)

    dimension_DCF_plot_gaussian("tied", mvgTiedMinDCF)

    #Tied Naive Bayes
    gaussianModelTiedNB = GaussianClassificator("Tied Naive Bayes", [prior, 1-prior])
    mvgNBTiedMinDCF = []
    for i, options in enumerate(optionsList):
        min_DCF, scores, labels = kfold(DTR, LTR, gaussianModelTiedNB, options)
        mvgNBTiedMinDCF.append(min_DCF)

    dimension_DCF_plot_gaussian("tied_naive_bayes", mvgNBTiedMinDCF)

    """ 4. Logistic Regression """

    optionsPca6ZNorm =   {"K":5,"pca":6,"pi":0.5,"costs":(1,10),"znorm" : True}
    optionsPca7ZNorm =   {"K":5,"pca":7,"pi":0.5,"costs":(1,10),"znorm" : True}
    optionsPca8ZNorm =   {"K":5,"pca":8,"pi":0.5,"costs":(1,10),"znorm" : True}
    optionsPca9ZNorm =   {"K":5,"pca":9,"pi":0.5,"costs":(1,10),"znorm" : True}
    optionsNoPcaZNorm =   {"K":5,"pca":None,"pi":0.5,"costs":(1,10),"znorm" : True}

    optionsListZNorm = [optionsPca6ZNorm, optionsPca7ZNorm, optionsPca8ZNorm, optionsPca9ZNorm, optionsNoPcaZNorm]

    # Logistic Regression

    for pi in ([0.1, 0.5, 0.9]):
        lrsPCA = {"6" : [], "7" : [], "8" : [], "9" : [], "None" : []}
        for l in np.logspace(-6, 2, num=9):
            logRegModel = LogRegClassificator(l, pi)
            for i, options in enumerate(optionsList):
                min_DCF, scores, labels = kfold(DTR, LTR, logRegModel, options)
                lrsPCA[str(options["pca"])].append(min_DCF)

        lrsPCAZnorm = {"6" : [], "7" : [], "8" : [], "9" : [], "None" : []}
        for l in np.logspace(-6, 2, num=9):
            logRegModel = LogRegClassificator(l, pi)
            for i, options in enumerate(optionsListZNorm):
                min_DCF, scores, labels = kfold(DTR, LTR, logRegModel, options)
                lrsPCAZnorm[str(options["pca"])].append(min_DCF)

        plot_log_reg(lrsPCA, lrsPCAZnorm, "plots/logistic_regression/lr_" + str(pi) + ".png")
    
    # Quadratic Logistic Regression

    for pi in ([0.1, 0.5, 0.9]):
        lrsPCA = {"6" : [], "7" : [], "8" : [], "9" : [], "None" : []}
        for l in np.logspace(-6, 2, num=9):
            logRegModel = QuadLogRegClassificator(l, pi)
            for i, options in enumerate(optionsList):
                min_DCF, scores, labels = kfold(DTR, LTR, logRegModel, options)
                lrsPCA[str(options["pca"])].append(min_DCF)

        lrsPCAZnorm = {"6" : [], "7" : [], "8" : [], "9" : [], "None" : []}
        for l in np.logspace(-6, 2, num=9):
            logRegModel = QuadLogRegClassificator(l, pi)
            for i, options in enumerate(optionsListZNorm):
                min_DCF, scores, labels = kfold(DTR, LTR, logRegModel, options)
                lrsPCAZnorm[str(options["pca"])].append(min_DCF)
        
        plot_log_reg(lrsPCA, lrsPCAZnorm, "plots/logistic_regression/quadratic/lr_" + str(pi) + ".png")

    """ 5. Support Vector Machines """

    print('ciao')

    # Linear SVM

    optionsSVM1 =   {"K":5,"pca":6,"pi":0.5,"costs":(1,10),"znorm" : False}
    optionsSVM2 =   {"K":5,"pca":8,"pi":0.5,"costs":(1,10),"znorm" : False}
    optionsSVM3 =   {"K":5,"pca":None,"pi":0.5,"costs":(1,10),"znorm" : False}

    optionsListSVM = [optionsSVM1, optionsSVM2, optionsSVM3]

    optionsSVM1Znorm =   {"K":5,"pca":6,"pi":0.5,"costs":(1,10),"znorm" : True}
    optionsSVM2Znorm =   {"K":5,"pca":8,"pi":0.5,"costs":(1,10),"znorm" : True}
    optionsSVM3Znorm =   {"K":5,"pca":None,"pi":0.5,"costs":(1,10),"znorm" : True}

    optionsListSVMZnorm = [optionsSVM1Znorm, optionsSVM2Znorm, optionsSVM3Znorm]

    svm_res = {"6" : [], "8" : [], "None" : []}
    svm_res_Znorm = {"6" : [], "8" : [], "None" : []}
    for C in np.logspace(-5, 2, num=8):
        SVMModel = SVMClassificator(1, C, 0.1)
        for i, options in enumerate(optionsListSVM):
            min_DCF, scores, labels = kfold(DTR, LTR, SVMModel, options)
            min_DCF = min_DCF if min_DCF <= 1 else 1
            svm_res[str(options["pca"])].append(min_DCF)
            
        for i, options in enumerate(optionsListSVMZnorm):
            min_DCF, scores, labels = kfold(DTR, LTR, SVMModel,options)
            min_DCF = min_DCF if min_DCF <= 1 else 1
            svm_res_Znorm[str(options["pca"])].append(min_DCF)
    
    plot_svm(svm_res, svm_res_Znorm, "plots/support_vectors_machines/linear/SVM_linear")

    # Quadratic SVM - Polynomial Kernel
    for value in [0, 1]:  
        quad_svm_res = {"6" : [], "8" : [], "None" : []}
        for K_svm in [0,1]:
            for C in np.logspace(-3, -1, num=3):
                SVMkernelModel = SVMKernelClassificator(K_svm, C, 0.1, "polynomial", value)
                for i, options in enumerate(optionsListSVM):
                    min_DCF, scores, labels = kfold(DTR, LTR, SVMkernelModel, options)
                    min_DCF = min_DCF if min_DCF <= 1 else 1
                    quad_svm_res[str(options["pca"])].append(min_DCF)
            
            plot_svm(svm_res, svm_res_Znorm, "plots/support_vectors_machines/quadratic/polynomial/SVM_polynomialC" + str(value) + "K" + str(K_svm))   
    
    # Quadratic SVM - RBF
    for value in [0.01,0.001,0.0001]:  
        quad_svm_res = {"6" : [], "8" : [], "None" : []}
        for K_svm in [0,1]:
            for C in np.logspace(-3, -1, num=3):
                SVMkernelModel = SVMKernelClassificator(K_svm, C, 0.1, "RBF", value)
                for i, options in enumerate(optionsListSVM):
                    min_DCF, scores, labels = kfold(DTR, LTR, SVMkernelModel, options)
                    min_DCF = min_DCF if min_DCF <= 1 else 1
                    quad_svm_res[str(options["pca"])].append(min_DCF)
            
            plot_svm(svm_res, svm_res_Znorm, "plots/support_vectors_machines/quadratic/RBF/SVM_polynomialGamma" + str(value) + "K" + str(K_svm))   

    """ 6. Gaussian Mixture Models """

    optionsGMM1 =   {"K":5,"pca":6,"pi":0.5,"costs":(1,10),"znorm" : False}
    optionsGMM2 =   {"K":5,"pca":8,"pi":0.5,"costs":(1,10),"znorm" : False}
    optionsGMM3 =   {"K":5,"pca":None,"pi":0.5,"costs":(1,10),"znorm" : False}

    optionsListGMM = [optionsGMM1, optionsGMM2, optionsGMM3]

    modes_auth_spoofed = ["full", "diag", "tied"]
    n_components_auth_spoofed = [1, 2, 4 , 6, 8]
    for modes_a_s in itertools.product(modes_auth_spoofed, modes_auth_spoofed):
        for n_components_a_s in itertools.product(n_components_auth_spoofed, n_components_auth_spoofed):
            gmm_res = {"6" : [], "8" : [], "None" : []}
            GMMModel = GMMClassificator(n_components_a_s, modes_a_s, 0.01, 0.1, prior, Cfp, Cfn) 
            for i, options in enumerate(optionsListGMM):
                min_DCF, scores, labels = kfold(DTR, LTR, GMMModel, options)
                min_DCF = min_DCF if min_DCF <= 1 else 1
                gmm_res[str(options["pca"])].append(min_DCF)
                print("Not-Target mode : " + str(modes_a_s[0]) + " Target mode: " + str(modes_a_s[1]) 
                      + " Not-Target components : " + str(n_components_a_s[0]) + " Target components: " + str(n_components_a_s[1]) 
                      + " PCA: " + str(options["pca"]) + " DCF min: " + str(min_DCF))

            plot_gmm(gmm_res, modes_a_s, n_components_a_s, "plots/gaussian_mixture_models/GMM_AUTH" + str(modes_a_s[1] + str(n_components_a_s[1]) + "_SPOOFED") + str(modes_a_s[1] + str(n_components_a_s[1])))

    " ------------------------------------------- EVALUATION  -------------------------------------------"

    """ 7. LOGISTIC REGRESSION """

    prior,Cfp,Cfn = (0.5,10,1)
    pca= [6, None]
    znorm = [True, False]
    pi_tilde=(prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)

    lrsPCA = {"6" : [], "None" : []}  
    lrsPCAZNorm = {"6" : [], "None" : []}
    print("Inizio")
    for pca_znorm in itertools.product(pca, znorm):
        pca = pca_znorm[0]
        znorm = pca_znorm[1]
        for l in np.logspace(-3, 0, num=4):
            quadLogObj = QuadLogRegClassificator(l, pi_tilde)
            DTRt = DTR
            DTEt = DTE
            if znorm == True:
                DTRt,mu,sigma= normalize_zscore(DTR)
                DTEt,_,_ = normalize_zscore(DTE,mu,sigma)

            if pca is not None:
                DTRt, P = PCA_impl(DTRt, pca)
                DTEt = np.dot(P.T, DTEt)

            quadLogObj.train(DTRt, LTR);
            lr_scores = quadLogObj.compute_scores(DTEt)
            lr_scores = np.array(lr_scores)
            min_DCF,_,_ = DCF_min_impl(lr_scores, LTE, prior, Cfp, Cfn)
            print("pca: " + str(pca_znorm[0]) + " znorm: " + str(pca_znorm[1]) + " l: " + str(l) + " mic_DCF: " + str(min_DCF))
            
            if znorm == False:
                lrsPCA[str(pca)].append(min_DCF)
            else:
                lrsPCAZNorm[str(pca)].append(min_DCF)
    plot_log_reg_ev(lrsPCA, lrsPCAZNorm, "plots/evaluation/quadratic_lr.png")

    """ 8. SVM - POLYNOMIAL KERNEL """

    prior,Cfp,Cfn = (0.5,10,1)
    pca= [6, None]
    znorm = [True, False]
    pi_tilde=(prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)

    lrsPCA = {"6" : [], "None" : []}  
    lrsPCAZNorm = {"6" : [], "None" : []}
    print("Inizio")
    for c in [0,1]:
        for pca_znorm in itertools.product(pca, znorm):
            pca = pca_znorm[0]
            znorm = pca_znorm[1]
            for C in np.logspace(-4, 1, num=6):
                SVMObj = SVMKernelClassificator(0, C, pi_tilde, "polynomial", c)
                DTRt = DTR
                DTEt = DTE
                if znorm == True:
                    DTRt,mu,sigma= normalize_zscore(DTR)
                    DTEt,_,_ = normalize_zscore(DTE,mu,sigma)

                if pca is not None:
                    DTRt, P = PCA_impl(DTRt, pca)
                    DTEt = np.dot(P.T, DTEt)

                SVMObj.train(DTRt, LTR);
                lr_scores = SVMObj.compute_scores(DTEt)
                lr_scores = np.array(lr_scores)
                min_DCF,_,_ = DCF_min_impl(lr_scores, LTE, prior, Cfp, Cfn)
                print("pca: " + str(pca_znorm[0]) + " znorm: " + str(pca_znorm[1]) + " C: " + str(C) + " c: " + str(c) + " mic_DCF: " + str(min_DCF))
                
                if znorm == False:
                    lrsPCA[str(pca)].append(min_DCF)
                else:
                    lrsPCAZNorm[str(pca)].append(min_DCF)
        plot_SVM_ev(lrsPCA, lrsPCAZNorm, "plots/evaluation/SVMPolc " + str(c) + ".png")

    """ 9. SVM - RBF """

    prior,Cfp,Cfn = (0.5,10,1)
    pca= [6, None]
    znorm = [True, False]
    pi_tilde=(prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)

    lrsPCA = {"6" : [], "None" : []}  
    lrsPCAZNorm = {"6" : [], "None" : []}
    print("Inizio")
    for gamma in [0.0001, 0.001, 0.01]:
        for pca_znorm in itertools.product(pca, znorm):
            pca = pca_znorm[0]
            znorm = pca_znorm[1]
            for C in np.logspace(-4, 1, num=6):
                SVMObj = SVMKernelClassificator(0, C, pi_tilde, "RBF", gamma)
                DTRt = DTR
                DTEt = DTE
                if znorm == True:
                    DTRt,mu,sigma= normalize_zscore(DTR)
                    DTEt,_,_ = normalize_zscore(DTE,mu,sigma)

                if pca is not None:
                    DTRt, P = PCA_impl(DTRt, pca)
                    DTEt = np.dot(P.T, DTEt)

                SVMObj.train(DTRt, LTR);
                lr_scores = SVMObj.compute_scores(DTEt)
                lr_scores = np.array(lr_scores)
                min_DCF,_,_ = DCF_min_impl(lr_scores, LTE, prior, Cfp, Cfn)
                print("pca: " + str(pca_znorm[0]) + " znorm: " + str(pca_znorm[1]) + " C: " + str(C) + " gamma: " + str(gamma) + " mic_DCF: " + str(min_DCF))
                
                if znorm == False:
                    lrsPCA[str(pca)].append(min_DCF)
                else:
                    lrsPCAZNorm[str(pca)].append(min_DCF)
        plot_SVM_ev(lrsPCA, lrsPCAZNorm, "plots/evaluation/SVMrbfGamma " + str(gamma) + ".png")

    """ 10. GMM """

    prior,Cfp,Cfn = (0.5,10,1)
    pca= [6, None]

    modes_auth_spoofed = ["full", "diag", "tied"]
    n_components_auth_spoofed = [1, 2, 4 , 6, 8]

    lrsPCA = {"6" : [], "None" : []}  
    print("Inizio")
    for modes_a_s in itertools.product(modes_auth_spoofed, modes_auth_spoofed):
        for n_components_a_s in itertools.product(n_components_auth_spoofed, n_components_auth_spoofed):
            for i, pca in enumerate(pca):
                for C in np.logspace(-4, 1, num=6):
                    GMMObj = GMMClassificator(n_components_a_s, modes_a_s, 0.01, 0.1, prior, Cfp, Cfn) 
                    DTRt = DTR
                    DTEt = DTE

                    if pca is not None:
                        DTRt, P = PCA_impl(DTRt, pca)
                        DTEt = np.dot(P.T, DTEt)

                    GMMObj.train(DTRt, LTR);
                    lr_scores = GMMObj.compute_scores(DTEt)
                    lr_scores = np.array(lr_scores)
                    min_DCF,_,_ = DCF_min_impl(lr_scores, LTE, prior, Cfp, Cfn)
                    
                    lrsPCA[str(pca)].append(min_DCF)

            plot_GMM_ev(lrsPCA, "plots/evaluation/GMM " + str(n_components_a_s) + str(modes_a_s) + pca + ".png")

    " ------------------------------------------- CALIBRATION  ------------------------------------------- "

    """ 11. Quadratic Logistic regression """

    prior,Cfp,Cfn = (0.5,10,1)
    l=0.01
    pi_tilde=(prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
    QuadLogReg = QuadLogRegClassificator(l, pi_tilde)

    LogObj = LRCalibrClass(1e-2, 0.5)

    options={ "K" : 5, "pi": 0.5, "pca" : 6, "costs" :(1,10), "logCalibration" :LogObj, "znorm": False }

    DCF_effPrior,DCF_effPrior_min,lr_not_calibr_scores,lr_labels = kfold_calib(DTR, LTR, QuadLogReg, options,True)

    post_prob = binary_posterior_prob(scores, pi, Cfn, Cfp)
    thresholds = np.sort(post_prob)
    lr_FPR,lr_TPR = ROC_plot(thresholds, post_prob, lr_labels)

    """ 12. SVM """

    prior,Cfp,Cfn = (0.5,10,1)
    C=10
    gamma = 1e-3
    pi_tilde=(prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)
    SVMObj = SVMClassificator(0, C, pi_tilde, "polynomial", gamma) 

    LogObj = LRCalibrClass(1e-2, 0.5)

    options={ "K" : 5, "pi": 0.5, "pca" : 6, "costs" :(1,10), "logCalibration" :LogObj, "znorm": False }

    DCF_effPrior,DCF_effPrior_min,lr_not_calibr_scores,lr_labels = kfold_calib(DTR, LTR, SVMObj, options,True)

    post_prob = binary_posterior_prob(scores, pi, Cfn, Cfp)
    thresholds = np.sort(post_prob)
    lr_FPR,lr_TPR = ROC_plot(thresholds, post_prob, lr_labels)

    """ 13. GMM """

    prior,Cfp,Cfn = (0.5,10,1)
    max_comps = [2,8]
    modes = ["diag", "diag"]
    psi=0.01
    alpha=0.1
    pca=None
    pi_tilde=(prior * Cfn) / (prior * Cfn + (1 - prior) * Cfp)

    GMMObj = GMMClassificator(max_comps, modes, psi, alpha, prior, Cfp, Cfn)

    LogObj = LRCalibrClass(1e-2, 0.5)

    options={ "K" : 5, "pi": 0.5, "pca" : 6, "costs" :(1,10), "logCalibration" :LogObj, "znorm": False }

    DCF_effPrior,DCF_effPrior_min,gmm_not_calibr_scores,gmm_labels = kfold_calib(DTR,LTR,GMMObj,options,True)
    post_prob = binary_posterior_prob(gmm_not_calibr_scores,prior,Cfn,Cfp)
    thresholds = np.sort(post_prob)
    gmm_FPR,gmm_TPR = ROC_plot(thresholds,post_prob,gmm_labels)   
