import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, 'libs')
sys.path.insert(1, 'dataset')
from libs.utils import *
from libs.dimensionality_reduction_lib import pca_projection, lda_projection
from libs.gaussian_classification import *
from libs.binary_logistic_regression import *

if __name__=='__main__':
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

    for pi in ([0.1, 0.5, 0.9]):
        print("Inizio..." + str(pi))
        lrsPCA = {"6" : [], "7" : [], "8" : [], "9" : [], "None" : [],}
        for l in np.logspace(-6, 2, num=9):
            for i, options in enumerate(optionsList):
                logRegModel = LogRegClassificator(l, pi)
                min_DCF, scores, labels = kfold(DTR, LTR, logRegModel, options)
                lrsPCA[str(options["pca"])].append(min_DCF)
        print("fatto 1")

        lrsPCAZnorm = {"6" : [], "7" : [], "8" : [], "9" : [], "None" : [],}
        for l in np.logspace(-6, 2, num=9):
            for i, options in enumerate(optionsListZNorm):
                logRegModel = LogRegClassificator(l, pi)
                min_DCF, scores, labels = kfold(DTR, LTR, logRegModel, options)
                lrsPCAZnorm[str(options["pca"])].append(min_DCF)
        print("fatto 2")

        plot_log_reg(lrsPCA, lrsPCAZnorm, "plots/logistic_regression/lr_" + str(pi))
        print("Salvato")

    print("")

if __name__=='a':
    """ 1. PLOT FEATURES """

    # remove dataset mean
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