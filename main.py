import sys
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, 'libs')
sys.path.insert(1, 'dataset')
from libs.utils import load, center_data, plot_feature, plot_cross_feature, component_variance_PCA_plot, kfold, dimension_DCF_plot_gaussian
from libs.dimensionality_reduction_lib import pca_projection, lda_projection
from libs.gaussian_classification import *

if __name__=='__main__':
    (DTR,LTR), (DTE,LTE)=load('dataset/Train.txt','dataset/Test.txt')
    DTR = DTR.T
    LTR = LTR.T
    DTE = DTE.T

    ## plot the features
    # remove dataset mean
    DC = center_data(DTR)
    plot_feature(DC, LTR, "plots/feature_display/single_feature/feature_")
    plot_cross_feature(DC, LTR, "plots/feature_display/cross_feature/feature_")

    ##dimensionality reduction
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

    prior,Cfp,Cfn = (0.5,10,1)
    optionsPca6 =   {"K":5,"pca":6,"pi":0.5,"costs":(1,10),"znorm" : False}
    optionsPca7 =   {"K":5,"pca":7,"pi":0.5,"costs":(1,10),"znorm" : False}
    optionsPca8 =   {"K":5,"pca":8,"pi":0.5,"costs":(1,10),"znorm" : False}
    optionsPca9 =   {"K":5,"pca":9,"pi":0.5,"costs":(1,10),"znorm" : False}
    optionsNoPca =   {"K":5,"pca":None,"pi":0.5,"costs":(1,10),"znorm" : False}
    
    optionsList = [optionsPca6, optionsPca7, optionsPca8, optionsPca9, optionsNoPca]

    ## MVG

    #MVG
    mvgMinDCF = []
    for i, options in enumerate(optionsList):
        min_DCF, scores, labels = kfold(DTR, LTR, gaussian_params_stimator, options)
        mvgMinDCF.append(min_DCF)

    dimension_DCF_plot_gaussian("MVG", mvgMinDCF)

    #Naive Bayes
    mvgNBMinDCF = []
    for i, options in enumerate(optionsList):
        min_DCF, scores, labels = kfold(DTR, LTR, naive_bayes_gaussian_params_estimator, options)
        mvgNBMinDCF.append(min_DCF)

    dimension_DCF_plot_gaussian("naive_bayes", mvgNBMinDCF)
    
    #Tied
    mvgTiedMinDCF = []
    for i, options in enumerate(optionsList):
        min_DCF, scores, labels = kfold(DTR, LTR, tied_gaussian_params_estimator, options)
        mvgTiedMinDCF.append(min_DCF)

    dimension_DCF_plot_gaussian("tied", mvgTiedMinDCF)

    #Tied Naive Bayes
    mvgNBTiedMinDCF = []
    for i, options in enumerate(optionsList):
        min_DCF, scores, labels = kfold(DTR, LTR, tied_naive_bayes_gaussian_params_estimator, options)
        mvgNBTiedMinDCF.append(min_DCF)

    dimension_DCF_plot_gaussian("tied_naive_bayes", mvgNBTiedMinDCF)

    print("")