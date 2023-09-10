from mllib import *

if __name__ == '__main__':
    S = vrow(np.load('..\\data\\01_log-reg_scores.npy'))
    L = np.load('..\\data\\01_log-reg_labels.npy')
    
    # folds
    K = 10
    
    # effective prior
    p = 1/11

    # logRatioCumulative = np.array([])
    # cumulativeLabels = np.array([])

    # for i in range(0, K):
    #     (DTR, LTR), (DTE, LTE) = k_fold(S, L, K, i)

    #     # MVG
    #     # compute mean and covariance for all classes
    #     (mu0, C0) = compute_mu_C(DTR, LTR, 0, False)
    #     (mu1, C1) = compute_mu_C(DTR, LTR, 1, False)

    #     # compute score matrix S of shape [2, x], which is the number of classes times the number of samples in the test set
    #     S0 = logpdf_GAU_ND(DTE, mu0, C0)
    #     S1 = logpdf_GAU_ND(DTE, mu1, C1)

    #     logRatio = S1 - S0

    #     logRatioCumulative = np.append(logRatioCumulative, logRatio)
    #     cumulativeLabels = np.append(cumulativeLabels, LTE)

    # np.save('..\\data\\01_log-reg_scores_c_mvg.npy', logRatioCumulative)
    # np.save('..\\data\\01_log-reg_labels_c_mvg.npy', cumulativeLabels)

    # actualDCF, minDCF = bayer_error_plots(logRatioCumulative, cumulativeLabels)
    # np.save('..\\data\\01_log-reg_actualDCF_c_mvg.npy', actualDCF)

    # logRatioCumulative = np.array([])
    # cumulativeLabels = np.array([])

    # for i in range(0, K):
    #     (DTR, LTR), (DTE, LTE) = k_fold(S, L, K, i)

    #     # use maxfun=[>1500], maxiter[>30], factr=[<10**7] to increment precision
    #     x0 = np.zeros(DTR.shape[0] + 1)
    #     x, f, d = sp.optimize.fmin_l_bfgs_b(logreg_obj_weight_wrap(DTR, LTR, 1e-4, p), x0)

    #     w, b = x[0:-1], x[-1]
    #     Sc = np.dot(w, DTE) + b

    #     logRatioCumulative = np.append(logRatioCumulative, Sc)
    #     cumulativeLabels = np.append(cumulativeLabels, LTE)

    # np.save('..\\data\\01_log-reg_scores_c_linear-log.npy', logRatioCumulative)
    # np.save('..\\data\\01_log-reg_labels_c_linear-log.npy', cumulativeLabels)

    # actualDCF, minDCF = bayer_error_plots(logRatioCumulative, cumulativeLabels)
    # np.save('..\\data\\01_log-reg_actualDCF_c_linear-log.npy', actualDCF)

    ## QUADRATIC LOGISTIC REGRESSION PRIOR WEIGHTED ##
    logRatioCumulative = np.array([])
    cumulativeLabels = np.array([])

    expD = expand_feature_space(S)
    for i in range(0, K):
        (DTR, LTR), (DTE, LTE) = k_fold(expD, L, K, i)

        # use maxfun=[>1500], maxiter[>30], factr=[<10**7] to increment precision
        x0 = np.zeros(DTR.shape[0] + 1)
        x, f, d = sp.optimize.fmin_l_bfgs_b(logreg_obj_weight_wrap(DTR, LTR, 1e-4, p), x0)

        w, b = x[0:-1], x[-1]
        Sc = np.dot(w, DTE) + b

        logRatioCumulative = np.append(logRatioCumulative, Sc)
        cumulativeLabels = np.append(cumulativeLabels, LTE)

    np.save('..\\data\\01_log-reg_scores_c_q-log.npy', logRatioCumulative)
    np.save('..\\data\\01_log-reg_labels_c_q-log.npy', cumulativeLabels)

    actualDCF, minDCF = bayer_error_plots(logRatioCumulative, cumulativeLabels)
    np.save('..\\data\\01_log-reg_actualDCF_c_q-log.npy', actualDCF)
    np.save('..\\data\\01_log-reg_minDCF_c_q-log.npy', minDCF)
