from mllib import *

if __name__ == '__main__':
    D, L = load('../Train.txt')
    
    DPCA7 = PCA(D, L, 7)

    Dc = centering(DPCA7)
    Ds = std_variances(Dc)
    Dw = whitening(Ds, DPCA7)
    Dl = l2(Dw)
    expD = expand_feature_space(Dl)

    # folds
    K = 10

    # lambda
    l = 1e-4

    # threshold
    p = 1/11

    logRatioCumulative = np.array([])
    cumulativeLabels = np.array([])

    for i in range(0, K):
        (DTR, LTR), (DTE, LTE) = k_fold(expD, L, K, i)

        # use maxfun=[>1500], maxiter[>30], factr=[<10**7] to increment precision
        x0 = np.zeros(DTR.shape[0] + 1)
        x, f, d = sp.optimize.fmin_l_bfgs_b(logreg_obj_wrap(DTR, LTR, l), x0)

        w, b = x[0:-1], x[-1]
        S = np.dot(w, DTE) + b

        pemp = LTR.sum()/LTR.shape[0]
        logOdds = np.log(pemp/(1-pemp))

        S -= logOdds

        logRatioCumulative = np.append(logRatioCumulative, S)
        cumulativeLabels = np.append(cumulativeLabels, LTE)

    np.save('..\\data\\01_log-reg_scores.npy', logRatioCumulative)
    np.save('..\\data\\01_log-reg_labels.npy', cumulativeLabels)

    minDCF = DCF_min(p, 1, 1, logRatioCumulative, cumulativeLabels)
    actualDCF = DCF_actual(p, 1, 1, logRatioCumulative, cumulativeLabels)

    print(f"minDCF={minDCF}, actualDCF={actualDCF}")

    TPR, FPR = ROC_curve(p, 1, 1, logRatioCumulative, cumulativeLabels)
    np.save('..\\data\\01_log-reg_TPR.npy', TPR)
    np.save('..\\data\\01_log-reg_FPR.npy', FPR)

    FNR, FPR = DET_curve(p, 1, 1, logRatioCumulative, cumulativeLabels)
    np.save('..\\data\\01_log-reg_FNR.npy', FNR)
    np.save('..\\data\\01_log-reg_FPR.npy', FPR)

    actualDCF, minDCF = bayer_error_plots(logRatioCumulative, cumulativeLabels)
    np.save('..\\data\\01_log-reg_actualDCF.npy', actualDCF)
    np.save('..\\data\\01_log-reg_minDCF.npy', minDCF)
