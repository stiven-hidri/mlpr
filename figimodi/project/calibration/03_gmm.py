from mllib import *

if __name__ == '__main__':
    D, L = load('../Train.txt')

    DPCA7 = PCA(D, L, 7)

    Kc1 = 1
    Kc0 = 4

    # folds
    K = 10
 
    # effective prior
    p = 1/11

    logRatioCumulative = np.array([])
    cumulativeLabels = np.array([])

    for i in range(0, K):
        (DTR, LTR), (DTE, LTE) = k_fold(DPCA7, L, K, i)

        DTR0 = DTR[:, LTR == 0]
        DTR1 = DTR[:, LTR == 1]

        gmm0, _ = LBG_wrap(DTR0, n_iter=int(np.log2(Kc0) + 1))
        gmm1, _ = LBG_wrap(DTR1, n_iter=int(np.log2(Kc1) + 1))

        S0 = logpdf_GMM(DTE, gmm0)
        S1 = logpdf_GMM(DTE, gmm1)

        S = S1 - S0

        logRatioCumulative = np.append(logRatioCumulative, S)
        cumulativeLabels = np.append(cumulativeLabels, LTE)

    np.save('..\\data\\03_gmm_scores.npy', logRatioCumulative)
    np.save('..\\data\\03_gmm_labels.npy', cumulativeLabels)

    minDCF = DCF_min(p, 1, 1, logRatioCumulative, cumulativeLabels)
    actualDCF = DCF_actual(p, 1, 1, logRatioCumulative, cumulativeLabels)

    print(f"minDCF={minDCF}, actualDCF={actualDCF}")

    TPR, FPR = ROC_curve(p, 1, 1, logRatioCumulative, cumulativeLabels)
    np.save('..\\data\\03_gmm_TPR.npy', TPR)
    np.save('..\\data\\03_gmm_FPR.npy', FPR)   

    FNR, FPR = DET_curve(p, 1, 1, logRatioCumulative, cumulativeLabels)
    np.save('..\\data\\03_gmm_FNR.npy', FNR)
    np.save('..\\data\\03_gmm_reg_FPR.npy', FPR)

    actualDCF, minDCF = bayer_error_plots(logRatioCumulative, cumulativeLabels)
    np.save('..\\data\\03_gmm_actualDCF.npy', actualDCF)
    np.save('..\\data\\03_gmm_minDCF.npy', minDCF)