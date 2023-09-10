from mllib import *

if __name__ == '__main__':
    D, L = load('../Train.txt')

    DPCA7 = PCA(D, L, 7)

    Dc = centering(DPCA7)
    Ds = std_variances(Dc)
    Dw = whitening(Ds, DPCA7)
    Dl = l2(Dw)

    # C values
    C = 1

    # folds
    K = 10

    # parameters of svm
    KSVM = 1
    degree = 2
    c_poly = 1
 
    # effective prior
    p = 1/11


    logRatioCumulative = np.array([])
    cumulativeLabels = np.array([])

    for i in range(0, K):
        (DTR, LTR), (DTE, LTE) = k_fold(Dl, L, K, i)

        S = compute_svm_polykernel(DTR, LTR, DTE, KSVM, C, degree, c_poly)

        pemp = LTR.sum()/LTR.shape[0]
        logOdds = np.log(pemp/(1-pemp))

        S -= logOdds

        logRatioCumulative = np.append(logRatioCumulative, S)
        cumulativeLabels = np.append(cumulativeLabels, LTE)

    np.save('..\\data\\02_svm_scores.npy', logRatioCumulative)
    np.save('..\\data\\02_svm_labels.npy', cumulativeLabels)

    minDCF = DCF_min(p, 1, 1, logRatioCumulative, cumulativeLabels)
    actualDCF = DCF_actual(p, 1, 1, logRatioCumulative, cumulativeLabels)

    print(f"minDCF={minDCF}, actualDCF={actualDCF}")

    TPR, FPR = ROC_curve(p, 1, 1, logRatioCumulative, cumulativeLabels)
    np.save('..\\data\\02_svm_TPR.npy', TPR)
    np.save('..\\data\\02_svm_FPR.npy', FPR)    

    FNR, FPR = DET_curve(p, 1, 1, logRatioCumulative, cumulativeLabels)
    np.save('..\\data\\02_svm_FNR.npy', FNR)
    np.save('..\\data\\02_svm_FPR.npy', FPR)

    actualDCF, minDCF = bayer_error_plots(logRatioCumulative, cumulativeLabels)
    np.save('..\\data\\02_svm_actualDCF.npy', actualDCF)
    np.save('..\\data\\02_svm_minDCF.npy', minDCF)
