from mllib import *

if __name__ == '__main__':
    D, L = load('../Train.txt')

    # folds
    K = 10

    DPCA9 = PCA(D, L, 9)
    DPCA8 = PCA(D, L, 8)
    DPCA7 = PCA(D, L, 7)
    DPCA6 = PCA(D, L, 6)

    # effective prior
    p = 0.9

    logRatioCumulative = np.array([])
    cumulativeLabels = np.array([])

    for i in range(0, K):
        (DTR, LTR), (DTE, LTE) = k_fold(DPCA7, L, K, i)

        # MVG
        # compute mean and covariance for all classes
        (mu0, C0) = compute_mu_C(DTR, LTR, 0, False)
        (mu1, C1) = compute_mu_C(DTR, LTR, 1, False)

        # Naive-Bayes
        # compute mean and covariance for all classes
        # (mu0, C0) = compute_mu_C(DTR, LTR, 0, True)
        # (mu1, C1) = compute_mu_C(DTR, LTR, 1, True)

        # Tied-Covariance
        # C0 = C1 = 1/DTR.shape[1]*(C0*(LTR == 0).sum() + C1*(LTR == 1).sum())

        # compute score matrix S of shape [2, x], which is the number of classes times the number of samples in the test set
        S0 = logpdf_GAU_ND(DTE, mu0, C0)
        S1 = logpdf_GAU_ND(DTE, mu1, C1)

        logRatio = S1 - S0

        logRatioCumulative = np.append(logRatioCumulative, logRatio)
        cumulativeLabels = np.append(cumulativeLabels, LTE)

    mindcf = DCF_min(p, 1, 1, logRatioCumulative, cumulativeLabels)
    print(f"min dcf: {mindcf}")
