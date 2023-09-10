from mllib import *

if __name__ == '__main__':
    DTR, LTR = load('../Train.txt')
    DTE, LTE = load('../Test.txt')

    P = PCA_directions(DTR, 7)
    DTR = np.dot(P.T, DTR)
    DTE = np.dot(P.T, DTE)

    Dc = centering(DTR)
    Ds = std_variances(Dc)
    Dw = whitening(Ds, DTR)
    Dl = l2(Dw)

    DcT = centering(DTE)
    DsT = std_variances(DcT)
    DwT = whitening(DsT, DTE)
    DlT = l2(DwT)

    C = [1]

    # effective prior
    p = 1/11
    
    # folds
    K = 10

    for ci in C:

        logRatioCumulative = np.array([])
        cumulativeLabels = np.array([])

        for i in range(0, K):
            (DTRf, LTRf), (DTEf, LTEf) = k_fold(Dl, LTR, K, i)

            STR = compute_svm_polykernel(DTRf, LTRf, DTEf, 1, ci, 2, 1)

            pemp = LTR.sum()/LTR.shape[0]
            logOdds = np.log(pemp/(1-pemp))

            STR -= logOdds

            logRatioCumulative = np.append(logRatioCumulative, STR)
            cumulativeLabels = np.append(cumulativeLabels, LTEf)
        
        S = compute_svm_polykernel(Dl, LTR, DlT, 1, ci, 2, 1)

        # MVG
        # compute mean and covariance for all classes
        (mu0, C0) = compute_mu_C(vrow(logRatioCumulative), cumulativeLabels, 0, False)
        (mu1, C1) = compute_mu_C(vrow(logRatioCumulative), cumulativeLabels, 1, False)

        # compute score matrix S of shape [2, x], which is the number of classes times the number of samples in the test set
        S0 = logpdf_GAU_ND(S, mu0, C0)
        S1 = logpdf_GAU_ND(S, mu1, C1)

        S = S1 - S0

        print(f'for C={ci}')
        print(DCF_min(p, 1, 1, S, LTE))
        print(DCF_actual(p, 1, 1, S, LTE))

    # print(DCF_actual(p, 1, 1, S, LTE))

    FNR, FPR = DET_curve(p, 1, 1, S, LTE)
    np.save('..\\data\\02_svm_eval_FNR.npy', FNR)
    np.save('..\\data\\02_svm_eval_FPR.npy', FPR)
