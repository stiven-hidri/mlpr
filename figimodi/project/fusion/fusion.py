from mllib import *

if __name__ == '__main__':
    S_log = np.load('..\\data\\01_log-reg_scores.npy')
    S_svm = np.load('..\\data\\02_svm_scores.npy')
    S_gmm = np.load('..\\data\\03_gmm_scores.npy')
    L = np.load('..\\data\\01_log-reg_labels.npy')

    S_log_svm = (S_log + S_svm)/2
    S_log_gmm = (S_log + S_gmm)/2
    S_svm_gmm = (S_gmm + S_svm)/2
    S_log_svm_gmm = (S_log + S_gmm + S_svm)/3

    np.save('..\\data\\04_log_svm_scores.npy', S_log_svm)
    np.save('..\\data\\05_log_gmm_scores.npy', S_log_gmm)
    np.save('..\\data\\06_svm_gmm_scores.npy', S_svm_gmm)
    np.save('..\\data\\07_log_svm_gmm_scores.npy', S_log_svm_gmm)

    minDCF_log_svm = DCF_min(1/11, 1, 1, S_log_svm, L)
    actualDCF_log_svm = DCF_actual(1/11, 1, 1, S_log_svm, L)
    minDCF_log_gmm = DCF_min(1/11, 1, 1, S_log_gmm, L)
    actualDCF_log_gmm = DCF_actual(1/11, 1, 1, S_log_gmm, L)
    minDCF_svm_gmm = DCF_min(1/11, 1, 1, S_svm_gmm, L)
    actualDCF_svm_gmm = DCF_actual(1/11, 1, 1, S_svm_gmm, L)
    minDCF_log_svm_gmm = DCF_min(1/11, 1, 1, S_log_svm_gmm, L)
    actualDCF_log_svm_gmm = DCF_actual(1/11, 1, 1, S_log_svm_gmm, L)

    print("prior=0.09")
    print(f"log+svm: minDCF={minDCF_log_svm}, actualDCF={actualDCF_log_svm}")
    print(f"log+gmm: minDCF={minDCF_log_gmm}, actualDCF={actualDCF_log_gmm}")
    print(f"svm+gmm: minDCF={minDCF_svm_gmm}, actualDCF={actualDCF_svm_gmm}")
    print(f"log+svm+gmm: minDCF={minDCF_log_svm_gmm}, actualDCF={actualDCF_log_svm_gmm}")

    minDCF_log_svm = DCF_min(0.5, 1, 1, S_log_svm, L)
    actualDCF_log_svm = DCF_actual(0.5, 1, 1, S_log_svm, L)
    minDCF_log_gmm = DCF_min(0.5, 1, 1, S_log_gmm, L)
    actualDCF_log_gmm = DCF_actual(0.5, 1, 1, S_log_gmm, L)
    minDCF_svm_gmm = DCF_min(0.5, 1, 1, S_svm_gmm, L)
    actualDCF_svm_gmm = DCF_actual(0.5, 1, 1, S_svm_gmm, L)
    minDCF_log_svm_gmm = DCF_min(0.5, 1, 1, S_log_svm_gmm, L)
    actualDCF_log_svm_gmm = DCF_actual(0.5, 1, 1, S_log_svm_gmm, L)

    print("prior=0.5")
    print(f"log+svm: minDCF={minDCF_log_svm}, actualDCF={actualDCF_log_svm}")
    print(f"log+gmm: minDCF={minDCF_log_gmm}, actualDCF={actualDCF_log_gmm}")
    print(f"svm+gmm: minDCF={minDCF_svm_gmm}, actualDCF={actualDCF_svm_gmm}")
    print(f"log+svm+gmm: minDCF={minDCF_log_svm_gmm}, actualDCF={actualDCF_log_svm_gmm}")

    minDCF_log_svm = DCF_min(0.9, 1, 1, S_log_svm, L)
    actualDCF_log_svm = DCF_actual(0.9, 1, 1, S_log_svm, L)
    minDCF_log_gmm = DCF_min(0.9, 1, 1, S_log_gmm, L)
    actualDCF_log_gmm = DCF_actual(0.9, 1, 1, S_log_gmm, L)
    minDCF_svm_gmm = DCF_min(0.9, 1, 1, S_svm_gmm, L)
    actualDCF_svm_gmm = DCF_actual(0.9, 1, 1, S_svm_gmm, L)
    minDCF_log_svm_gmm = DCF_min(0.9, 1, 1, S_log_svm_gmm, L)
    actualDCF_log_svm_gmm = DCF_actual(0.9, 1, 1, S_log_svm_gmm, L)

    print("prior=0.9")
    print(f"log+svm: minDCF={minDCF_log_svm}, actualDCF={actualDCF_log_svm}")
    print(f"log+gmm: minDCF={minDCF_log_gmm}, actualDCF={actualDCF_log_gmm}")
    print(f"svm+gmm: minDCF={minDCF_svm_gmm}, actualDCF={actualDCF_svm_gmm}")
    print(f"log+svm+gmm: minDCF={minDCF_log_svm_gmm}, actualDCF={actualDCF_log_svm_gmm}")
    
    K=10

    logRatioCumulative = np.array([])
    cumulativeLabels = np.array([])

    for i in range(0, K):
        (DTR, LTR), (DTE, LTE) = k_fold(vrow(S_log_svm), L, K, i)

        # MVG
        # compute mean and covariance for all classes
        (mu0, C0) = compute_mu_C(DTR, LTR, 0, False)
        (mu1, C1) = compute_mu_C(DTR, LTR, 1, False)

        # compute score matrix S of shape [2, x], which is the number of classes times the number of samples in the test set
        S0 = logpdf_GAU_ND(DTE, mu0, C0)
        S1 = logpdf_GAU_ND(DTE, mu1, C1)

        S_log_svm_c = S1 - S0

        logRatioCumulative = np.append(logRatioCumulative, S_log_svm_c)
        cumulativeLabels = np.append(cumulativeLabels, LTE)

    minDCF_log_svm_c = DCF_min(1/11, 1, 1, logRatioCumulative, cumulativeLabels)
    actualDCF_log_svm_c = DCF_actual(1/11, 1, 1, logRatioCumulative, cumulativeLabels)

    print(f"log+svm calibrated: minDCF={minDCF_log_svm_c}, actualDCF={actualDCF_log_svm_c}")

    logRatioCumulative = np.array([])
    cumulativeLabels = np.array([])

    for i in range(0, K):
        (DTR, LTR), (DTE, LTE) = k_fold(vrow(S_log_gmm), L, K, i)

        # MVG
        # compute mean and covariance for all classes
        (mu0, C0) = compute_mu_C(DTR, LTR, 0, False)
        (mu1, C1) = compute_mu_C(DTR, LTR, 1, False)

        # compute score matrix S of shape [2, x], which is the number of classes times the number of samples in the test set
        S0 = logpdf_GAU_ND(DTE, mu0, C0)
        S1 = logpdf_GAU_ND(DTE, mu1, C1)

        S_log_gmm_c = S1 - S0

        logRatioCumulative = np.append(logRatioCumulative, S_log_gmm_c)
        cumulativeLabels = np.append(cumulativeLabels, LTE)

    minDCF_log_gmm_c = DCF_min(1/11, 1, 1, logRatioCumulative, cumulativeLabels)
    actualDCF_log_gmm_c = DCF_actual(1/11, 1, 1, logRatioCumulative, cumulativeLabels)

    print(f"log+gmm calibrated: minDCF={minDCF_log_gmm_c}, actualDCF={actualDCF_log_gmm_c}")

    logRatioCumulative = np.array([])
    cumulativeLabels = np.array([])

    for i in range(0, K):
        (DTR, LTR), (DTE, LTE) = k_fold(vrow(S_svm_gmm), L, K, i)

        # MVG
        # compute mean and covariance for all classes
        (mu0, C0) = compute_mu_C(DTR, LTR, 0, False)
        (mu1, C1) = compute_mu_C(DTR, LTR, 1, False)

        # compute score matrix S of shape [2, x], which is the number of classes times the number of samples in the test set
        S0 = logpdf_GAU_ND(DTE, mu0, C0)
        S1 = logpdf_GAU_ND(DTE, mu1, C1)

        S_svm_gmm_c = S1 - S0

        logRatioCumulative = np.append(logRatioCumulative, S_svm_gmm_c)
        cumulativeLabels = np.append(cumulativeLabels, LTE)

    minDCF_svm_gmm_c = DCF_min(1/11, 1, 1, logRatioCumulative, cumulativeLabels)
    actualDCF_svm_gmm_c = DCF_actual(1/11, 1, 1, logRatioCumulative, cumulativeLabels)

    print(f"svm+gmm calibrated: minDCF={minDCF_svm_gmm_c}, actualDCF={actualDCF_svm_gmm_c}")

    logRatioCumulative = np.array([])
    cumulativeLabels = np.array([])

    for i in range(0, K):
        (DTR, LTR), (DTE, LTE) = k_fold(vrow(S_log_svm_gmm), L, K, i)

        # MVG
        # compute mean and covariance for all classes
        (mu0, C0) = compute_mu_C(DTR, LTR, 0, False)
        (mu1, C1) = compute_mu_C(DTR, LTR, 1, False)

        # compute score matrix S of shape [2, x], which is the number of classes times the number of samples in the test set
        S0 = logpdf_GAU_ND(DTE, mu0, C0)
        S1 = logpdf_GAU_ND(DTE, mu1, C1)

        S_log_svm_gmm_c = S1 - S0

        logRatioCumulative = np.append(logRatioCumulative, S_log_svm_gmm_c)
        cumulativeLabels = np.append(cumulativeLabels, LTE)

    minDCF_log_svm_gmm_c = DCF_min(1/11, 1, 1, logRatioCumulative, cumulativeLabels)
    actualDCF_log_svm_gmm_c = DCF_actual(1/11, 1, 1, logRatioCumulative, cumulativeLabels)

    print(f"svm+gmm calibrated: minDCF={minDCF_log_svm_gmm_c}, actualDCF={actualDCF_log_svm_gmm_c}")
    
