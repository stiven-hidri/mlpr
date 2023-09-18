from lib import * 

if __name__ == '__main__':
    DTR_raw = np.load("../data/DTR.npy")
    DTR_pca11 = np.load("../data/pca/pca_11.npy")

    Dc = centerData(DTR_raw)
    Ds = std_variances(Dc)
    DTR_raw_pre = whitening(Ds, DTR_raw)

    LTR = np.load("../data/LTR.npy")


    DTE = np.load("../data/DTE.npy")
    DTE_PCA = pca(DTE, 11)
    LTE = np.load("../data/LTE.npy")

    """ #GMM FULL
    DTR0=DTR_pca11[:,LTR==0]                                  
    gmm_class0=GMM_LBG(DTR0, 2, "full")  
    _, SM0=logpdf_GMM(DTE_PCA, gmm_class0)                    
    
    DTR1=DTR_pca11[:,LTR==1]                                  
    gmm_class1= GMM_LBG(DTR1, 2, "full")
    _, SM1=logpdf_GMM(DTE_PCA,gmm_class1)
    llr_gmm_full = SM1 - SM0
    np.save("llr_gmm_full", llr_gmm_full)

    #MVG TIED
    (mu0, mu1), (C0, C1), (CsDiag0, CsDiag1), wC = calculateParametersMVG(DTR_raw, LTR)
    s0 = logpdf_GAU_ND(DTE, mu0, wC)
    s1 = logpdf_GAU_ND(DTE, mu1, wC)
    llr_mvg_tied = s1-s0
    np.save("llr_mvg_tied", llr_mvg_tied)

    #LOGISTIC REGRESSION
    logreg_obj = logreg_obj_wrap(DTR_raw_pre, LTR, 1e-4)
    xmin, f, d  = bfgs(logreg_obj, np.zeros(DTR_raw_pre.shape[0] + 1), approx_grad=True, factr=1e10, maxiter=100)
    w, b = xmin[0:-1], xmin[-1]
    llr_lr = np.dot(w, DTE) + b
    np.save("llr_lr", llr_lr)

    #SVM POLY KERNEL
    llr_svm_poly = compute_svm_polykernel(DTR_raw, LTR, DTE, 1, 1e-4, 2, 1)
    np.save("llr_svm_poly", llr_svm_poly) """
    
    llr_gmm_full = np.load("./llr_gmm_full.npy")
    llr_mvg_tied = np.load("./llr_mvg_tied.npy")
    llr_lr = np.load("./llr_lr.npy")
    llr_svm_poly = np.load("./llr_svm_poly.npy")

    llr_gmm_full = np.load("./llr_gmm_full.npy")
    
    LLRs = [llr_gmm_full, llr_mvg_tied, llr_lr, llr_svm_poly]
    PIs = [0.1, 0.5, 0.9]

    for llr in LLRs:
        for pi in PIs:
            minDCF = computeMinDCF(pi, 1, 1, llr, LTE)
            print(f"{minDCF} ", end="", flush=True)
        print()
