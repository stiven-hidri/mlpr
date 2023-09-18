from lib import * 

if __name__ == '__main__':
    DTR = np.load("../data/DTR.npy")
    LTR = np.load("../data/LTR.npy")
    DTE = np.load("../data/DTE.npy")
    LTE = np.load("../data/LTE.npy")

    # #GMM FULL
    # DTR0=DTR[:,LTR==0]                                  
    # gmm_class0=GMM_LBG(DTR0, 2, "full")  
    # _, SM0=logpdf_GMM(DTE, gmm_class0)                    
    
    # DTR1=DTR[:,LTR==1]                                  
    # gmm_class1= GMM_LBG(DTR1, 2, "full")
    # _, SM1=logpdf_GMM(DTE,gmm_class1)
    # llr_gmm_full = SM1 - SM0
    # np.save("llr_gmm_full", llr_gmm_full)

    # #MVG TIED
    # (mu0, mu1), (C0, C1), (CsDiag0, CsDiag1), wC = calculateParametersMVG(DTR, LTR)
    # s0 = logpdf_GAU_ND(DTE, mu0, wC)
    # s1 = logpdf_GAU_ND(DTE, mu1, wC)
    # llr_mvg_tied = s1-s0
    # np.save("llr_mvg_tied", llr_mvg_tied)

    # #LOGISTIC REGRESSION
    # logreg_obj = logreg_obj_wrap(DTR, LTR, 1e-4)
    # xmin, f, d  = bfgs(logreg_obj, np.zeros(DTR.shape[0] + 1), approx_grad=True, factr=1e10, maxiter=100)
    # w, b = xmin[0:-1], xmin[-1]
    # llr_lr = np.dot(w, DTE) + b
    # np.save("llr_lr", llr_lr)

    # #SVM POLY KERNEL
    # llr_svm_poly = compute_svm_polykernel(DTR, LTR, DTE, 1, 1e-4, 2, 1)
    # np.save("llr_svm_poly", llr_svm_poly)
    llr_gmm_full = np.load("./llr_gmm_full.npy")
    llr_mvg_tied = np.load("./llr_mvg_tied.npy")
    llr_lr = np.load("./llr_lr.npy")
    #llr_svm_poly = np.load("./llr_svm_poly.npy")

    effPriorLogOdds = np.linspace(-4, 4, 25)

    minDCFs_gmm_full, DCFs_gmm_full = bayesErrorPlots(effPriorLogOdds, llr_gmm_full, LTE)
    minDCFs_mvg_tied, DCFs_mvg_tied = bayesErrorPlots(effPriorLogOdds, llr_mvg_tied, LTE)
    minDCFs_lr, DCFs_lr = bayesErrorPlots(effPriorLogOdds, llr_lr, LTE)
    #minDCFs_svm_poly, DCFs_svm_poly = bayesErrorPlots(effPriorLogOdds, llr_svm_poly, LTE)

    plt.figure()
    plt.plot(effPriorLogOdds, minDCFs_gmm_full, label='min DCF GMM Full', color='red', linestyle="dashed")
    plt.plot(effPriorLogOdds, DCFs_gmm_full, label='actual DCF GMM Full', color='red')
    plt.plot(effPriorLogOdds, minDCFs_mvg_tied, label='min DCF MVG Tied', color='green', linestyle="dashed")
    plt.plot(effPriorLogOdds, DCFs_mvg_tied, label='actual DCF MVG Tied', color='green')
    plt.plot(effPriorLogOdds, minDCFs_lr, label='min DCF Log. Reg.', color='blue', linestyle="dashed")
    plt.plot(effPriorLogOdds, DCFs_lr, label='actual DCF Log. Reg.', color='blue')
    #plt.plot(effPriorLogOdds, minDCFs_svm_poly, label='min DCF SVM KERNEL Poly', color='black', linestyle="dashed")
    #plt.plot(effPriorLogOdds, DCFs_svm_poly, label='actual DCF SVM Kernel Poly', color='black')
    plt.legend()
    plt.title("Model comparisons")
    plt.xlim([-4, 4])
    # plt.ylim([0, 1.1])
    plt.savefig("ModelComparison")
