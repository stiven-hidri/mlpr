from lib import *

k=10
v='full'

def comparison(x_raw, x_raw_pre, x_pca11, labels):
    print("start...")
    """ GMM_S_cumulative = np.array([])
    mvg_llr_tied_cumulative = np.array([])
    LR_llr_cumulative = np.array([])
    SVM_S_cumulative = np.array([])
    labels_cumulative = np.array([])

    num_samples = x_raw.shape[1]
    indices = np.random.permutation(num_samples)
    fold_size = num_samples//k


    for i in range(k):
        fold_start = i * fold_size
        fold_end = (i + 1) * fold_size
        val_indices = indices[fold_start:fold_end]
        train_indices = np.concatenate([indices[:fold_start], indices[fold_end:]])
        
        labels_train = labels[train_indices]
        labels_val = labels[val_indices]
        #RAW
        x_train_raw = x_raw[:,train_indices]
        x_val_raw = x_raw[:,val_indices]
        #RAW_pre
        x_train_raw_pre = x_raw_pre[:,train_indices]
        x_val_raw_pre = x_raw_pre[:,val_indices]
        #pca_11
        x_train_pca11 = x_pca11[:,train_indices]
        x_val_pca11 = x_pca11[:,val_indices]
        
        #GMM
        DTR0=x_train_pca11[:,labels_train==0]                                  
        gmm_class0=GMM_LBG(DTR0, 2, v)  
        _, SM0=logpdf_GMM(x_val_pca11,gmm_class0)                    
        DTR1=x_train_pca11[:,labels_train==1]                                  
        gmm_class1= GMM_LBG(DTR1, 2, v)
        _, SM1=logpdf_GMM(x_val_pca11,gmm_class1)
        S = SM1 - SM0 
        GMM_S_cumulative = np.append(GMM_S_cumulative, S)

        #svm poly
        S = compute_svm_polykernel(x_train_raw, labels_train, x_val_raw, 1, 1e-4, 2, 1)
        SVM_S_cumulative = np.append(SVM_S_cumulative, S)

        #lr
        logreg_obj = logreg_obj_wrap(x_train_raw_pre, labels_train, 1e-4)
        xmin, f, d  = bfgs(logreg_obj, np.zeros(x_train_raw_pre.shape[0] + 1), approx_grad=True, factr=1e10, maxiter=100)
        w, b = xmin[0:-1], xmin[-1]
        S = np.dot(w, x_val_raw_pre) + b
        LR_llr_cumulative = np.append(LR_llr_cumulative, S)

        #mvg tied
        (mu0, mu1), (C0, C1), (CsDiag0, CsDiag1), wC = calculateParametersMVG(x_train_raw, labels_train)
        s0 = logpdf_GAU_ND(x_val_raw, mu0, wC)
        s1 = logpdf_GAU_ND(x_val_raw, mu1, wC)
        mvg_llr_tied = s1-s0
        mvg_llr_tied_cumulative = np.append(mvg_llr_tied_cumulative, mvg_llr_tied)

        labels_cumulative = np.append(labels_cumulative, labels_val)

        print("done k...")

    np.save("gmm_llr", GMM_S_cumulative)
    np.save("svm_llr", SVM_S_cumulative)
    np.save("lr_llr", LR_llr_cumulative)
    np.save("mvgtied_llr", mvg_llr_tied_cumulative)
    np.save("labels", labels_cumulative) """

    GMM_S_cumulative = np.load("gmm_llr.npy")
    SVM_S_cumulative = np.load("svm_llr.npy")
    LR_llr_cumulative = np.load("lr_llr.npy")
    mvg_llr_tied_cumulative = np.load("mvgtied_llr.npy")
    labels_cumulative = np.load("labels.npy")

    print("Done training...")

    effPriorLogOdds = np.linspace(-4, 4, 25)

    minDCFs_gmm_full, DCFs_gmm_full = bayesErrorPlots(effPriorLogOdds, GMM_S_cumulative, labels_cumulative)
    print("done gmm dcfs...")
    minDCFs_mvg_tied, DCFs_mvg_tied = bayesErrorPlots(effPriorLogOdds, mvg_llr_tied_cumulative, labels_cumulative)
    print("done mvg dcfs...")
    minDCFs_lr, DCFs_lr = bayesErrorPlots(effPriorLogOdds, LR_llr_cumulative, labels_cumulative)
    print("done lr dcfs...")
    minDCFs_svm_poly, DCFs_svm_poly = bayesErrorPlots(effPriorLogOdds, SVM_S_cumulative, labels_cumulative)
    print("done svm dcfs...")

    np.save("minDCFs_gmm_full", minDCFs_gmm_full)
    np.save("DCFs_gmm_full", DCFs_gmm_full)
    np.save("minDCFs_mvg_tied", minDCFs_mvg_tied)
    np.save("DCFs_mvg_tied", DCFs_mvg_tied)
    np.save("minDCFs_lr", minDCFs_lr)
    np.save("DCFs_lr", DCFs_lr)
    np.save("minDCFs_svm_poly", minDCFs_svm_poly)
    np.save("DCFs_svm_poly", DCFs_svm_poly)

    print("done dcfs...")

    plt.figure()
    plt.plot(effPriorLogOdds, minDCFs_gmm_full, label='min DCF GMM Full', color='red', linestyle="dashed")
    plt.plot(effPriorLogOdds, DCFs_gmm_full, label='actual DCF GMM Full', color='red')
    plt.plot(effPriorLogOdds, minDCFs_mvg_tied, label='min DCF MVG Tied', color='green', linestyle="dashed")
    plt.plot(effPriorLogOdds, DCFs_mvg_tied, label='actual DCF MVG Tied', color='green')
    plt.plot(effPriorLogOdds, minDCFs_lr, label='min DCF Log. Reg.', color='blue', linestyle="dashed")
    plt.plot(effPriorLogOdds, DCFs_lr, label='actual DCF Log. Reg.', color='blue')
    plt.plot(effPriorLogOdds, minDCFs_svm_poly, label='min DCF SVM KERNEL Poly', color='black', linestyle="dashed")
    plt.plot(effPriorLogOdds, DCFs_svm_poly, label='actual DCF SVM Kernel Poly', color='black')
    plt.legend()
    plt.title("Models comparison")
    plt.xlim([-4, 4])
    # plt.ylim([0, 1.1])
    plt.savefig("ModelComparison2")

    print("done plotting...")

if __name__ == '__main__':
    DTR_raw = np.load("../data/DTR.npy")
    DTR_pca11 = np.load("../data/pca/pca_11.npy")

    Dc = centerData(DTR_raw)
    Ds = std_variances(Dc)
    DTR_raw_pre = whitening(Ds, DTR_raw)

    LTR = np.load("../data/LTR.npy")
    comparison(DTR_raw, DTR_raw_pre, DTR_pca11, LTR)