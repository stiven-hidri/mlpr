from lib import *;
k=10

def featureAnalisys(DTR, LTR):
    statistics(DTR, LTR)
    lda(DTR, LTR, 1, 1)
    explainedVariance(DTR)
    heatmaps_binary(DTR,LTR)

def MVGEvaluationWrap(DTR, LTR):
    data = [np.load("data/pca/pca_8.npy"), np.load("data/pca/pca_9.npy"), np.load("data/pca/pca_10.npy"), np.load("data/pca/pca_11.npy"), DTR]
    app = [(0.5, 1, 1), (0.1, 1, 1), (0.9, 1, 1)]
    for d in data:
        for (p, Cfn, Cfp) in app:
            MVGEvaluation(d, LTR, p, Cfn, Cfp)

def MVGEvaluation(x, labels, p, Cfn, Cfp):
    num_samples = x.shape[1]
    indices = np.random.permutation(num_samples)
    fold_size = num_samples // k

    mvg_llr_cumulative = np.array([])
    mvg_llr_nb_cumulative = np.array([])
    mvg_llr_tied_cumulative = np.array([])
    labels_cumulative = np.array([])

    for i in range(k):
        fold_start = i * fold_size
        fold_end = (i + 1) * fold_size
        val_indices = indices[fold_start:fold_end]
        train_indices = np.concatenate([indices[:fold_start], indices[fold_end:]])
        
        x_train, labels_train = x[:,train_indices], labels[train_indices]
        x_val, labels_val = x[:,val_indices], labels[val_indices]

        #MVG
        (mu0, mu1), (C0, C1), (CsDiag0, CsDiag1), wC = calculateParametersMVG(x_train, labels_train)

        s0 = logpdf_GAU_ND(x_val, mu0, C0)
        s1 = logpdf_GAU_ND(x_val, mu1, C1)
        mvg_llr = s1-s0

        s0 = logpdf_GAU_ND(x_val, mu0, CsDiag0)
        s1 = logpdf_GAU_ND(x_val, mu1, CsDiag1)
        mvg_llr_nb = s1-s0

        s0 = logpdf_GAU_ND(x_val, mu0, wC)
        s1 = logpdf_GAU_ND(x_val, mu1, wC)
        mvg_llr_tied = s1-s0

        mvg_llr_cumulative = np.append(mvg_llr_cumulative, mvg_llr)
        mvg_llr_nb_cumulative = np.append(mvg_llr_nb_cumulative, mvg_llr_nb)
        mvg_llr_tied_cumulative = np.append(mvg_llr_tied_cumulative, mvg_llr_tied)
        labels_cumulative = np.append(labels_cumulative, labels_val)

    min_DCF = computeMinDCF(p, Cfp, Cfn, mvg_llr_cumulative, labels_cumulative)
    print(f"min_dcf mvg: {round(min_DCF, 3)} p: {p}")
    min_DCF = computeMinDCF(p, Cfp, Cfn, mvg_llr_nb_cumulative, labels_cumulative)
    print(f"min_dcf nb: {round(min_DCF, 3)} p: {p}")
    min_DCF = computeMinDCF(p, Cfp, Cfn, mvg_llr_tied_cumulative, labels_cumulative)
    print(f"min_dcf tied: {round(min_DCF, 3)} p: {p}")
    
def LREEvaluationWrap(DTR, LTR):
    # data = [DTR, np.load("data/pca/pca_11.npy"), np.load("data/pca/pca_10.npy"), np.load("data/pca/pca_9.npy"), np.load("data/pca/pca_8.npy")]
    app = [(0.1, 1, 1), (0.5, 1, 1), (0.9, 1, 1)]
    
    # min_DCFs = np.array([])
    # lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]

    # for i, d in enumerate(data):
    #     min_DCFs = []
    #     for (p, Cfn, Cfp) in app:
    #         res = LREvaluation(d, LTR, p, Cfn, Cfp, 12-i)
    #         min_DCFs.append(res)

    data = [DTR, np.load("data/pca/pca_11.npy")]

    for i, d in enumerate(data):
        Dc = centerData(d)
        Ds = std_variances(Dc)
        Dw = whitening(Ds, d)
        data[i] = Dw

    for i, d in enumerate(data):
        for (p, Cfn, Cfp) in app:
            LREEvaluation(d, LTR, p, Cfn, Cfp, 12-i)

def LREEvaluation(x, labels, p, Cfn, Cfp, msg):
    lambdas = [1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100, 1000]
    num_samples = x.shape[1]
    indices = np.random.permutation(num_samples)
    fold_size = num_samples // k
    min_DCFs = np.array([], dtype=float)

    LR_llr_cumulative = np.array([])
    labels_cumulative = np.array([])

    for li in lambdas:
        for i in range(k):
            fold_start = i * fold_size
            fold_end = (i + 1) * fold_size
            val_indices = indices[fold_start:fold_end]
            train_indices = np.concatenate([indices[:fold_start], indices[fold_end:]])
            
            x_train, labels_train = x[:,train_indices], labels[train_indices]
            x_val, labels_val = x[:,val_indices], labels[val_indices]

            #MVG
            logreg_obj = logreg_obj_wrap(x_train, labels_train, li)
            xmin, f, d  = bfgs(logreg_obj, np.zeros(x_train.shape[0] + 1), approx_grad=True, factr=1e10, maxiter=100)
            w, b = xmin[0:-1], xmin[-1]
            S = np.dot(w, x_val) + b

            LR_llr_cumulative = np.append(LR_llr_cumulative, S)
            labels_cumulative = np.append(labels_cumulative, labels_val)

        min_DCF = computeMinDCF(p, Cfp, Cfn, LR_llr_cumulative, labels_cumulative)
        min_DCFs = np.append(min_DCFs, min_DCF)
        print(f"min_dcf LR: {min_DCF} p: {p} lambda: {li} dim: {msg}")

    return min_DCFs

def SVMEvaluationWrap(DTR, LTR):
    data = [np.load("data/pca/pca_11.npy"), DTR]
    app = [(0.1, 1, 1), (0.5, 1, 1), (0.9, 1, 1)]
    
    min_DCFs = np.array([])

    for d in data:
        min_DCFs = []
        for (p, Cfn, Cfp) in app:
            res = SVMEvaluation(d, LTR, p, Cfn, Cfp)
            min_DCFs.append(res)

def SVMEvaluation(x, labels, p, Cfn, Cfp):
    num_samples = x.shape[1]
    C = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5]
    indices = np.random.permutation(num_samples)
    fold_size = num_samples // k
    min_DCFs = np.array([], dtype=float)

    SVM_S_cumulative = np.array([])
    labels_cumulative = np.array([])
    
    print("[ ", end="", flush=True)

    for c in C:        
        for i in range(k):
            fold_start = i * fold_size
            fold_end = (i + 1) * fold_size
            val_indices = indices[fold_start:fold_end]
            train_indices = np.concatenate([indices[:fold_start], indices[fold_end:]])
            
            x_train, labels_train = x[:,train_indices], labels[train_indices]
            x_val, labels_val = x[:,val_indices], labels[val_indices]

            S = compute_svm(x_train, labels_train, x_val, 1, c)

            SVM_S_cumulative = np.append(SVM_S_cumulative, S)
            labels_cumulative = np.append(labels_cumulative, labels_val)

        min_DCF = computeMinDCF(p, Cfp, Cfn, SVM_S_cumulative, labels_cumulative)
        min_DCFs = np.append(min_DCFs, min_DCF)
        print(f"{min_DCF}, ", end="", flush=True)

    print()

    return min_DCFs

def main():
    (DTR, LTR), (DTE, LTE) = readTrainAndTestData()
    #featureAnalisys(DTR, LTR) DONE
    #explainedVariance(DTR)
    #MVGEvaluationWrap(DTR, LTR)
    #LREEvaluationWrap(DTR, LTR)
    SVMEvaluationWrap(DTR, LTR)