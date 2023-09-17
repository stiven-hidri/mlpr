from lib import *

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

if __name__ == '__main__':
    DTR = np.load("../data/DTR.npy")
    LTR = np.load("../data/LTR.npy")
    LREEvaluationWrap(DTR, LTR)