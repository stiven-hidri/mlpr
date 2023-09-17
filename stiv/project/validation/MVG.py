from lib import *;

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
    


if __name__ == '__main__':
    DTR = np.load("../data/DTR.npy")
    LTR = np.load("../data/LTR.npy")
    MVGEvaluationWrap(DTR, LTR)