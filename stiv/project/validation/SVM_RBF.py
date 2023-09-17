from lib import *
from numba import njit

k=10
def SVMEvaluationWrap(DTR, LTR):
    data = [np.load("../data/pca/pca_11.npy"), DTR]
    PIs = [0.5]
    log = ""
    for d in data:
        for pi in PIs:
            s = SVMRBFEvaluation(d, LTR, pi)
            log += s + "\n"

    txt=open("SVM_RBF_MINDCFS.txt", "w")
    txt.write(log) 
    txt.close()

def SVMRBFEvaluation(x, labels, pi):
    Cs = [1e-5, 1e5]
    Ys = [0.01,0.001]
    SVM_S_cumulative = np.array([])
    labels_cumulative = np.array([])

    num_samples = x.shape[1]
    indices = np.random.permutation(num_samples)
    fold_size = num_samples//k

    s=""

    for Y in Ys:
        s += "[ "
        print("[ ", end="", flush=True)
        for C in Cs:
            for i in range(k):
                fold_start = i * fold_size
                fold_end = (i + 1) * fold_size
                val_indices = indices[fold_start:fold_end]
                train_indices = np.concatenate([indices[:fold_start], indices[fold_end:]])
                
                x_train, labels_train = x[:,train_indices], labels[train_indices]
                x_val, labels_val = x[:,val_indices], labels[val_indices]

                S = compute_svm_RBF(x_train, labels_train, x_val, 1, C, Y)

                SVM_S_cumulative = np.append(SVM_S_cumulative, S)
                labels_cumulative = np.append(labels_cumulative, labels_val)

            min_DCF = computeMinDCF(pi, 1, 1, SVM_S_cumulative, labels_cumulative)
            
            s+=  f"{min_DCF}, "
            print(str(min_DCF)+", ", end="", flush=True)

        s += "\n"
        print()

    

    return s

if __name__ == '__main__':
    DTR = np.load("../data/DTR.npy")
    LTR = np.load("../data/LTR.npy")
    SVMEvaluationWrap(DTR, LTR)