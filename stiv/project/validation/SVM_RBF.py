from lib import *
k=10

def SVMEvaluationWrap(DTR, LTR):
    data = [np.load("../data/pca/pca_11.npy"), DTR]
    PIs = [0.1, 0.5, 0.9]
    log = ""
    for d in data:
        for pi in PIs:
            s = SVMRBFEvaluation(d, LTR, pi)
            log += s + "\n"

    txt=open("SVM_RBF_MINDCFS.txt", "w")
    txt.write(log) 
    txt.close()


    

def SVMRBFEvaluation(x, labels, pi):
    Cs = [1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4]
    Ys = [0.1, 0.01, 0.001]
    SVM_S_cumulative = np.array([])
    labels_cumulative = np.array([])

    num_samples = x.shape[1]
    indices = np.random.permutation(num_samples)
    fold_size = num_samples//k

    s=""

    for Y in Ys:
        for C in Cs:
            s += "[ "
            print("[ ", end="", flush=True)
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
            print(f"{min_DCF}, ", end="", flush=True)

        s += "\n"
        print()

    s += "\n"
    print()

    

    return s

if __name__ == '__main__':
    DTR = np.load("../data/DTR.npy")
    LTR = np.load("../data/LTR.npy")
    DTE = np.load("../data/DTE.npy")
    LTE = np.load("../data/LTE.npy")

    SVMEvaluationWrap(DTR, LTR)