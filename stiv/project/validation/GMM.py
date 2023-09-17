from lib import *
from numba import njit

k=10
v='diagonal'
def GMMEvaluationWrap(DTR, LTR):
    data = [DTR, np.load("../data/pca/pca_11.npy")]
    PIs = [0.1, 0.5, 0.9]
    log = ""
    for d in data:
        for pi in PIs:
            s = GMMEvaluation(d, LTR, pi)
            log += s
        
        log += "\n"
        print()

    txt=open("GMM_RBF_MINDCFS_raw.txt", "w")
    txt.write(log) 
    txt.close()

def GMMEvaluation(x, labels, pi):
    doub = [0,1,2,3]
    GMM_S_cumulative = np.array([])
    labels_cumulative = np.array([])

    num_samples = x.shape[1]
    indices = np.random.permutation(num_samples)
    fold_size = num_samples//k

    s=""
    
    s += "[ "
    print("[ ", end="", flush=True)

    for d in doub:
        for i in range(k):
            fold_start = i * fold_size
            fold_end = (i + 1) * fold_size
            val_indices = indices[fold_start:fold_end]
            train_indices = np.concatenate([indices[:fold_start], indices[fold_end:]])
            
            x_train, labels_train = x[:,train_indices], labels[train_indices]
            x_val, labels_val = x[:,val_indices], labels[val_indices]

            DTR0=x_train[:,labels_train==0]                                  
            gmm_class0=GMM_LBG(DTR0, d, v)  
            _, SM0=logpdf_GMM(x_val,gmm_class0)                    
            
            # same for class 1
            DTR1=x_train[:,labels_train==1]                                  
            gmm_class1= GMM_LBG(DTR1, d, v)
            _, SM1=logpdf_GMM(x_val,gmm_class1)
            
            # compute scores
            S = SM1 - SM0 

            GMM_S_cumulative = np.append(GMM_S_cumulative, S)
            labels_cumulative = np.append(labels_cumulative, labels_val)

        min_DCF = computeMinDCF(pi, 1, 1, GMM_S_cumulative, labels_cumulative)
        
        s+=  f"{min_DCF}, "
        print(f"{min_DCF}, ", end="", flush=True)

    s+=  f"\n"
    print("")

    return s

if __name__ == '__main__':
    DTR = np.load("../data/DTR.npy")
    LTR = np.load("../data/LTR.npy")
    GMMEvaluationWrap(DTR, LTR)