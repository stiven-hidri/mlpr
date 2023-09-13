from lib import *
from sklearn import datasets

def load_iris():
	D, L = datasets.load_iris()['data'].T, datasets.load_iris()['target']
	return D, L

def split_db_2to1(D, L, seed=0):
    nTrain = int(D.shape[1]*2.0/3.0)
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTrain = idx[0:nTrain]
    idxTest = idx[nTrain:]
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]
    return (DTR, LTR), (DTE, LTE)

def svm_wrapper(H, DTR):
    def svm_obj(alpha):
        LD = 0.5 * np.dot(alpha.T, np.dot(H, alpha)) - np.dot(alpha.T, np.ones((DTR.shape[1], 1)))
        LDG = np.reshape(np.dot(H, alpha) - np.ones((1, DTR.shape[1])), (DTR.shape[1],1))
        return (LD, LDG)
    return svm_obj

def compute_svm(DTR, LTR, DTE, K, C):
    Z = LTR * 2 - 1
    DTRE = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])
    D = np.multiply(DTRE, Z.T)
    H = np.dot(D.T, D)

    # define the array of constraints for the objective
    BC = [(0, C) for i in range(0, DTR.shape[1])]
    [alpha, LD, d] = sp.optimize.fmin_l_bfgs_b(svm_wrapper(H, DTR), np.zeros((DTR.shape[1],1)), bounds=BC, factr=0.5)
    
    # need to compute the primal solution from the dual solution
    w = np.multiply(alpha, np.multiply(DTRE, Z.T)).sum(axis=1)

    # need to compute the duality gap
    S = -np.dot(w.T, D) + 1
    JP = 0.5 * (np.linalg.norm(w) ** 2) + C * (S[S>0]).sum()
    
    # we now need to compute the scores and check the predicted lables with threshold
    DTEE = np.vstack([DTE, np.ones((1, DTE.shape[1])) * K])
    return np.dot(w.T, DTEE)
    
def compute_svm_polykernel(DTR, LTR, DTE, K, C, d, c):
    Z = LTR * 2 - 1
    DTRE = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])
    
    Z = np.reshape(Z, (LTR.shape[0], 1))

    Kprime = np.dot(DTRE.T, DTRE)
    Zprime = np.dot(Z, Z.T)
    Kmat = ((Kprime + c) ** d) + K**2
    H = np.multiply(Zprime, Kmat)

    BC = [(0, C) for i in range(0, DTR.shape[1])]
    [alpha, f, d2] = sp.optimize.fmin_l_bfgs_b(svm_wrapper(H, DTR), np.zeros((DTR.shape[1],1)), bounds=BC, factr=0.5)
    
    DTEE = np.vstack([DTE, np.ones((1, DTE.shape[1])) * K])

    S = np.ones((DTE.shape[1]))

    alpha = np.reshape(alpha, (alpha.shape[0], 1))
    az = np.multiply(alpha, Z)
    Kprime = np.dot(DTRE.T, DTEE)
    Kmat = ((Kprime + c) ** d) + K**2
    S = np.multiply(az, Kmat).sum(axis=0)
    
    return S

def compute_svm_RBF(DTR, LTR, DTE, K, C, g):
    Z = LTR * 2 - 1
    DTRE = np.vstack([DTR, np.ones((1, DTR.shape[1])) * K])
    
    Z = np.reshape(Z, (LTR.shape[0], 1))
    H = np.dot(Z, Z.T)

    # will compute H in with for loops
    for i in range(0, DTR.shape[1]):
        for j in range(0, DTR.shape[1]):
            H[i][j] *= (np.exp(-g*(np.linalg.norm(DTRE.T[i] - DTRE.T[j]))**2) + K**2)

    BC = [(0, C) for i in range(0, DTR.shape[1])]
    [alpha, f, d2] = sp.optimize.fmin_l_bfgs_b(svm_wrapper(H, DTR), np.zeros((DTR.shape[1],1)), bounds=BC, factr=0.5)
    
    DTEE = np.vstack([DTE, np.ones((1, DTE.shape[1])) * K])

    S = np.ones((DTE.shape[1]))
    
    for t in range(0, DTE.shape[1]):
        result = 0
        for i in range(0, DTR.shape[1]):
            result += (alpha[i]*Z[i]*(np.exp(-g*(np.linalg.norm(DTRE.T[i] - DTEE.T[t]))**2) + K**2))[0]

        S[t] = result
        
    return S

if __name__ == '__main__':
    DTR, LTR = load_iris();
    (DTR, LTR), (DTE, LTE) = split_db_2to1(DTR, LTR)
    
    K = [0, 1]
    C = [0.1, 1, 10]
    for k in K:
        for c in C:
            res = compute_svm(DTR, LTR, DTE, k, c)
            pcl = res>0
            acc = (pcl == LTE).sum()/pcl.size
            err = 1-acc
            print(f"{k}\t{c}\t{round(err*100, 2)}%")

    print()
    K = [0, 1]
    C = 1
    p = [(2,0), (2,1)]

    for (d, c) in p:
        for k in K:
            res= compute_svm_polykernel(DTR, LTR, DTE, k, C, d, c)
            pcl = res>0
            acc = (pcl == LTE).sum()/pcl.size
            err = 1-acc
            print(f"{k}\t{C}\t{d}\t{c}\t{round(err*100, 2)}%")

    K = [0, 1]
    C = 1
    lambdas = [1, 10]

    print()

    for k in K:
        for l in lambdas:
            res= compute_svm_RBF(DTR, LTR, DTE, k, C, l)
            pcl = res>0
            acc = (pcl == LTE).sum()/pcl.size
            err = 1-acc
            print(f"{k}\t{C}\t{l}\t{round(err*100, 2)}%")

    print()

    res = compute_svm(DTR, LTR, DTE, 1, 0.1)
    pcl = res>0
    cnt = (pcl == LTE).sum()
    print(cnt)

    res = compute_svm_polykernel(DTR, LTR, DTE, 0, 1, 2, 0)
    pcl = res>0
    cnt = (pcl == LTE).sum()
    print(cnt)

    res = compute_svm_RBF(DTR, LTR, DTE, 0, 1, 1)
    pcl = res>0
    cnt = (pcl == LTE).sum()
    print(cnt)


