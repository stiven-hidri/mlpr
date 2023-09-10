from mllib import *
import sklearn.datasets

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
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

def k_fold(D, L, K, i, seed=0):
    np.random.seed(seed)
    idx = np.random.permutation(D.shape[1])
    idxTest = idx[int(i*D.shape[1]/K):int((i+1)*D.shape[1]/K)]
    idxTrain0 = idx[:int(i*D.shape[1]/K)]
    idxTrain1 = idx[int((i+1)*D.shape[1]/K):]
    idxTrain = np.hstack([idxTrain0, idxTrain1])
    DTR = D[:, idxTrain]
    DTE = D[:, idxTest]
    LTR = L[idxTrain]
    LTE = L[idxTest]

    return (DTR, LTR), (DTE, LTE)

if __name__ == '__main__':
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)
    