import sklearn.datasets
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def vcol(mat):
    return mat.reshape((mat.size, 1)) 

def vrow(mat):
    return mat.reshape((1, mat.size))

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

def compute_mu_C(D, L, label):
    DL = D[:, L == label]
    mu = DL.mean(1).reshape(DL.shape[0], 1)
    DLC = (DL - mu)
    C = 1/DLC.shape[1]*np.dot(DLC, DLC.T)
    return (mu, C)

def compute_mu_C_NB(D, L, label):
    DL = D[:, L == label]
    mu = DL.mean(1).reshape(DL.shape[0], 1)
    DLC = (DL - mu)
    C = np.multiply(1/DLC.shape[1]*np.dot(DLC, DLC.T), np.identity(DL.shape[0]))
    return (mu, C)

def logpdf_GAU_ND(X, mu, C):
    # X array of shape(M, N)
    # mu array of shape (M, 1)
    # C array of shape (M, M) that represents the covariance matrix
    M = C.shape[0] #number of features
    # N = X.shape[1] #number of samples
    invC = np.linalg.inv(C) #C^-1
    logDetC = np.linalg.slogdet(C)[1] #log|C|
    
    # with the for loop:
    # logN = np.zeros(N)
    # for i, sample in enumerate(X.T):
    #     const = -0.5*M*np.log(2*np.pi)
    #     dot1 = np.dot((sample.reshape(M, 1) - mu).T, invC)
    #     dot2 = np.dot(dot1, sample.reshape(M, 1) - mu)
    #     logN[i] = const - 0.5*logDetC - 0.5*dot2

    XC = (X - mu).T # XC has shape (N, M)
    const = -0.5*M*np.log(2*np.pi)

    # sum(1) sum elements of the same row togheter
    # multiply make an element wise multiplication
    logN = const - 0.5*logDetC - 0.5*np.multiply(np.dot(XC, invC), XC).sum(1)

    # logN is an array of length N (# of samples)
    # each element represents the log-density of each sample
    return logN

if __name__ == '__main__':
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    # MVG
    # compute mean and covariance for all classes
    (mu0, C0) = compute_mu_C(DTR, LTR, 0)
    (mu1, C1) = compute_mu_C(DTR, LTR, 1)
    (mu2, C2) = compute_mu_C(DTR, LTR, 2)

    # Naive-Bayes
    # compute mean and covariance for all classes
    # (mu0, C0) = compute_mu_C_NB(DTR, LTR, 0)
    # (mu1, C1) = compute_mu_C_NB(DTR, LTR, 1)
    # (mu2, C2) = compute_mu_C_NB(DTR, LTR, 2)

    # Tied-Covariance
    C0 = C1 = C2 = 1/DTR.shape[1]*(C0*(LTR == 0).sum() + C1*(LTR == 1).sum() + C2*(LTR == 2).sum())

    # compute score matrix S of shape [3, 50], which is the number of classes times the number of samples in the test set
    S0 = logpdf_GAU_ND(DTE, mu0, C0)
    S1 = logpdf_GAU_ND(DTE, mu1, C1)
    S2 = logpdf_GAU_ND(DTE, mu2, C2)

    # f_c|x
    S = np.vstack([S0, S1, S2])
    
    # working with exp
    # S = np.exp(S)

    # # f_x|c
    # SJoint = 1/3*S
    # SMarginal = vrow(SJoint.sum(0))
    # SPost = SJoint/SMarginal

    # working with logs
    logSJoint = S + np.log(1/3)
    logSMarginal = vrow(sp.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = np.exp(logSPost)

    PL = np.argmax(SPost, 0)

    acc = (PL == LTE).sum()/len(PL)
    err = 1 - acc

    print(acc)
