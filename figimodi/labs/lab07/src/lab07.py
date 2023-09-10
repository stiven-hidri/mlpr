import numpy as np
import scipy as sp
import sklearn.datasets

def func(x):
    f = (x[0] + 3)**2 + np.sin(x[0]) + (x[1] + 1)**2
    dy = 2*(x[0] + 3) + np.cos(x[0])
    dz = 2*(x[1] + 1)
    return f, np.array([dy, dz])

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
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
    
def logreg_obj_wrap(DTR, LTR, l):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        J = 0

        for i in range(0, DTR.shape[1]):
            zi = 2*LTR[i] - 1
            J += np.logaddexp(0, -zi*(np.dot(w, DTR[:, i]) + b))

        J /= DTR.shape[1]
        J += l/2*np.linalg.norm(w)**2

        return J

    return logreg_obj

if __name__ == '__main__':
    # x, f, d =  sp.optimize.fmin_l_bfgs_b(func, [0, 0])
    # print(x)
    # print(f)
    # print(d)

    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    logreg_obj = logreg_obj_wrap(DTR, LTR, 10**-3)

    x0 = np.zeros(DTR.shape[0] + 1)
    x, f, d = sp.optimize.fmin_l_bfgs_b(logreg_obj, x0, approx_grad=True)

    w, b = x[0:-1], x[-1]
    S = np.dot(w, DTE) + b
    
    LP = S > 0

    acc = (LP == LTE).mean()*100
    err = (1 - acc/100)*100 

    print(acc)
    print(err)
