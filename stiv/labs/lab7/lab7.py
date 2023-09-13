from scipy.optimize import fmin_l_bfgs_b as bfgs
import sklearn.datasets
import numpy as np
import ml

def load_iris_binary():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    D = D[:, L != 0] # We remove setosa from D
    L = L[L!=0] # We remove setosa from L
    L[L==2] = 0 # We assign label 0 to virginica (was label 2)
    return D, L

def logreg_obj_wrap(DTR, LTR, λ):
    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        N = DTR.shape[1]
        J = 0.5*λ*np.linalg.norm(w)**2
        sommatoria = 0
        for i in range(N):
            xi = DTR[:,i]
            ci = LTR[i]
            zi = 2*ci-1

            sommatoria+=np.logaddexp(0, -zi*(np.dot(w.T, xi) + b))

        J+=sommatoria/N
        return J

    return logreg_obj

def main():
    D, L = load_iris_binary()
    (DTR, LTR), (DTE, LTE) = ml.split_db_2to1(D, L)
    logreg_obj = logreg_obj_wrap(DTR, LTR, 1e-3)
    x0 = np.zeros(DTR.shape[0] + 1)
    x, f, d  = bfgs(logreg_obj, x0, approx_grad=True)

    w, b = x[0:-1], x[-1]
    S = np.dot(w, DTE) + b

    LP = S > 0

    err = 1-(LP == LTE).mean()

    print(str(round(err*100,2)) + "%")

if __name__ == '__main__':
  main()