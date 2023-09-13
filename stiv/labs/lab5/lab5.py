import ml
import matplotlib as plt
import numpy as np
import scipy as sp

def calcStuff(DTR, LTR):
    l = np.unique(LTR)
    x0 = DTR[:,LTR[:]==l[0]]
    x1 = DTR[:,LTR[:]==l[1]]
    x2 = DTR[:,LTR[:]==l[2]]

    mu0 = x0.mean(1).reshape(4,1)
    mu1 = x1.mean(1).reshape(4,1)
    mu2 = x2.mean(1).reshape(4,1)

    xc0 = x0 - mu0
    xc1 = x1 - mu1
    xc2 = x2 - mu2

    N0 = np.shape(xc0)[1]
    N1 = np.shape(xc1)[1]
    N2 = np.shape(xc2)[1]

    C0 = 1/N0 * np.dot(xc0, xc0.T)
    C1 = 1/N1 * np.dot(xc1, xc1.T)
    C2 = 1/N2 * np.dot(xc2, xc2.T)

    wC = (C0*x0.shape[1] + C1*x1.shape[1] + C2*x2.shape[1])/DTR.shape[1]

    return (mu0, mu1, mu2), (C0, C1, C2), wC

def mvg(mu0, mu1, mu2, C0, C1, C2):
    l0 = ml.logpdf_GAU_ND(DTE, mu0, C0)
    l1 = ml.logpdf_GAU_ND(DTE, mu1, C1)
    l2 = ml.logpdf_GAU_ND(DTE, mu2, C2)

    S = np.stack((l0, l1, l2))
    Pc = 1/3;

    ## work with logarithms
    logSJoint = S + np.log(Pc)

    logSMarginal = ml.vrow(sp.special.logsumexp(S, axis=0))

    logSPost = logSJoint-logSMarginal
    SPost = np.exp(logSPost)

    pcl = np.argmax(SPost, 0);

    acc = (pcl == LTE).sum()/len(pcl)
    err = 1-acc
    print(str(err)+"%")

data, labels = ml.load_iris()

(DTR, LTR), (DTE, LTE) = ml.split_db_2to1(data, labels)

(mu0, mu1, mu2), (C0, C1, C2), wC = calcStuff(DTR, LTR)

print("mvg")
mvg(mu0, mu1, mu2, C0, C1, C2)

#
#   Naive bayes 
#

C0d = np.diag(np.diag(C0))
C1d = np.diag(np.diag(C1))
C2d = np.diag(np.diag(C2))

print("naive bayes")
mvg(mu0, mu1, mu2, C0d, C1d, C2d)

print(wC)
print("tied")
mvg(mu0, mu1, mu2, wC, wC, wC)