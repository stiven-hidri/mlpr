import numpy as np
import matplotlib.pyplot as plt

#np.load([path])

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

def loglikelihood(X, mu_ML, C_ML):
    
    logN = logpdf_GAU_ND(X, mu_ML, C_ML)

    # the log-likelihood corresponds to the sum of the log-density computed over all the samples
    ll = logN.sum()
    
    return ll

if __name__ == '__main__':
    # 2 dimensional samples with a given mu and a given covariance matrix
    mu = np.load('muND.npy')
    X = np.load('XND.npy')
    C = np.load('CND.npy')

    # one dimensional samples 
    X1D = np.load('X1D.npy')

    # computing the mean and the covariance matrix for the 2D dataset
    muX = X.mean(1).reshape(X.shape[0], 1)
    DC = X - muX
    CD = 1/X.shape[1]*np.dot(DC, DC.T)

    # computing the mean and the covariance matrix for the 1D dataset
    muX1D = X1D.mean(1).reshape(X1D.shape[0], 1)
    D1DC = X1D - muX1D
    CD1D = 1/X1D.shape[1]*np.dot(D1DC, D1DC.T)

    # computing the loglikelihood for the 2D dataset
    ll = loglikelihood(X, muX, CD)
    # print(ll)

    # ravel() transform a 2D array of shape (1, N) into a 1D vector (N, )
    # bins=x set how many columns the histogram have
    # density = true => the sum of the histograms is normalized to 1
    plt.figure()
    plt.hist(X1D.ravel(), bins=50, density=True)
    
    # generation of the 
    XPlot = np.linspace(-10, 10, 1000)
    plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(XPlot.reshape(1, len(XPlot)), muX1D, CD1D)))
    plt.show()

    ll = loglikelihood(X1D, muX1D, CD1D)
    print(ll)

    # testing the results
    # result = logpdf_GAU_ND(X, mu, C)
    # solution = np.load('llND.npy')