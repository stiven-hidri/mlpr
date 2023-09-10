import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import sys
import seaborn as sb
import pandas as pd

def vcol(mat):
    return mat.reshape((mat.size, 1)) 

def vrow(mat):
    return mat.reshape((1, mat.size))

def load(file_name):
    Dlist = []
    L = []

    with open(file_name) as f:
        for line in f:
            x = vcol(np.array([float(i) for i in line.split(',')[0:10]]))
            Dlist.append(x)
            l = int(line.split(',')[-1][0])
            L = np.hstack([L, l])

    return np.hstack(Dlist), L

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

def feature_plot_binary(feature, D, L, classes):
    vetAttr = D[feature, :]
    
    plt.figure()
    for classIndex, className in enumerate(classes):
        mask = (L == classIndex)
        data = vetAttr[mask]
        plt.hist(data, bins = 50, density=True, alpha=0.3, label=f"{className}")
    plt.savefig(f"01_features_analysis_pt1\\feature{feature}.png")

    return

def feature_scatter_binary(f1, f2, D, L, classes):
    vetAttr1 = D[f1, :]
    vetAttr2 = D[f2, :]
    
    fig = plt.figure(figsize=(6, 6))
    for classIndex, className in enumerate(classes):
        mask = (L == classIndex)
        D0 = vetAttr1[mask]
        D1 = vetAttr2[mask]
        plt.scatter(D0, D1)
    plt.savefig(f"01_features_analysis_pt1\\ftr_{f1}_{f2}.png")
    plt.close()

def PCA(D, L, m):
    mu = D.mean(1) # mu will be a row vector so we have to convert it into a column vector
    Dc = D - vcol(mu) # centered dataset D - the column representation of mu
    C = (1/D.shape[1])*np.dot(Dc, Dc.T) # C is the covariance matrix

    # find eigenvalues and eigenvectors with the function numpy.linalg.eigh
    s, U = np.linalg.eigh(C) # eigh returns the eigenvalues and the eignevectors sorted from smallest to larger

    # find eigenvalues and eigenvectors with SVD 
    #U, s, Vh = np.linalg.svd(C)

    # [:, ::-1] takes all the columns and sort them in reverse order
    # [:, 0:m] takes all the row from 0 to index m
    P = U[:, ::-1][:, 0:m]
    Dp = np.dot(P.T, D) # project the dataset to the new space

    return Dp

def PCA_directions(D, m):
    mu = D.mean(1) # mu will be a row vector so we have to convert it into a column vector
    Dc = D - vcol(mu) # centered dataset D - the column representation of mu
    C = (1/D.shape[1])*np.dot(Dc, Dc.T) # C is the covariance matrix

    # find eigenvalues and eigenvectors with the function numpy.linalg.eigh
    s, U = np.linalg.eigh(C) # eigh returns the eigenvalues and the eignevectors sorted from smallest to larger

    # find eigenvalues and eigenvectors with SVD 
    #U, s, Vh = np.linalg.svd(C)

    # [:, ::-1] takes all the columns and sort them in reverse order
    # [:, 0:m] takes all the row from 0 to index m
    P = U[:, ::-1][:, 0:m]

    return P

def PCA_plot(D, L, m=2):
    mu = D.mean(1) # mu will be a row vector so we have to convert it into a column vector
    Dc = D - vcol(mu) # centered dataset D - the column representation of mu
    C = (1/D.shape[1])*np.dot(Dc, Dc.T) # C is the covariance matrix

    # find eigenvalues and eigenvectors with the function numpy.linalg.eigh
    s, U = np.linalg.eigh(C) # eigh returns the eigenvalues and the eignevectors sorted from smallest to larger

    # find eigenvalues and eigenvectors with SVD 
    #U, s, Vh = np.linalg.svd(C)

    # [:, ::-1] takes all the columns and sort them in reverse order
    # [:, 0:m] takes all the row from 0 to index m
    P = U[:, ::-1][:, 0:m]
    Dp = np.dot(P.T, D) # project the dataset to the new space
    
    # create the matrix related to the specific classes
    D0 = Dp[:, L==0]
    D1 = Dp[:, L==1]

    # Create the figure
    fig = plt.figure(figsize=(6, 6))

    if m == 2:
        plt.scatter(D0[0], D0[1], label='Spoofed')
        plt.scatter(D1[0], D1[1], label='Authentic')

    if m == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(D0[0], D0[1], D0[2], label='Spoofed')
        ax.scatter(D1[0], D1[1], D1[2], label='Authentic')

    plt.savefig(f"01_features_analysis_pt2\\PCA_scatter_{m}.png")
    plt.close()

    if m == 2:
        plt.figure()
        plt.hist(D0[0], bins = 50, density=True, alpha=0.3, label="Spoofed")
        plt.hist(D1[0], bins = 50, density=True, alpha=0.3, label="Authentic")
        plt.savefig(f"01_features_analysis_pt2\\PCA_hist_0.png")
        plt.figure()
        plt.hist(D0[1], bins = 50, density=True, alpha=0.3, label="Spoofed")
        plt.hist(D1[1], bins = 50, density=True, alpha=0.3, label="Authentic")
        plt.savefig(f"01_features_analysis_pt2\\PCA_hist_1.png")

    return Dp

def LDA_plot(D, L, m=2):
    # create the matrix related to the specific classes
    D0 = D[:, L==0]
    D1 = D[:, L==1]

    # compute the average of each class and the total one
    mu = vcol(D.mean(1))
    mu0 = vcol(D0.mean(1))
    mu1 = vcol(D1.mean(1))

    # compute centered datasets
    Dc0 = D0 - mu0
    Dc1 = D1 - mu1

    # compute the covariance matrices
    C0 = (1/D0.shape[1])*np.dot(Dc0, Dc0.T)
    C1 = (1/D1.shape[1])*np.dot(Dc1, Dc1.T)
    
    # within class covariance matrix
    Sw = (1/D.shape[1])*(D0.shape[1]*C0 + D1.shape[1]*C1)

    # between class covariance matrix
    Sb0 = D0.shape[1]*np.dot(mu0 - mu, (mu0 - mu).T)
    Sb1 = D1.shape[1]*np.dot(mu1 - mu, (mu1 - mu).T)
    Sb = (1/D.shape[1])*(Sb0 + Sb1)

    # solve the generalized eigenvalue problem Sb*w=lambda*Sw*w with sp.linalg.eigh
    s, U = sp.linalg.eigh(Sb, Sw)
    
    # get W as the first m eigenvectors of U
    W = U[:, ::-1][:, 0:m]

    # since the columns of W are not necessary orthogonal we can find a base U using SVD
    #Uw, s, Vh = np.linalg.svd(W)
    #U = Uw[:, 0:m]

    Dp = np.dot(W.T, D) # project the dataset to the new space
    
    # create the matrix related to the specific classes
    D0 = Dp[:, L==0]
    D1 = Dp[:, L==1]

    # Create the figure
    fig = plt.figure(figsize=(6, 6))
    
    if m == 2:
        plt.scatter(D0[0], D0[1], label='Spoofed')
        plt.scatter(D1[0], D1[1], label='Authentic')

    if m == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(D0[0], D0[1], D0[2], label='Spoofed')
        ax.scatter(D1[0], D1[1], D1[2], label='Authentic')

    plt.savefig(f"01_features_analysis_pt2\\LDA_scatter_{m}.png")

    if m == 2:
        plt.figure()
        plt.hist(D0[0], bins = 50, density=True, alpha=0.3, label="Spoofed")
        plt.hist(D1[0], bins = 50, density=True, alpha=0.3, label="Authentic")
        plt.savefig(f"01_features_analysis_pt2\\LDA_hist_0.png")
        plt.figure()
        plt.hist(D0[1], bins = 50, density=True, alpha=0.3, label="Spoofed")
        plt.hist(D1[1], bins = 50, density=True, alpha=0.3, label="Authentic")
        plt.savefig(f"01_features_analysis_pt2\\LDA_hist_1.png")

    return Dp

def PCA_data_variance(D):
    mu = D.mean(1) # mu will be a row vector so we have to convert it into a column vector
    Dc = D - vcol(mu) # centered dataset D - the column representation of mu
    C = (1/D.shape[1])*np.dot(Dc, Dc.T) # C is the covariance matrix

    # find eigenvalues and eigenvectors with the function numpy.linalg.eigh
    s, U = np.linalg.eigh(C) # eigh returns the eigenvalues and the eignevectors sorted from smallest to larger

    s = np.sort(s)[::-1]
    y = []

    for i in range(len(s)):
        n = s[0:i].sum()
        d = s.sum()
        y = np.append(y, n/d)

    x = np.linspace(0, 10, 10, endpoint=True)
    
    plt.figure()
    plt.plot(x, y)
    plt.grid()
    plt.xlabel('PCA dimensions')
    plt.ylabel('Fraction of explained variance')
    plt.savefig('01_features_analysis_pt2\\explained_variance.png')
    plt.close()

    return

def centering(D):
    mu = D.mean(1)
    return D - vcol(mu)

def std_variances(D):
    mu = D.mean(1) 
    Dc = D - vcol(mu) 
    C = (1/D.shape[1])*np.dot(Dc, Dc.T)
    
    diag = np.reshape(np.diag(C), (D.shape[0], 1))
    diag = np.sqrt(diag)

    D = D/diag

    return D

def whitening(Dx, D):
    mu = D.mean(1) 
    Dc = D - vcol(mu) 
    C = (1/D.shape[1])*np.dot(Dc, Dc.T)

    sqrtC = sp.linalg.fractional_matrix_power(C, 0.5)
    Dw = np.dot(sqrtC, Dx)
    
    return Dw

def l2(D): 
    for i in range(D.shape[1]):
        D[:, i] = D[:, i]/np.linalg.norm(D[:, i])

    return D

def heatmaps_binary(D, L):
    DT = D.T
    D_auth = D[:, L==1].T
    D_spoofed = D[:, L==0].T

    df = pd.DataFrame(DT)
    corr = df.corr()
    plt.figure()
    sb.heatmap(corr, cmap="Blues")
    plt.savefig("01_features_analysis_pt2\\heatmap.png")
    
    df = pd.DataFrame(D_auth)
    corr = df.corr()
    plt.figure()
    sb.heatmap(corr, cmap="Blues")
    plt.savefig("01_features_analysis_pt2\\heatmap_auth.png")
    
    df = pd.DataFrame(D_spoofed)
    corr = df.corr()
    plt.figure()
    sb.heatmap(corr, cmap="Blues")
    plt.savefig("01_features_analysis_pt2\\heatmap_spoofed.png")

def compute_mu_C(D, L, label, NB=False):
    DL = D[:, L == label]
    mu = DL.mean(1).reshape(DL.shape[0], 1)
    DLC = (DL - mu)
    C = 1/DLC.shape[1]*np.dot(DLC, DLC.T)

    if NB:
        C = np.multiply(C, np.identity(DL.shape[0]))

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

def logreg_obj_weight_wrap(DTR, LTR, l, pt):
    def logreg_derivative_b(v):
        w, b = np.array(v[0:-1]), v[-1]

        result = 0
        for i in range(0, DTR.shape[1]):
            z = 2 * LTR[i] -1
            exp = np.exp((-z) * (np.dot(w.T, DTR.T[i]) + b))
            result += (exp * (-z) / (1 + exp))
        return result / DTR.shape[1]

    def logreg_derivative_w(v):
        w, b = np.array(v[0:-1]), v[-1]

        result = 0
        for i in range(0, DTR.shape[1]):
            z = 2 * LTR[i] -1
            exp = np.exp((-z) * (np.dot(w.T, DTR.T[i]) + b))
            result += (exp * (-z * DTR.T[i]) / (1 + exp))
        return result / DTR.shape[1] + l * w

    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        J = 0

        for i in range(0, DTR.shape[1]):
            zi = 2*LTR[i] - 1
            if zi > 0:
                J += (pt/LTR.sum())*np.logaddexp(0, -zi*(np.dot(w, DTR[:, i]) + b))
            else:
                J += ((1 - pt)/(LTR.shape[0] - LTR.sum()))*np.logaddexp(0, -zi*(np.dot(w, DTR[:, i]) + b))

        J += l/2*np.linalg.norm(w)**2

        return (J, np.concatenate([logreg_derivative_w(v), [logreg_derivative_b(v)]]))

    return logreg_obj

def logreg_obj_wrap(DTR, LTR, l):
    def logreg_derivative_b(v):
        w, b = np.array(v[0:-1]), v[-1]

        result = 0
        for i in range(0, DTR.shape[1]):
            z = 2 * LTR[i] -1
            exp = np.exp((-z) * (np.dot(w.T, DTR.T[i]) + b))
            result += (exp * (-z) / (1 + exp))
        return result / DTR.shape[1]

    def logreg_derivative_w(v):
        w, b = np.array(v[0:-1]), v[-1]

        result = 0
        for i in range(0, DTR.shape[1]):
            z = 2 * LTR[i] -1
            exp = np.exp((-z) * (np.dot(w.T, DTR.T[i]) + b))
            result += (exp * (-z * DTR.T[i]) / (1 + exp))
        return result / DTR.shape[1] + l * w

    def logreg_obj(v):
        w, b = v[0:-1], v[-1]
        J = 0

        for i in range(0, DTR.shape[1]):
            zi = 2*LTR[i] - 1
            J += np.logaddexp(0, -zi*(np.dot(w, DTR[:, i]) + b))

        J /= DTR.shape[1]
        J += l/2*np.linalg.norm(w)**2

        return (J, np.concatenate([logreg_derivative_w(v), [logreg_derivative_b(v)]]))

    return logreg_obj

def expand_feature_space(D):
    newD = np.ndarray((D.shape[0]**2 + D.shape[0], 1))
    for i in range(0, D.shape[1]):
        x = D[:, i]
        x = x.reshape((D.shape[0], 1))
        
        XXT = np.dot(x, x.T)
        phy = np.matrix.flatten(XXT)
        
        phy = np.concatenate([phy, np.matrix.flatten(x)])
        # this is now our x. We now need to put all out xs together
        newD = np.concatenate([newD, phy.reshape(D.shape[0]**2 + D.shape[0], 1)], axis=1)
    
    newD = newD[:, 1:]
    return newD

def DCF_actual(prior, Cfn, Cfp, s_log_ratio, labels):

    t = -np.log((prior * Cfn)/((1 - prior) * Cfp))
    c = s_log_ratio > t

    CMD = np.zeros((2, 2), dtype=int)

    for i, p in enumerate(c):
        CMD[int(p), int(labels[i])] += 1

    FNR = CMD[0, 1]/(CMD[0, 1] + CMD[1, 1])
    FPR = CMD[1, 0]/(CMD[0, 0] + CMD[1, 0])

    DCF = prior*Cfn*FNR+(1-prior)*Cfp*FPR

    Bdummy = np.min([prior * Cfn, (1 - prior) * Cfp])
    return DCF / Bdummy

def DCF_min(prior, Cfn, Cfp, s_log_ratio, labels):
    
    Bdummy = np.min([prior * Cfn, (1 - prior) * Cfp])
    DCF = np.array([])

    for t in s_log_ratio:
        #print(f"analysing threshold {t} for min dcf")
        c = s_log_ratio > t
        CMD = np.zeros((2, 2), dtype=int)

        for i, p in enumerate(c):
            #print(f"computing sample number {i} for confusion matrix")
            CMD[int(p), int(labels[i])] += 1

        FNR = CMD[0, 1]/(CMD[0, 1] + CMD[1, 1])
        FPR = CMD[1, 0]/(CMD[0, 0] + CMD[1, 0])

        DCF = np.append(DCF, (prior*Cfn*FNR+(1-prior)*Cfp*FPR)/Bdummy)

    return np.min(DCF)

def bayer_error_plots(s_log_ratio, labels):
    effPriorLogOdds = np.linspace(-4, 4, 20)

    actualdcf = np.array([])
    mindcf = np.array([])

    effective_prior = 1/(np.exp(-effPriorLogOdds) + 1)

    for p in effective_prior:
        actualdcf = np.append(actualdcf, DCF_actual(p, 1, 1, s_log_ratio, labels))
        mindcf = np.append(mindcf, DCF_min(p, 1, 1, s_log_ratio, labels))

    return actualdcf, mindcf

def svm_wraper(H, DTR):
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
    [alpha, LD, d] = sp.optimize.fmin_l_bfgs_b(svm_wraper(H, DTR), np.zeros((DTR.shape[1],1)), bounds=BC, factr=1.0)
    
    
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
    Z = np.reshape(Z, (LTR.shape[0], 1))

    Kprime = np.dot(DTR.T, DTR)
    Zprime = np.dot(Z, Z.T)
    Kmat = ((Kprime + c) ** d) + K**2
    H = np.multiply(Zprime, Kmat)

    BC = [(0, C) for i in range(0, DTR.shape[1])]
    [alpha, f, d2] = sp.optimize.fmin_l_bfgs_b(svm_wraper(H, DTR), np.zeros((DTR.shape[1],1)), bounds=BC, factr=1.0)
    
    S = np.ones((DTE.shape[1]))

    alpha = np.reshape(alpha, (alpha.shape[0], 1))
    az = np.multiply(alpha, Z)
    Kprime = np.dot(DTR.T, DTE)
    Kmat = ((Kprime + c) ** d) + K**2
    S = np.multiply(az, Kmat).sum(axis=0)
    
    return S

def compute_svm_polykernel_weighted(DTR, LTR, DTE, K, C, d, c, pt):
    Z = LTR * 2 - 1
    Z = np.reshape(Z, (LTR.shape[0], 1))

    Kprime = np.dot(DTR.T, DTR)
    Zprime = np.dot(Z, Z.T)
    Kmat = ((Kprime + c) ** d) + K**2
    H = np.multiply(Zprime, Kmat)

    # coputing Ci
    pemp = LTR.sum()/LTR.shape[0]
    Ct = C*pt/pemp
    Cf = C*(1 - pt)/(1-pemp)

    BC = []
    for i in range(DTR.shape[1]):
        if LTR[i] == 1:
            BC.append((0, Ct))
        else:
            BC.append((0, Cf))

    [alpha, f, d2] = sp.optimize.fmin_l_bfgs_b(svm_wraper(H, DTR), np.zeros((DTR.shape[1],1)), bounds=BC, factr=1.0)
    
    S = np.ones((DTE.shape[1]))

    alpha = np.reshape(alpha, (alpha.shape[0], 1))
    az = np.multiply(alpha, Z)
    Kprime = np.dot(DTR.T, DTE)
    Kmat = ((Kprime + c) ** d) + K**2
    S = np.multiply(az, Kmat).sum(axis=0)
    
    return S

def compute_svm_RBF(DTR, LTR, DTE, K, C, g):
    Z = LTR * 2 - 1
    
    Z = np.reshape(Z, (LTR.shape[0], 1))
    H = np.dot(Z, Z.T)

    # will compute H in with for loops
    for i in range(0, DTR.shape[1]):
        for j in range(0, DTR.shape[1]):
            H[i][j] *= (np.exp(-g*(np.linalg.norm(DTR.T[i] - DTR.T[j]))**2) + K**2)
            
    BC = [(0, C) for i in range(0, DTR.shape[1])]
    [alpha, f, d2] = sp.optimize.fmin_l_bfgs_b(svm_wraper(H, DTR), np.zeros((DTR.shape[1],1)), bounds=BC, factr=1.0)
    
    S = np.ones((DTE.shape[1]))

    for t in range(0, DTE.shape[1]):
        result = 0
        for i in range(0, DTR.shape[1]):
            result += alpha[i]*Z[i]*(np.exp(-g*(np.linalg.norm(DTR.T[i] - DTE.T[t]))**2) + K**2)
        S[t] = result
        
    return S

def compute_svm_RBF_weighted(DTR, LTR, DTE, K, C, g, pt):
    Z = LTR * 2 - 1
    
    Z = np.reshape(Z, (LTR.shape[0], 1))
    H = np.dot(Z, Z.T)

    # will compute H in with for loops
    for i in range(0, DTR.shape[1]):
        for j in range(0, DTR.shape[1]):
            H[i][j] *= (np.exp(-g*(np.linalg.norm(DTR.T[i] - DTR.T[j]))**2) + K**2)
            
    # coputing Ci
    pemp = LTR.sum()/LTR.shape[0]
    Ct = C*pt/pemp
    Cf = C*(1 - pt)/(1-pemp)

    BC = []
    for i in range(DTR.shape[1]):
        if LTR[i] == 1:
            BC.append((0, Ct))
        else:
            BC.append((0, Cf))
            
    [alpha, f, d2] = sp.optimize.fmin_l_bfgs_b(svm_wraper(H, DTR), np.zeros((DTR.shape[1],1)), bounds=BC, factr=1.0)
    
    S = np.ones((DTE.shape[1]))

    for t in range(0, DTE.shape[1]):
        result = 0
        for i in range(0, DTR.shape[1]):
            result += alpha[i]*Z[i]*(np.exp(-g*(np.linalg.norm(DTR.T[i] - DTE.T[t]))**2) + K**2)
        S[t] = result
        
    return S

def logpdf_GMM(X, gmm):

    S = np.empty(shape=(1, X.shape[1]))

    for g in range(len(gmm)):
        Sg = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2])
        Sg += np.log(gmm[g][0])
        S = np.vstack([S, Sg])

    S = S[1:, :]

    logdens = sp.special.logsumexp(S, axis=0)

    return logdens

def E_step(X, gmm):
    S = np.empty(shape=(1, X.shape[1]))

    for g in range(len(gmm)):
        Sg = logpdf_GAU_ND(X, gmm[g][1], gmm[g][2])
        Sg += np.log(gmm[g][0])
        S = np.vstack([S, Sg])

    S = S[1:, :]

    marginals = sp.special.logsumexp(S, axis=0)
    posteriors = S - marginals

    resps = np.exp(posteriors)
    ll = marginals.sum()/X.shape[1]

    return resps, ll

def M_step(X, resps, psi=0, diag=False, tied=False):
    triplets = []
    gmm = []
    Zsum = 0
    Csum = np.zeros(shape=(X.shape[0], X.shape[0]))
    for g in range(resps.shape[0]):
       Zg, Fg, Sg = M_step_g(X, resps[g, :])
       triplets.append((Zg, Fg, Sg))
       Zsum += Zg

    for g in range(resps.shape[0]):
        (Zg, Fg, Sg) = triplets[g]
        mu = vcol(Fg/Zg)
        C = (Sg/Zg) - np.dot(mu, mu.T)
        w = Zg/Zsum
        Csum += (Zg*C)

        gmm.append((w, mu, C))

    for g in range(resps.shape[0]):
        (w, mu, C) = gmm[g]
        
        if tied == True:
            C = (1/X.shape[1])*Csum

        if diag == True:
            C = np.eye(C.shape[0])
        
        U, s, _ = np.linalg.svd(C)
        s[s < psi] = psi
        C = np.dot(U, vcol(s)*U.T)

        gmm[g] = (w, mu, C)

    return gmm

def M_step_g(X, resp):
    Zg = resp.sum()
    Fg = np.multiply(X, resp).sum(axis=1)
    Sg = np.zeros(shape=(X.shape[0], X.shape[0]))
    for i in range(X.shape[1]):
        xi = vcol(X[:, i])
        Sg += resp[i]*np.dot(xi, xi.T)

    return (Zg, Fg, Sg) 

def splitGMM(gmm, alpha=0.1):
    newGMM = []
    for i in range(len(gmm)):
        w = gmm[i][0]
        mu = gmm[i][1]
        C = gmm[i][2]
        U, s, Vh = np.linalg.svd(C)
        d = U[:, 0:1] * s[0]**0.5 * alpha
        newGMM.append((w/2, mu + d, C))
        newGMM.append((w/2, mu - d, C))

    return newGMM

def EM(D, gmm, precision=1e-6, psi=0, diag=False, tied=False):
    ll = sys.float_info.min
    end = False
    while not end:
        resps, ll_n = E_step(D, gmm)
        if np.abs(ll-ll_n) < precision:
            break
        ll = ll_n
        gmm = M_step(D, resps, psi, diag, tied)

    return gmm, ll

def LBG(D, gmm=None, precision=1e-6, psi=0, diag=False, tied=False, alpha=0.1):
    mu = vcol(D.mean(1))
    C = (1/D.shape[1])*np.dot(D - mu, (D - mu).T)
    
    if gmm == None:
        U, s, _ = np.linalg.svd(C)
        s[s < psi] = psi
        C = np.dot(U, vcol(s)*U.T)
        gmm = [(1.0, mu, C)]
    else:
        gmm = splitGMM(gmm, alpha)

    ll = sys.float_info.min
    end = False
    while not end:
        resps, ll_n = E_step(D, gmm)
        if np.abs(ll - ll_n) < 1e-6:
            break
        ll = ll_n
        gmm = M_step(D, resps, psi, diag, tied)

    return gmm, ll

def LBG_wrap(D, gmm=None, n_iter=1, precision=1e-6, psi=0, diag=False, tied=False, alpha=0.1):
    for i in range(n_iter):
        [gmm, ll] = LBG(D, gmm, precision, psi, diag, tied, alpha)

    return gmm, ll

def ROC_curve(prior, Cfn, Cfp, s_log_ratio, labels):

    FPR = np.array([])
    TPR = np.array([])

    thresholds = np.array(s_log_ratio)

    thresholds = np.insert(thresholds, 0, sys.float_info.min)
    thresholds = np.insert(thresholds, 0, sys.float_info.max)

    thresholds = np.sort(thresholds)

    for t in thresholds:
        c = s_log_ratio > t
        CMD = np.zeros((2, 2), dtype=int)

        for i, p in enumerate(c):
            CMD[int(p), int(labels[i])] += 1

        FPR = np.append(FPR, CMD[0, 1]/(CMD[0, 1] + CMD[1, 1]))
        TPR = np.append(TPR , 1-CMD[0, 1]/(CMD[0, 1] + CMD[1, 1]))

    FPR = np.sort(FPR)
    TPR = np.sort(TPR)

    return TPR, FPR

def DET_curve(prior, Cfn, Cfp, s_log_ratio, labels):
    FPR = np.array([])
    FNR = np.array([])

    thresholds = np.array(s_log_ratio)

    thresholds = np.insert(thresholds, 0, sys.float_info.min)
    thresholds = np.insert(thresholds, 0, sys.float_info.max)

    thresholds = np.sort(thresholds)

    for t in thresholds:
        c = s_log_ratio > t
        CMD = np.zeros((2, 2), dtype=int)

        for i, p in enumerate(c):
            CMD[int(p), int(labels[i])] += 1

        FPR = np.append(FPR, CMD[0, 1]/(CMD[0, 1] + CMD[1, 1]))
        FNR = np.append(FNR, CMD[1, 0]/(CMD[0, 0] + CMD[1, 0]))

    FPR = np.sort(FPR)
    FNR = np.sort(FNR)[::-1]

    return FNR, FPR