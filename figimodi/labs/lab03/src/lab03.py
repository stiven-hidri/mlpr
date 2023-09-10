import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def vcol(mat):
    return mat.reshape((mat.size, 1)) 

def vrow(mat):
    return mat.reshape((1, mat.size))

def load(fname):
    DList = []
    labelsList = []
    hLabels = {
        'Iris-setosa': 0,
        'Iris-versicolor': 1,
        'Iris-virginica': 2
        }

    with open(fname) as f:
        for line in f:
            try:
                attrs = line.split(',')[0:4]
                attrs = vcol(np.array([float(i) for i in attrs]))
                name = line.split(',')[-1].strip()
                label = hLabels[name]
                DList.append(attrs)
                labelsList.append(label)
            except:
                pass

    return np.hstack(DList), np.array(labelsList, dtype=np.int32)

def PCA(D, L, m=2):
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
    D2 = Dp[:, L==2]

    # Create the figure
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(f'PCA for {m} components')

    if m == 2:
        plt.scatter(D0[0], D0[1], label='Setosa')
        plt.scatter(D1[0], D1[1], label='Versicolor')
        plt.scatter(D2[0], D2[1], label='Virginica')

    if m == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(D0[0], D0[1], D0[2], label='Setosa')
        ax.scatter(D1[0], D1[1], D1[2], label='Versicolor')
        ax.scatter(D2[0], D2[1], D2[2], label='Virginica')

def LDA(D, L, m=2):
    # create the matrix related to the specific classes
    D0 = D[:, L==0]
    D1 = D[:, L==1]
    D2 = D[:, L==2]

    # compute the average of each class and the total one
    mu = vcol(D.mean(1))
    mu0 = vcol(D0.mean(1))
    mu1 = vcol(D1.mean(1))
    mu2 = vcol(D2.mean(1))

    # compute centered datasets
    Dc0 = D0 - mu0
    Dc1 = D1 - mu1
    Dc2 = D2 - mu2

    # compute the covariance matrices
    C0 = (1/D0.shape[1])*np.dot(Dc0, Dc0.T)
    C1 = (1/D1.shape[1])*np.dot(Dc1, Dc1.T)
    C2 = (1/D2.shape[1])*np.dot(Dc2, Dc2.T)
    
    # within class covariance matrix
    Sw = (1/D.shape[1])*(D0.shape[1]*C0 + D1.shape[1]*C1 + D2.shape[1]*C2)

    # between class covariance matrix
    Sb0 = D0.shape[1]*np.dot(mu0 - mu, (mu0 - mu).T)
    Sb1 = D1.shape[1]*np.dot(mu1 - mu, (mu1 - mu).T)
    Sb2 = D2.shape[1]*np.dot(mu2 - mu, (mu2 - mu).T)
    Sb = (1/D.shape[1])*(Sb0 + Sb1 + Sb2)

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
    D2 = Dp[:, L==2]

    # Create the figure
    fig = plt.figure(figsize=(6, 6))
    fig.suptitle(f'LDA for {m} components')

    if m == 2:
        plt.scatter(D0[0], D0[1], label='Setosa')
        plt.scatter(D1[0], D1[1], label='Versicolor')
        plt.scatter(D2[0], D2[1], label='Virginica')

    if m == 3:
        ax = fig.add_subplot(projection='3d')
        ax.scatter(D0[0], D0[1], D0[2], label='Setosa')
        ax.scatter(D1[0], D1[1], D1[2], label='Versicolor')
        ax.scatter(D2[0], D2[1], D2[2], label='Virginica')

    return

if __name__ == '__main__':
    D, L = load('iris.csv')
    PCA(D, L, 2) # call PCA with m=2
    #PCA(D, L, 3) # call PCA with m=3
    LDA(D, L, 2) # call LDA with m=2
    #LDA(D, L, 3) # call LDA with m=3
    plt.show()
