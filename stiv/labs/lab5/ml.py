import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

colors = ["b", "g", "r", "c", "m", "y", "k", "w"];

def load_iris():
	D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
	return D, L

#from row to col
def vcol(row):
	return row.reshape(row.size, 1)

#from col to row
def vrow(col):
	return col.reshape(1, col.size)

def pca(table, classes, m, plotYN):
	mean = table.mean(1)
	nAttr, nRecords = np.shape(table)
	centeredTable = table - vcol(mean)
	covMatrix = 1/nRecords*(np.dot(centeredTable, centeredTable.T))
	U, _, _ = np.linalg.svd(covMatrix)
	P=U[:, 0:m]
	P[:,1]*=-1
	DP=np.dot(P.T, table)
	
	if(plotYN): 
		for i, x in enumerate(classes):
			plt.scatter(DP[0,classes[:]==x], DP[1,classes[:]==x], color=colors[int(i/50)])

		plt.show();

	return DP

def lda(table, classes, m, plotYN):
	mean = table.mean(1)
	nAttr, nRecords = np.shape(table)
	centeredTable = table - vcol(mean)
	covMatrix = 1/nRecords*(np.dot(centeredTable, centeredTable.T))

	withinCovarianceMatrix = np.zeros(covMatrix.shape, dtype = float)
	betweenCovarianceMatrix = np.zeros(covMatrix.shape, dtype = float)

	for x in classes:
		classData = table[:,classes[:]==x]
		nAttrClass, nRecordsClass = np.shape(table)
		meanClass = classData.mean(1) #now this is a 1-D array. ocho!
		centerdClass = classData - vcol(meanClass)
		withinCovarianceMatrix += np.dot(centerdClass, centerdClass.T)/nRecordsClass
		
		betweenCovarianceMatrix += np.dot(vcol(meanClass)-vcol(mean), (vcol(meanClass)-vcol(mean)).T)

	withinCovarianceMatrix/=nRecords
	betweenCovarianceMatrix/=nRecords

	U, s, _ = np.linalg.svd(withinCovarianceMatrix)
	P1 = np.dot(U * vrow(1.0/(s**0.5)), U.T) #We find Pw through whitening transformation
	# or P1 = numpy.dot( numpy.dot(U, numpy.diag(1.0/(s**0.5))), U.T )

	Sbt = np.dot(np.dot(P1, betweenCovarianceMatrix), P1.T)

	s, U = np.linalg.eigh(Sbt)
	P2 = U[:, ::-1][:, 0:m]
	W = np.dot(P1.T, P2)
	W[:,0]*=-1 
	y = np.dot(W.T, table)

	if plotYN:
		for i, x in enumerate(classes):
			plt.scatter(y[0,classes[:]==x], y[1,classes[:]==x], color=colors[int(i/50)])

		plt.show();

	return y

def logpdf_GAU_ND(x, mu, C):
    # X array of shape(M, N)
    # mu array of shape (M, 1)
    # C array of shape (M, M) that represents the covariance matrix
    M = C.shape[0] #number of features
    xc = x - mu
    log_det_C = np.linalg.slogdet(C)[1]
    
    log_N = -0.5*M*np.log(np.pi*2) - 0.5*log_det_C - 0.5*np.multiply(np.dot(xc.T, np.linalg.inv(C)), xc.T).sum(1)

    # logN is an array of length N (# of samples)
    # each element represents the log-density of each sample

    return log_N

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