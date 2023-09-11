import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b as bfgs
import os.path as path
import numpy as np
import math
from scipy import special as sp
import seaborn as sb
import pandas as pd
from scipy.interpolate import make_interp_spline

colors = ['cyan','red']
etichetta = ["Same speaker", "Different speakers"]

def vcol(row): 
  return row.reshape(row.size, 1)

def vrow(col): 
  return col.reshape(1, col.size)

def readTrainAndTestData():
	DTR=None
	LTR=None
	DTE=None
	LTE=None
	if not path.exists("data/DTR.npy") or not path.exists("data/LTR.npy") or not path.exists("data/DTE.npy") or not path.exists("data/LTE.npy"):
		LTR = np.array([], dtype=int)
		LTE = np.array([], dtype=int)

		ftr = open("data/Train.txt", "r")
		list = []

		for line in ftr:
			parts = line.split(",")
			l=int(parts.pop())
			LTR = np.append(LTR, l)
			list.append(parts)

		DTR = np.array(list, dtype=float).T

		ftr.close()

		fte = open("data/Test.txt", "r")
		list = []

		for line in fte:
			parts = line.split(",")
			l=int(parts.pop())
			LTE = np.append(LTE, l)
			list.append(parts)

		DTE = np.array(list, dtype=float).T
		fte.close()

		np.save("data/DTR", DTR)
		np.save("data/LTR", LTR)
		np.save("data/DTE", DTE)
		np.save("data/LTE", LTE)

	else:
		DTR = np.load("data/DTR.npy")
		LTR = np.load("data/LTR.npy")
		DTE = np.load("data/DTE.npy")
		LTE = np.load("data/LTE.npy")

	return (DTR, LTR), (DTE, LTE)

def centerData(data):
	u = data.mean(1)
	return data-vcol(u)

def statistics(data, labels):
	centeredData = centerData(data)
	for i, feature in enumerate(centeredData):
		plt.figure()
		plt.hist(feature[labels==1], density=True, alpha=0.5, label="female", color='red' , bins=50)
		plt.hist(feature[labels==0], density=True, alpha=0.4, label="male", color='blue', bins=50)
		plt.savefig("plots/hist/feature" + str(i+1))
		plt.title('feature ' + str(i+1))
		plt.close()

	for i in range(data.shape[0]):
		for j in range(i+1, data.shape[0]):
			plt.figure()
			plt.scatter(data[i, labels==0], data[j, labels==0], c="blue", alpha=0.7, label="male")
			plt.scatter(data[i, labels==1], data[j, labels==1], c="red", alpha=0.7, label="female")
			plt.xlabel(str(i+1))
			plt.ylabel(str(j+1))
			plt.legend()
			plt.savefig("plots/scatter/" + str(i+1) + "vs" + str(j+1))
			plt.close() 

def explainedVariance(D):
	Dc = centerData(D)
	C = (1/D.shape[1])*np.dot(Dc, Dc.T)
	s = np.linalg.eigh(C)[0]
	s = np.sort(s)[::-1]
	y = []

	for i in range(len(s)):
		n = s[0:i].sum()
		d = s.sum()
		y = np.append(y, n/d)

	x = np.linspace(1,12,12, endpoint=True)

	plt.figure()
	plt.plot(x,y,marker='o')
	plt.grid()
	plt.xlabel('#dimensions')
	
	plt.ylabel('Fraction of explained variance')
	plt.savefig('plots/explainedVariance')
	plt.close()

def heatmaps_binary(data, labels):
	data_male = data[:, labels==0]
	data_female = data[:, labels==1]

	df = pd.DataFrame(data.T)
	corr = df.corr()
	plt.figure()
	sb.heatmap(corr, cmap="Greys")
	plt.title("Data")
	plt.savefig("plots/heatmap/heatmap_all")

	df = pd.DataFrame(data_female.T)
	corr = df.corr()
	plt.figure()
	sb.heatmap(corr, cmap="Reds")
	plt.title("Data female")
	plt.savefig("plots/heatmap/heatmap_female")

	df = pd.DataFrame(data_male.T)
	corr = df.corr()
	plt.figure()
	sb.heatmap(corr, cmap="Blues")
	plt.title("Data male")
	plt.savefig("plots/heatmap/heatmap_male")

def pca(data, m):
	data_centered = centerData(data)
	N = data.shape[1]
	C = 1/N*(np.dot(data_centered, data_centered.T))
	U = np.linalg.svd(C)[0]
	P = U[:, 0:m]
	DP = np.dot(P.T, data)

	return DP

def lda(x, labels, m, plot):
	u = vcol(x.mean(1))
	N = x.shape[1]
	data_centered = x - u
	C = 1/N*(np.dot(data_centered, data_centered.T))

	wC = np.zeros(C.shape, dtype = float)
	bC = np.zeros(C.shape, dtype = float)

	for l in np.unique(labels):
		xl = x[:,labels==l]		#class data
		Nl = xl.shape[1]		#class number of samples
		ul = vcol(xl.mean(1)) 	#class mean
		xcl = xl - vcol(ul)		#class centered data

		bC += Nl*np.dot(ul-u, (ul-u).T)

		Cl = np.dot(xcl, xcl.T)
		wC += Cl

	wC/=N
	bC/=N

	U, s, _ = np.linalg.svd(wC)
	P1 = np.dot( np.dot(U, np.diag(1.0/(s**0.5))), U.T )

	Sbt = np.dot(np.dot(P1, bC), P1.T)

	U = np.linalg.svd(Sbt)[0]
	P2 = U[:, 0:m]
	W = np.dot(P1.T, P2)
	 
	y = np.dot(W.T, x)

	y = np.reshape(y, y.shape[1])

	if plot:
		plt.figure()
		plt.hist(y[labels==1], density=True, alpha=0.5, label="female", color='red' , bins=50)
		plt.hist(y[labels==0], density=True, alpha=0.4, label="male", color='blue', bins=50)
		plt.title('LDA')
		plt.legend()
		plt.savefig("plots/lda/LDA")
		plt.close()

	return y

def calculateParametersMVG(x, labels):
		M, N = x.shape

		x0 = x[:,labels==0]
		M0, N0 = x0.shape
		mu0 = x0.mean(1).reshape(M0,1)
		xc0 = x0-mu0
		C0 = 1/N0 * np.dot(xc0, xc0.T)
		CsDiag0 = C0*np.eye(C0.shape[0])

		x1 = x[:,labels==1]
		M1, N1 = x1.shape
		mu1 = x1.mean(1).reshape(M1,1)
		xc1 = x1-mu1
		C1 = 1/N1 * np.dot(xc1, xc1.T)
		CsDiag1 = C1*np.eye(C1.shape[0])

		wC = (C0*N0 + C1*N1)/N

		return (mu0, mu1), (C0, C1), (CsDiag0, CsDiag1), wC 

def logpdf_GAU_ND(x, u, C):
	M = C.shape[0] #number of features
	xc = x-u

	log_det_C = np.linalg.slogdet(C)[1]

	log_N = -0.5*M*np.log(np.pi*2) - 0.5*log_det_C - 0.5*np.multiply(np.dot(xc.T, np.linalg.inv(C)), xc.T).sum(1)

	return log_N

#def llr_mvg():

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

def computeROC(S, LTE):
    S_sorted = np.sort(S)
    TPRs = np.array([])
    FPRs = np.array([])

    for t in S_sorted:
        predictions = np.array((S>t), dtype=int)

        CM = np.zeros((2,2), dtype=int)

        for i in range(predictions.size):
            CM[predictions[i], LTE[i]] += 1

        TPR = 1 - CM[0,1]/(CM[0,1]+CM[1,1])
        FPR = CM[1,0]/(CM[0,0]+CM[1,0])

        TPRs = np.append(TPRs, TPR)
        FPRs = np.append(FPRs, FPR)

    plt.figure()
    plt.plot(FPRs, TPRs)
    plt.title("ROC")
    plt.show()
    
def computeMinDCF(pi1, Cfn, Cfp, S, LTE):
	S_sorted = np.sort(S)
	Bdummy = np.min([pi1*Cfn,(1-pi1)*Cfp])

	DCFs = np.array([], dtype=float)

	for t in S_sorted:
		predictions = np.array((S>t), dtype=int)

		CM = np.zeros((2,2), dtype=int)
		for i in range(predictions.size):
			CM[predictions[i], int(LTE[i])] += 1

		FNR = CM[0,1]/(CM[0,1]+CM[1,1])
		FPR = CM[1,0]/(CM[0,0]+CM[1,0])
		DCF = (pi1*Cfn*FNR +(1-pi1)*Cfp*FPR)/Bdummy

		DCFs= np.append(DCFs, DCF)

	minDCF = np.min(DCFs)

	return minDCF

def computeActualDCF(pi1, Cfn, Cfp, llr, labels):
    CM = np.zeros((2,2), dtype=int)
    t  = -np.log((pi1*Cfn)/((1-pi1)*Cfp))
    predictions = np.array((llr>t), dtype=int)
    Bdummy = np.min([pi1*Cfn, (1-pi1)*Cfp])
    for i in range(predictions.size):
        CM[predictions[i], labels[i]] += 1

    FNR = CM[0,1]/(CM[0,1]+CM[1,1])
    FPR = CM[1,0]/(CM[0,0]+CM[1,0])
    DCF = (pi1*Cfn*FNR +(1-pi1)*Cfp*FPR)/Bdummy

    return DCF

def std_variances(D):
    Dc = centerData(D) 
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

def bayesErrorPlots(eplo, llr, labels):
    DCFs = np.array([], dtype=float)
    minDCFs = np.array([], dtype=float)
    
    for p in eplo:
        ep = 1/(1+math.exp(-p))
        DCF = computeActualDCF(ep, 1, 1, llr, labels)
        minDCF = computeMinDCF(ep, 1, 1, llr, labels)

        DCFs= np.append(DCFs, DCF)
        minDCFs= np.append(minDCFs, minDCF)

    plt.figure()
    plt.plot(eplo, minDCFs, label='min DCF', color='b')
    plt.plot(eplo, DCFs, label='DCF', color='r')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()