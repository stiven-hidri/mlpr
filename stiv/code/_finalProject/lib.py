import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b as bfgs
import os.path as path
import numpy as np
import math
from scipy import special as sp
import seaborn as sb
import pandas as pd

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

def calculateParameters(x, labels):
		M, N = x.shape
		mus = np.array([])
		Cs = np.array([])
		wC = np.zeros((M,M))

		for l in np.unique(labels):
			xl = x[:,labels==l]
			mul = xl.mean(1).reshape(M,1)

			mus = np.column_stack((mus, mul)) if mus.size > 0 else mul

			xcl = xl-mul
			Nl = np.shape(xl)[1]
			Cl = 1/Nl * np.dot(xcl, xcl.T)
			Cs = np.vstack((Cs, [Cl])) if Cs.shape[0] > 0 else np.array([Cl])

			wC += Cl*Nl

		wC /= N

		wCs = np.array([wC, wC])

		CsDiag = np.vstack( ( [ Cs[0]*np.eye(Cs[0].shape[0]) ] , [ Cs[1]*np.eye(Cs[1].shape[0]) ] ) )

		return mus, Cs, wCs, CsDiag

def logpdf_GAU_ND(x, u, C):
	M = C.shape[0] #number of features
	xc = centerData(x)

	log_det_C = np.linalg.slogdet(C)[1]

	log_N = -0.5*M*np.log(np.pi*2) - 0.5*log_det_C - 0.5*np.multiply(np.dot(xc.T, np.linalg.inv(C)), xc.T).sum(1)

	return log_N

def mvg(mus, Cs, DTE, LTE):
	S = np.array([])

	for i in range(Cs.shape[0]):
		Si = logpdf_GAU_ND(DTE, mus[:,i], Cs[i])
		S = np.stack((S, Si))# if S.size>0 else Si

	Pc = 0.1

	## work with logarithms
	logSJoint = S + np.log(Pc)

	logSMarginal = vrow(sp.logsumexp(S, axis=0))

	logSPost = logSJoint-logSMarginal
	SPost = np.exp(logSPost)

	pcl = np.argmax(SPost, 0);

	acc = (pcl == LTE).mean()
	
	return acc

def k_fold_cross_validation(x, labels, k):
	num_samples = x.shape[1]
	indices = np.random.permutation(num_samples)
	fold_size = num_samples // k

	for i in range(k):
		fold_start = i * fold_size
		fold_end = (i + 1) * fold_size

		val_indices = indices[fold_start:fold_end]
		train_indices = np.concatenate([indices[:fold_start], indices[fold_end:]])

		x_train, labels_train = x[:,train_indices], labels[train_indices]
		x_val, labels_val = x[:,val_indices], labels[val_indices]

		# MVG
		# mus, Cs, wCs, CsDiag = calculateParameters(x_train, labels_train)
		# scoresMVG = np.append(scoresMVG, mvg(mus, Cs, x_val, labels_val))
		# scoresBayes = np.append(scoresBayes, mvg(mus, CsDiag, x_val, labels_val))
		# scoresTied = np.append(scoresTied, mvg(mus, wCs, x_val, labels_val))
		# naivetied is missing

		# SVM
		# logreg_obj = logreg_obj_wrap(x_train, labels_train, 1e-1)
		# x0 = np.zeros(x_train.shape[0] + 1)
		# xmin = bfgs(logreg_obj, x0, approx_grad=True, factr=1e9, maxiter=5*1e3)[0]
		# w, b = xmin[0:-1], xmin[-1]
		# S = np.dot(w, x_val) + b

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
    
def computeMinDCF(S, LTE, pi1, Cfn, Cfp):
    S_sorted = np.sort(S)
    Bdummy = np.min([pi1*Cfn,(1-pi1)*Cfp])
    
    DCFs = np.array([], dtype=float)

    for t in S_sorted:
        predictions = np.array((S>t), dtype=int)

        CM = np.zeros((2,2), dtype=int)

        for i in range(predictions.size):
            CM[predictions[i], LTE[i]] += 1

        FNR = CM[0,1]/(CM[0,1]+CM[1,1])
        FPR = CM[1,0]/(CM[0,0]+CM[1,0])
        DCF = (pi1*Cfn*FNR +(1-pi1)*Cfp*FPR)/Bdummy

        DCFs= np.append(DCFs, DCF)

    minDCF = np.min(DCFs)

    return minDCF

def computeActualDCF(llr, labels, pi1, Cfn, Cfp):
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

def bayesErrorPlots(eplo, llr, labels):
    DCFs = np.array([], dtype=float)
    minDCFs = np.array([], dtype=float)
    
    for p in eplo:
        ep = 1/(1+math.exp(-p))
        DCF = computeActualDCF(llr, labels, ep, 1,1)
        minDCF = computeMinDCF(llr, labels, ep, 1,1)

        DCFs= np.append(DCFs, DCF)
        minDCFs= np.append(minDCFs, minDCF)

    plt.figure()
    plt.plot(eplo, minDCFs, label='min DCF', color='b')
    plt.plot(eplo, DCFs, label='DCF', color='r')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()