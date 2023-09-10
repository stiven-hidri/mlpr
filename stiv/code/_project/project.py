import matplotlib.pyplot as plt
from scipy.optimize import fmin_l_bfgs_b as bfgs
import os.path as path
import numpy as np
from scipy import special as sp

colors = ['#2ef527b3','#f5272799']
etichetta = ["Same speaker", "Different speakers"]

def vcol(row): 
  return row.reshape(row.size, 1)

def vrow(col): 
  return col.reshape(1, col.size)

def readTrainAndTestData():
	DTR = np.array([], float)
	LTR = np.array([], int)
	DTE = np.array([], float)
	LTE = np.array([], int)

	if not path.exists("DTR.npy") or not path.exists("LTR.npy") or not path.exists("DTE.npy") or not path.exists("LTE.npy"):
		f = open("Train.txt", "r")
		
		for line in f:
			parts = line.split(",")
			LTR = np.append(LTR, [int(parts.pop(-1))], 0)
			if DTR.size > 0:
				DTR = np.column_stack((DTR, vcol(np.array([[float(field) for field in parts]], dtype=float))))
			else:
				DTR = np.append(DTR, vcol(np.array([[float(field) for field in parts]], dtype=float)))

		f.close()

		np.save("DTR.npy", DTR)
		np.save("LTR.npy", LTR)

		f = open("Test.txt", "r")
		
		for line in f:
			parts = line.split(",")
			LTE = np.append(LTE, [int(parts.pop(-1))], 0)
			if DTE.size > 0:
				DTE = np.column_stack((DTE, vcol(np.array([[float(field) for field in parts]], dtype=float))))
			else:
				DTE = np.append(DTE, vcol(np.array([[float(field) for field in parts]], dtype=float)))

		f.close()

		np.save("DTE.npy", DTE)
		np.save("LTE.npy", LTE)

	else:
		DTR = np.load("DTR.npy", allow_pickle=True)
		LTR = np.load("LTR.npy", allow_pickle=True)
		DTE = np.load("DTE.npy", allow_pickle=True)
		LTE = np.load("LTE.npy", allow_pickle=True)

	return (DTR, LTR), (DTE, LTE)

def statistics(data, labels):
	k = 1
	for a in data:
		plt.figure()
		plt.hist(a[labels==0], density=True, label=etichetta[0], color=colors[0])
		plt.hist(a[labels==1], density=True, label=etichetta[1], color=colors[1])
		plt.savefig("plots/singleAttributes/" + str(k)) 
		plt.close()
		k+=1

	plt.close('all')

	k = 1
	for i in range(data.shape[0]):
		for j in range(i+1, data.shape[0]):
			plt.figure()
			plt.scatter(data[i, labels==0], data[j, labels==0], c=colors[0])
			plt.scatter(data[i, labels==1], data[j, labels==1], c=colors[1])
			plt.xlabel(str(i+1))
			plt.ylabel(str(j+1))
			plt.savefig("plots/pairedAttributes/" + str(k)) 
			plt.close()
			k+=1

	k = 1
	for i in range(5):
		plt.figure()
		plt.scatter(data[i, labels==0], data[i+5, labels==0], c=colors[0])
		plt.scatter(data[i, labels==1], data[i+5, labels==1], c=colors[1])
		plt.xlabel(str(i+1))
		plt.ylabel(str(j+1))
		plt.savefig("plots/beta/" + str(k)) 
		plt.close()
		k+=1

	plt.close('all')

def pca(x, labels, m):
	u = x.mean(1)
	N = x.shape[1]
	xc = x - vcol(u)
	C = 1/N*(np.dot(xc, xc.T))
	U = np.linalg.svd(C)[0]
	P=U[:, 0:m]
	#P[:,1]*=-1
	DP=np.dot(P.T, x)
	
	if m==2:
		plt.figure()
		for l in labels:
			plt.scatter(DP[0,labels[:]==l], DP[1,labels[:]==l], color=colors[l])
		plt.show()
              
	return DP

def lda(x, labels, m):
	mu = x.mean(1)
	N = x.shape[1]
	xc = x - vcol(mu)
	C = 1/N*(np.dot(xc, xc.T))

	wC = np.zeros(C.shape, dtype = float)
	bC = np.zeros(C.shape, dtype = float)

	for l in labels:
		xl = x[:,labels[:]==l]
		Nl = xl.shape[1]
		mul = xl.mean(1) #now this is a 1-D array. ocho!
		xcl = xl - vcol(mul)
		Cl = np.dot(xcl, xcl.T)/xl.shape[1]
		bC += np.dot(vcol(mul)-vcol(mu), (vcol(mul)-vcol(mu)).T)
		wC+=Cl*xl.shape[1]

	wC/=N
	bC/=N

	U, s, _ = np.linalg.svd(wC)
	P1 = np.dot(U * vrow(1.0/(s**0.5)), U.T) #We find Pw through whitening transformation
	# or P1 = numpy.dot( numpy.dot(U, numpy.diag(1.0/(s**0.5))), U.T )

	Sbt = np.dot(np.dot(P1, bC), P1.T)

	s, U = np.linalg.eigh(Sbt)
	P2 = U[:, ::-1][:, 0:m]
	W = np.dot(P1.T, P2)
	#W[:,0]*=-1 
	y = np.dot(W.T, x)

	if m==2:
		plt.figure()
		for l in labels:
			plt.scatter(y[0,labels[:]==l], y[1,labels[:]==l], color=colors[l])
		plt.show()

	return y

def calculateParameters(x, labels):
		M, N = x.shape;
		lunique = np.unique(labels)
		mus = np.array([]);
		Cs = np.array([]);
		wC = np.zeros((M,M))

		for l in lunique:
			xl = x[:,labels[:]==l]
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

def logpdf_GAU_ND(x, mu, C):
	M = C.shape[0] #number of features
	xc = np.array(x)

	for i in range(x.shape[1]):
		xc[:,i] -= mu

	log_det_C = np.linalg.slogdet(C)[1]

	log_N = -0.5*M*np.log(np.pi*2) - 0.5*log_det_C - 0.5*np.multiply(np.dot(xc.T, np.linalg.inv(C)), xc.T).sum(1)

	return log_N

def statx100(value):
	return str(round(value.mean(0)*100, 2)) + "%"

def mvg(mus, Cs, DTE, LTE):
	S = np.array([])

	for i in range(Cs.shape[0]):
		Si = logpdf_GAU_ND(DTE, mus[:,i], Cs[i])
		S = np.stack((S, Si)) if S.size>0 else Si

	Pc = 0.1;

	## work with logarithms
	logSJoint = S + np.log(Pc)

	logSMarginal = vrow(sp.logsumexp(S, axis=0))

	logSPost = logSJoint-logSMarginal
	SPost = np.exp(logSPost)

	pcl = np.argmax(SPost, 0);

	acc = (pcl == LTE).mean()
	
	return acc

def k_fold_cross_validation(x, labels, k):
	scoresMVG=np.array([])
	scoresBayes=np.array([])
	scoresTied=np.array([])
	scoresLR=np.array([])
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

		mus, Cs, wCs, CsDiag = calculateParameters(x_train, labels_train)

		scoresMVG = np.append(scoresMVG, mvg(mus, Cs, x_val, labels_val))
		scoresBayes = np.append(scoresBayes, mvg(mus, CsDiag, x_val, labels_val))
		scoresTied = np.append(scoresTied, mvg(mus, wCs, x_val, labels_val))

		logreg_obj = logreg_obj_wrap(x_train, labels_train, 1e-1)

		x0 = np.zeros(x_train.shape[0] + 1)
		xmin, f, d  = bfgs(logreg_obj, x0, approx_grad=True, factr=1e9, maxiter=5*1e3)

		w, b = xmin[0:-1], xmin[-1]
		S = np.dot(w, x_val) + b

		scoresLR = np.append(scoresLR, (((S > 0) == labels_val).mean()))

		# Print the performance metrics for the current fold
	
	print("MultiVariateGaussian: " + statx100(scoresMVG))
	print("MultiVariateGaussian Bayes: " + statx100(scoresBayes))
	print("MultiVariateGaussian Tied: " + statx100(scoresTied))
	print("Logistic Regression: " + statx100(scoresLR))

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
	(DTR, LTR), (DTE, LTE) = readTrainAndTestData()
	k_fold_cross_validation(DTR, LTR, 2)

if __name__ == '__main__':
  main()