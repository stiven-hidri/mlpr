import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as llg

def row2col(row):
    return row.reshape(row.size, 1)

def col2row(col):
    return col.reshape(1, col.size)

colors=["green","red","blue"]

N = 150
NC = 50
m = 2

### READ FILE ###

sepls = []
sepws = []
petls = []
petws = []
labels = []

f=open("../sources/iris.csv", "r")

for line in f:
    parts=line.split(",")

    #sepal_length, sepal_width, petal_length, petal_width, family
    sepls.append(float(parts[0]))
    sepws.append(float(parts[1]))
    petls.append(float(parts[2]))
    petws.append(float(parts[3]))
    labels.append(parts[4].strip())

table = np.array([sepls, sepws, petls, petws], dtype=float)
classes = np.array(labels, dtype=str)

### PCA ###

#THESE COMPUTATIONS ARE CORRECT BUT ARE BASED ON LOOPS AND STUFF -> SLOW
#TRY TO USE FUNCTIONS OF THE NUMPY LIBRARY WHICH ARE WRITTE IN C AND THUS ARE FASTER

# mu = 0
# for i in range(table.shape[1]):
#     mu = mu + table[:, i:i+1] #add all the column vectors

# mu = mu / float(table.shape[1]) #computer mean of each attribute

# C = 0

# for i in range(table.shape[1]):
#     C = C + ((table[:, i:i+1]) - mu)@(table[:, i:i+1] - mu).T  #data covariance matrix. @: dot product

# C = C / float(table.shape[1])

mean = table.mean(1) #now this is a 1-D array. ocho!
centeredTable = table - row2col(mean)
covMatrix = 1/N*(np.dot(centeredTable, centeredTable.T))

#GET EIGENVECTORS AND EIGENVALUES AND THEN TAKE THE m LEADING EIGENVECTORS
#s, U = np.linalg.eigh(covMatrix)    .eig() finds eigenvalues but since is symmetrix we use .eigh() which is more performant
#                                    eigh returns into s the eigenvalues from smallest to largest, .eig() sorts una bella cippa di niente
#P = U[:, ::-1][:, 0:m]  reverse and then take the firs m columns

#WE CAN ACHIEVE THE SAME THING VIA SINGULAR VALUE DECOMPOSITION
U, _, _ = np.linalg.svd(covMatrix) #since the singular values correspond to the eigenvalues (the covMatrix is semidefinite)
P=U[:, 0:m]
P[:,1]*=-1  #just to get same orientation as professor

DP=np.dot(P.T, table)

### PLOT ###
for i, x in enumerate(classes):
    plt.scatter(DP[0,classes[:]==x], DP[1,classes[:]==x], color=colors[int(i/50)])

plt.show();

### LDA ###

withinCovarianceMatrix = np.zeros(covMatrix.shape, dtype = float)
betweenCovarianceMatrix = np.zeros(covMatrix.shape, dtype = float)

for x in classes:
    classData = table[:,classes[:]==x]
    meanClass = classData.mean(1) #now this is a 1-D array. ocho!
    centerdClass = classData - row2col(meanClass)
    withinCovarianceMatrix += np.dot(centerdClass, centerdClass.T)/NC
    
    betweenCovarianceMatrix += np.dot(row2col(meanClass)-row2col(mean), (row2col(meanClass)-row2col(mean)).T)

withinCovarianceMatrix/=N
betweenCovarianceMatrix/=N

### I don't know what this is ###
#s, U = llg.eigh(betweenCovarianceMatrix, withinCovarianceMatrix) #solves the generalized eigenvalue problem: Sb*w = λ*Sw*w
#W=U[:, ::-1][:,0:m]  #take m eigenvectors corresponging to the m greatest eigenvalues
#UW, _, _ = np.linalg.svd(W)     #singular value decomposition
#U = UW[0,0:m]       #basis spanned by W

U, s, _ = np.linalg.svd(withinCovarianceMatrix)
P1 = np.dot(U * col2row(1.0/(s**0.5)), U.T) #We find Pw through whitening transformation
# or P1 = numpy.dot( numpy.dot(U, numpy.diag(1.0/(s**0.5))), U.T )

Sbt = np.dot(np.dot(P1, betweenCovarianceMatrix), P1.T)

s, U = np.linalg.eigh(Sbt)
P2 = U[:, ::-1][:, 0:m]
W = np.dot(P1.T, P2)
W[:,0]*=-1 
y = np.dot(W.T, table)

for i, x in enumerate(classes):
    plt.scatter(y[0,classes[:]==x], y[1,classes[:]==x], color=colors[int(i/50)])

plt.show();