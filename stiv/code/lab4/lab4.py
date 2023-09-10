import matplotlib.pyplot as plt
import matplotlib.colors as colors
from ml import *

plt.figure()

X1D = np.load("lab4/sources/X1D.npy")

[M, N] = np.shape(X1D)
mu = np.array(X1D.mean())
xc = X1D-mu
C = 1/N*(np.dot(xc, xc.T))

# ravel() transform a 2D array of shape (1, N) into a 1D vector (N, )
# bins=x set how many columns the histogram have
# density = true => the sum of the histograms is normalized to 1

plt.hist(X1D.ravel(), bins=50, density=True)
XPlot = np.linspace(X1D.min(), X1D.max(), 1000)
plt.plot(XPlot.ravel(), np.exp(logpdf_GAU_ND(vrow(XPlot), mu, C)))

plt.show()