from GMM_load import *
from mllib import *

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
        [gmm, ll] = LBG(D, gmm)

    return gmm, ll

if __name__ == '__main__':
    D = np.load('GMM_data_4D.npy')
    # D = np.load('GMM_data_1D.npy')
    gmm = load_gmm('GMM_4D_3G_init.json')
    solutionInit = np.load('GMM_4d_3G_init_ll.npy')
    solutionEM = load_gmm('GMM_4D_3G_EM.json')
    solutionLBG = load_gmm('GMM_4D_4G_EM_LBG.json')

    [gmm, ll] = EM(D, gmm)
    print(ll)

    [gmm, ll] = LBG_wrap(D, n_iter=2)

    print(ll)
