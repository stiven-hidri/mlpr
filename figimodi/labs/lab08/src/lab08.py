import sklearn.datasets
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
import scipy.special
import itertools
import sys
import string


def mcol(v):
    return v.reshape((v.size, 1))

def load_data():

    lInf = []

    f=open('data/inferno.txt', encoding="ISO-8859-1")

    for line in f:
        lInf.append(line.strip())
    f.close()

    lPur = []

    f=open('data/purgatorio.txt', encoding="ISO-8859-1")

    for line in f:
        lPur.append(line.strip())
    f.close()

    lPar = []

    f=open('data/paradiso.txt', encoding="ISO-8859-1")

    for line in f:
        lPar.append(line.strip())
    f.close()
    
    return lInf, lPur, lPar

def split_data(l, n):

    lTrain, lTest = [], []
    for i in range(len(l)):
        if i % n == 0:
            lTest.append(l[i])
        else:
            lTrain.append(l[i])
            
    return lTrain, lTest

### Solution 1 - Dictionaries of frequencies ###

def S1_buildDictionary(lTercets):

    '''
    Create a set of all words contained in the list of tercets lTercets
    lTercets is a list of tercets (list of strings)
    '''

    sDict = set([])
    for s in lTercets:
        words = s.split()
        for w in words:
            sDict.add(w)
    return sDict

def S1_estimateModel(hlTercets, eps = 0.1):

    '''
    Build frequency dictionaries for each class.

    hlTercets: dict whose keys are the classes, and the values are the list of tercets of each class.
    eps: smoothing factor (pseudo-count)

    Return: dictionary h_clsLogProb whose keys are the classes. For each class, h_clsLogProb[cls] is a dictionary whose keys are words and values are the corresponding log-frequencies (model parameters for class cls)
    '''

    # Build the set of all words appearing at least once in each class
    sDictCommon = set([])

    for cls in hlTercets: # Loop over class labels
        lTercets = hlTercets[cls]
        sDictCls = S1_buildDictionary(lTercets)
        sDictCommon = sDictCommon.union(sDictCls)

    # Initialize the counts of words for each class with eps
    h_clsLogProb = {}
    for cls in hlTercets: # Loop over class labels
        h_clsLogProb[cls] = {w: eps for w in sDictCommon} # Create a dictionary for each class that contains all words as keys and the pseudo-count as initial values

    # Estimate counts
    for cls in hlTercets: # Loop over class labels
        lTercets = hlTercets[cls]
        for tercet in lTercets: # Loop over all tercets of the class
            words = tercet.split()
            for w in words: # Loop over words of the given tercet
                h_clsLogProb[cls][w] += 1
            
    # Compute frequencies
    for cls in hlTercets: # Loop over class labels
        nWordsCls = sum(h_clsLogProb[cls].values()) # Get all occurrencies of words in cls and sum them. this is the number of words (including pseudo-counts)
        for w in h_clsLogProb[cls]: # Loop over all words
            h_clsLogProb[cls][w] = np.log(h_clsLogProb[cls][w]) - np.log(nWordsCls) # Compute log N_{cls,w} / N

    return h_clsLogProb

def S1_compute_logLikelihoods(h_clsLogProb, text):

    '''
    Compute the array of log-likelihoods for each class for the given text
    h_clsLogProb is the dictionary of model parameters as returned by S1_estimateModel
    The function returns a dictionary of class-conditional log-likelihoods
    '''
    
    logLikelihoodCls = {cls: 0 for cls in h_clsLogProb}
    for cls in h_clsLogProb: # Loop over classes
        for word in text.split(): # Loop over words
            if word in h_clsLogProb[cls]:
                logLikelihoodCls[cls] += h_clsLogProb[cls][word]
    return logLikelihoodCls

def S1_compute_logLikelihoodMatrix(h_clsLogProb, lTercets, hCls2Idx = None):

    '''
    Compute the matrix of class-conditional log-likelihoods for each class each tercet in lTercets

    h_clsLogProb is the dictionary of model parameters as returned by S1_estimateModel
    lTercets is a list of tercets (list of strings)
    hCls2Idx: map between textual labels (keys of h_clsLogProb) and matrix rows. If not provided, automatic mapping based on alphabetical oreder is used
   
    Returns a #cls x #tercets matrix. Each row corresponds to a class.
    '''
    
    if hCls2Idx is None:
        hCls2Idx = {cls:idx for idx, cls in enumerate(sorted(h_clsLogProb))}

    S = np.zeros((len(h_clsLogProb), len(lTercets)))
    for tIdx, tercet in enumerate(lTercets):
        hScores = S1_compute_logLikelihoods(h_clsLogProb, tercet)
        for cls in h_clsLogProb: # We sort the class labels so that rows are ordered according to alphabetical order of labels
            clsIdx = hCls2Idx[cls]
            S[clsIdx, tIdx] = hScores[cls]

    return S

### Solution 2 - Arrays of occurrencies ###

def S2_buildDictionary(lTercets):

    '''
    Create a dictionary of all words contained in the list of tercets lTercets
    The dictionary allows storing the words, and mapping each word to an index i (the corresponding index in the array of occurrencies)

    lTercets is a list of tercets (list of strings)
    '''

    hDict = {}
    nWords = 0
    for tercet in lTercets:
        words = tercet.split()
        for w in words:
            if w not in hDict:
                hDict[w] = nWords
                nWords += 1
    return hDict

def S2_estimateModel(hlTercets, eps = 0.1):

    '''
    Build word log-probability vectors for all classes

    hlTercets: dict whose keys are the classes, and the values are the list of tercets of each class.
    eps: smoothing factor (pseudo-count)

    Return: tuple (h_clsLogProb, h_wordDict). h_clsLogProb is a dictionary whose keys are the classes. For each class, h_clsLogProb[cls] is an array containing, in position i, the log-frequency of the word whose index is i. h_wordDict is a dictionary that maps each word to its corresponding index.
    '''

    # Since the dictionary also includes mappings from word to indices it's more practical to build a single dict directly from the complete set of tercets, rather than doing it incrementally as we did in Solution S1
    lTercetsAll = list(itertools.chain(*hlTercets.values())) 
    hWordDict = S2_buildDictionary(lTercetsAll)
    nWords = len(hWordDict) # Total number of words

    h_clsLogProb = {}
    for cls in hlTercets:
        h_clsLogProb[cls] = np.zeros(nWords) + eps # In this case we use 1-dimensional vectors for the model parameters. We will reshape them later.
    
    # Estimate counts
    for cls in hlTercets: # Loop over class labels
        lTercets = hlTercets[cls]
        for tercet in lTercets: # Loop over all tercets of the class
            words = tercet.split()
            for w in words: # Loop over words of the given tercet
                wordIdx = hWordDict[w]
                h_clsLogProb[cls][wordIdx] += 1 # h_clsLogProb[cls] ius a 1-D array, h_clsLogProb[cls][wordIdx] is the element in position wordIdx

    # Compute frequencies
    for cls in h_clsLogProb.keys(): # Loop over class labels
        vOccurrencies = h_clsLogProb[cls]
        vFrequencies = vOccurrencies / vOccurrencies.sum()
        vLogProbabilities = np.log(vFrequencies)
        h_clsLogProb[cls] = vLogProbabilities

    return h_clsLogProb, hWordDict
    
def S2_tercet2occurrencies(tercet, hWordDict):
    
    '''
    Convert a tercet in a (column) vector of word occurrencies. Word indices are given by hWordDict
    '''
    v = np.zeros(len(hWordDict))
    for w in tercet.split():
        if w in hWordDict: # We discard words that are not in the dictionary
            v[hWordDict[w]] += 1
    return mcol(v)

def S2_compute_logLikelihoodMatrix(h_clsLogProb, hWordDict, lTercets, hCls2Idx = None):

    '''
    Compute the matrix of class-conditional log-likelihoods for each class each tercet in lTercets

    h_clsLogProb and hWordDict are the dictionary of model parameters and word indices as returned by S2_estimateModel
    lTercets is a list of tercets (list of strings)
    hCls2Idx: map between textual labels (keys of h_clsLogProb) and matrix rows. If not provided, automatic mapping based on alphabetical oreder is used
   
    Returns a #cls x #tercets matrix. Each row corresponds to a class.
    '''

    if hCls2Idx is None:
        hCls2Idx = {cls:idx for idx, cls in enumerate(sorted(h_clsLogProb))}
    
    numClasses = len(h_clsLogProb)
    numWords = len(hWordDict)

    # We build the matrix of model parameters. Each row contains the model parameters for a class (the row index is given from hCls2Idx)
    MParameters = np.zeros((numClasses, numWords)) 
    for cls in h_clsLogProb:
        clsIdx = hCls2Idx[cls]
        MParameters[clsIdx, :] = h_clsLogProb[cls] # MParameters[clsIdx, :] is a 1-dimensional view that corresponds to the row clsIdx, we can assign to the row directly the values of another 1-dimensional array

    SList = []
    for tercet in lTercets:
        v = S2_tercet2occurrencies(tercet, hWordDict)
        STercet = np.dot(MParameters, v) # The log-lieklihoods for the tercets can be computed as a matrix-vector product. Each row of the resulting column vector corresponds to M_c v = sum_j v_j log p_c,j
        SList.append(np.dot(MParameters, v))

    S = np.hstack(SList)
    return S


################################################################################

def compute_classPosteriors(S, logPrior = None):

    '''
    Compute class posterior probabilities

    S: Matrix of class-conditional log-likelihoods
    logPrior: array with class prior probability (shape (#cls, ) or (#cls, 1)). If None, uniform priors will be used

    Returns: matrix of class posterior probabilities
    '''

    if logPrior is None:
        logPrior = np.log( np.ones(S.shape[0]) / float(S.shape[0]) )
    J = S + mcol(logPrior) # Compute joint probability
    ll = scipy.special.logsumexp(J, axis = 0) # Compute marginal likelihood log f(x)
    P = J - ll # Compute posterior log-probabilities P = log ( f(x, c) / f(x)) = log f(x, c) - log f(x)
    return np.exp(P)

def compute_accuracy(P, L):

    '''
    Compute accuracy for posterior probabilities P and labels L. L is the integer associated to the correct label (in alphabetical order)
    '''

    PredictedLabel = np.argmax(P, axis=0)
    NCorrect = (PredictedLabel.ravel() == L.ravel()).sum()
    NTotal = L.size
    return float(NCorrect)/float(NTotal)

def vcol(mat):
    return mat.reshape((mat.size, 1)) 

def vrow(mat):
    return mat.reshape((1, mat.size))

def load_iris():
    D, L = sklearn.datasets.load_iris()['data'].T, sklearn.datasets.load_iris()['target']
    return D, L

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

def compute_mu_C(D, L, label):
    DL = D[:, L == label]
    mu = DL.mean(1).reshape(DL.shape[0], 1)
    DLC = (DL - mu)
    C = 1/DLC.shape[1]*np.dot(DLC, DLC.T)
    return (mu, C)

def compute_mu_C_NB(D, L, label):
    DL = D[:, L == label]
    mu = DL.mean(1).reshape(DL.shape[0], 1)
    DLC = (DL - mu)
    C = np.multiply(1/DLC.shape[1]*np.dot(DLC, DLC.T), np.identity(DL.shape[0]))
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

def opt_bayes(prior, Cfn, Cfp, s_log_ratio):

    t = -np.log((prior * Cfn)/((1 - prior) * Cfp))
    c = s_log_ratio > t
    
    return c

def bayes_risk(prior, Cfn, Cfp, s_log_ratio, labels):

    c = opt_bayes(prior, Cfn, Cfp, s_log_ratio)

    CMD = np.zeros((2, 2), dtype=int)

    for i, p in enumerate(c):
        CMD[int(p), int(labels[i])] += 1

    FNR = CMD[0, 1]/(CMD[0, 1] + CMD[1, 1])
    FPR = CMD[1, 0]/(CMD[0, 0] + CMD[1, 0])

    DCF = prior*Cfn*FNR+(1-prior)*Cfp*FPR

    return DCF

def normalized_bayes_risk(prior, Cfn, Cfp, s_log_ratio, labels):

    print(f"calulating dcf for pi = {prior}")

    c = opt_bayes(prior, Cfn, Cfp, s_log_ratio)

    CMD = np.zeros((2, 2), dtype=int)

    for i, p in enumerate(c):
        CMD[int(p), int(labels[i])] += 1

    FNR = CMD[0, 1]/(CMD[0, 1] + CMD[1, 1])
    FPR = CMD[1, 0]/(CMD[0, 0] + CMD[1, 0])

    DCF = prior*Cfn*FNR+(1-prior)*Cfp*FPR

    Bdummy = np.min([prior * Cfn, (1 - prior) * Cfp])
    return DCF / Bdummy

def DCF_min(prior, Cfn, Cfp, s_log_ratio, labels):
    
    print(f"calulating dcf min for pi = {prior}")

    Bdummy = np.min([prior * Cfn, (1 - prior) * Cfp])
    DCF = np.array([])

    for t in s_log_ratio:
        print(f"analysing threshold {t} for min dcf")
        c = s_log_ratio > t
        CMD = np.zeros((2, 2), dtype=int)

        for i, p in enumerate(c):
            print(f"computing sample number {i} for confusion matrix")
            CMD[int(p), int(labels[i])] += 1

        FNR = CMD[0, 1]/(CMD[0, 1] + CMD[1, 1])
        FPR = CMD[1, 0]/(CMD[0, 0] + CMD[1, 0])

        DCF = np.append(DCF, (prior*Cfn*FNR+(1-prior)*Cfp*FPR)/Bdummy)

    return np.min(DCF)

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

    plt.plot(FPR, TPR)
    plt.show()


def bayer_error_plots(prior, Cfn, Cfp, s_log_ratio, labels):
    effPriorLogOdds = np.linspace(-3, 3, 21)

    dcf = np.array([])
    mindcf = np.array([])

    effective_prior = 1/(np.exp(-effPriorLogOdds) + 1)

    print("ciao")

    for i, p in enumerate(effective_prior):
        print(f"calculating dcf number {i}")
        dcf = np.append(dcf, normalized_bayes_risk(p, 1, 1, s_log_ratio, labels))

    for i, p in enumerate(effective_prior):
        print(f"calculating min dcf number {i}")
        mindcf = np.append(mindcf, DCF_min(p, 1, 1, s_log_ratio, labels))

    plt.plot(effPriorLogOdds, dcf, label='DCF', color='r')
    plt.plot(effPriorLogOdds, mindcf, label='min DCF', color='b')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])

    plt.show()

    return

if  __name__  == '__main__':
    D, L = load_iris()
    (DTR, LTR), (DTE, LTE) = split_db_2to1(D, L)

    # MVG
    # compute mean and covariance for all classes
    (mu0, C0) = compute_mu_C(DTR, LTR, 0)
    (mu1, C1) = compute_mu_C(DTR, LTR, 1)
    (mu2, C2) = compute_mu_C(DTR, LTR, 2)

    # Naive-Bayes
    # compute mean and covariance for all classes
    # (mu0, C0) = compute_mu_C_NB(DTR, LTR, 0)
    # (mu1, C1) = compute_mu_C_NB(DTR, LTR, 1)
    # (mu2, C2) = compute_mu_C_NB(DTR, LTR, 2)

    # Tied-Covariance
    # C0 = C1 = C2 = 1/DTR.shape[1]*(C0*(LTR == 0).sum() + C1*(LTR == 1).sum() + C2*(LTR == 2).sum())

    # compute score matrix S of shape [3, 50], which is the number of classes times the number of samples in the test set
    S0 = logpdf_GAU_ND(DTE, mu0, C0)
    S1 = logpdf_GAU_ND(DTE, mu1, C1)
    S2 = logpdf_GAU_ND(DTE, mu2, C2)

    # f_c|x
    S = np.vstack([S0, S1, S2])
    
    # working with exp
    # S = np.exp(S)

    # # f_x|c
    # SJoint = 1/3*S
    # SMarginal = vrow(SJoint.sum(0))
    # SPost = SJoint/SMarginal

    # working with logs
    logSJoint = S + np.log(1/3)
    logSMarginal = vrow(sp.special.logsumexp(logSJoint, axis=0))
    logSPost = logSJoint - logSMarginal
    SPost = np.exp(logSPost)

    PL = np.argmax(SPost, 0)

    CM = np.zeros((3,3), dtype=int)
    
    for i, p in enumerate(PL):
        CM[p, LTE[i]] += 1

    print(CM)


    # INFERNO PARADISO PURGATORIO
    lInf, lPur, lPar = load_data()

    lInfTrain, lInfEval = split_data(lInf, 4)
    lPurTrain, lPurEval = split_data(lPur, 4)
    lParTrain, lParEval = split_data(lPar, 4)


    ### Solution 1 ###
    ### Multiclass ###

    # hCls2Idx = {'inferno': 0, 'purgatorio': 1, 'paradiso': 2}

    # hlTercetsTrain = {
    #     'inferno': lInfTrain,
    #     'purgatorio': lPurTrain,
    #     'paradiso': lParTrain
    #     }

    # lTercetsEval = lInfEval + lPurEval + lParEval

    # S1_model = S1_estimateModel(hlTercetsTrain, eps = 0.001)

    # S1_predictions = compute_classPosteriors(
    #     S1_compute_logLikelihoodMatrix(
    #         S1_model,
    #         lTercetsEval,
    #         hCls2Idx,
    #         ),
    #     np.log(np.array([1./3., 1./3., 1./3.]))
    #     )

    # labelsInf = np.zeros(len(lInfEval))
    # labelsInf[:] = hCls2Idx['inferno']

    # labelsPar = np.zeros(len(lParEval))
    # labelsPar[:] = hCls2Idx['paradiso']

    # labelsPur = np.zeros(len(lPurEval))
    # labelsPur[:] = hCls2Idx['purgatorio']

    # labelsEval = np.hstack([labelsInf, labelsPur, labelsPar])

    # # Per-class accuracy
    # print('Multiclass - S1 - Inferno - Accuracy:', compute_accuracy(S1_predictions[:, labelsEval==hCls2Idx['inferno']], labelsEval[labelsEval==hCls2Idx['inferno']]))
    # print('Multiclass - S1 - Purgatorio - Accuracy:', compute_accuracy(S1_predictions[:, labelsEval==hCls2Idx['purgatorio']], labelsEval[labelsEval==hCls2Idx['purgatorio']]))
    # print('Multiclass - S1 - Paradiso - Accuracy:', compute_accuracy(S1_predictions[:, labelsEval==hCls2Idx['paradiso']], labelsEval[labelsEval==hCls2Idx['paradiso']]))

    # # Overall accuracy
    # print('Multiclass - S1 - Accuracy:', compute_accuracy(S1_predictions, labelsEval))

    hCls2Idx = {'inferno': 0, 'paradiso': 1}

    hlTercetsTrain = {
        'inferno': lInfTrain,
        'paradiso': lParTrain
        }

    lTercetsEval = lInfEval + lParEval

    S1_model = S1_estimateModel(hlTercetsTrain, eps = 0.001)

    S1_logLikelihood = S1_compute_logLikelihoodMatrix(
            S1_model,
            lTercetsEval,
            hCls2Idx,
            )

    llr_infpar = np.load('commedia_llr_infpar.npy')
    labels = np.load('commedia_labels_infpar.npy')

    br = bayes_risk(0.8, 1, 10, llr_infpar, labels)

    print(br)

    nbr = normalized_bayes_risk(0.8, 1, 10, llr_infpar, labels)

    print(nbr)

    dcfmin = DCF_min(0.5, 1, 1, llr_infpar, labels)

    print(dcfmin)

    #ROC_curve(0.5, 1, 1, llr_infpar, labels)
    
    bayer_error_plots(0.5, 1, 1, llr_infpar, labels)
