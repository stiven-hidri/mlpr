import numpy
import ml
import matplotlib.pyplot as plt
import math

def computeROC(llr, LTE):
    S_sorted = numpy.sort(S)
    TPRs = numpy.array([])
    FPRs = numpy.array([])

    for t in S_sorted:
        predictions = numpy.array((S>t), dtype=int)

        CM = numpy.zeros((2,2), dtype=int)

        for i in range(predictions.size):
            CM[predictions[i], LTE[i]] += 1

        TPR = 1 - CM[0,1]/(CM[0,1]+CM[1,1])
        FPR = CM[1,0]/(CM[0,0]+CM[1,0])

        TPRs = numpy.append(TPRs, TPR)
        FPRs = numpy.append(FPRs, FPR)

    plt.figure()
    plt.plot(FPRs, TPRs)
    plt.title("ROC")
    plt.show()
    
def computeMinDCF(S, LTE, pi1, Cfn, Cfp):
    S_sorted = numpy.sort(S)
    Bdummy = numpy.min([pi1*Cfn,(1-pi1)*Cfp])
    
    DCFs = numpy.array([], dtype=float)

    for t in S_sorted:
        predictions = numpy.array((S>t), dtype=int)

        CM = numpy.zeros((2,2), dtype=int)

        for i in range(predictions.size):
            CM[predictions[i], LTE[i]] += 1

        FNR = CM[0,1]/(CM[0,1]+CM[1,1])
        FPR = CM[1,0]/(CM[0,0]+CM[1,0])
        DCF = (pi1*Cfn*FNR +(1-pi1)*Cfp*FPR)/Bdummy

        DCFs= numpy.append(DCFs, DCF)

    minDCF = numpy.min(DCFs)

    return minDCF

def computeActualDCF(llr, labels, pi1, Cfn, Cfp):
    CM = numpy.zeros((2,2), dtype=int)
    t  = -numpy.log((pi1*Cfn)/((1-pi1)*Cfp))
    predictions = numpy.array((llr>t), dtype=int)
    Bdummy = numpy.min([pi1*Cfn, (1-pi1)*Cfp])
    for i in range(predictions.size):
        CM[predictions[i], labels[i]] += 1

    FNR = CM[0,1]/(CM[0,1]+CM[1,1])
    FPR = CM[1,0]/(CM[0,0]+CM[1,0])
    DCF = (pi1*Cfn*FNR +(1-pi1)*Cfp*FPR)/Bdummy

    return DCF

def bayesErrorPlots(eplo, llr, labels):
    DCFs = numpy.array([], dtype=float)
    minDCFs = numpy.array([], dtype=float)
    
    for p in eplo:
        ep = 1/(1+math.exp(-p))
        DCF = computeActualDCF(llr, labels, ep, 1,1)
        minDCF = computeMinDCF(llr, labels, ep, 1,1)

        DCFs= numpy.append(DCFs, DCF)
        minDCFs= numpy.append(minDCFs, minDCF)

    plt.figure()
    plt.plot(eplo, minDCFs, label='min DCF', color='b')
    plt.plot(eplo, DCFs, label='DCF', color='r')
    plt.ylim([0, 1.1])
    plt.xlim([-3, 3])
    plt.show()

if __name__ == '__main__':
    llr = numpy.load('commedia_llr_infpar.npy')
    labels = numpy.load('commedia_labels_infpar.npy')

    effPriorLogOdds = numpy.linspace(-3, 3, 21)
    bayesErrorPlots(effPriorLogOdds, llr, labels)




    
            

    
    
