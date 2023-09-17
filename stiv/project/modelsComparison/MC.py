from lib import * 

if __name__ == '__main__':



    effPriorLogOdds = np.linspace(-4, 4, 21)
    bayesErrorPlots(effPriorLogOdds, llr, labels)