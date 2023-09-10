from lib import *;

def main():
    (DTR, LTR), (DTE, LTE) = readTrainAndTestData()
    #statistics(DTR, LTR) DONE
    #pca(DTR, LTR, 2, 1)
    #lda(DTR, LTR, 1, 1)
    heatmaps_binary(DTR,LTR)
    


if __name__ == '__main__':
    main()