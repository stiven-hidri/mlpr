from mllib import *

if __name__ == '__main__':
    D, L = load('../Train.txt')

    # DPCA9 = PCA(D, L, 9)
    # DPCA8 = PCA(D, L, 8)
    DPCA7 = PCA(D, L, 7)
    # DPCA6 = PCA(D, L, 6)

    # Dc = centering(DPCA7)
    # Ds = std_variances(Dc)
    # Dw = whitening(Ds, DPCA7)
    # Dl = l2(Dw)

    Kc1s = [1]
    Kc0s = [4]

    # folds
    K = 10
 
    # effective prior
    p = 1/11

    for Kc1 in Kc1s:
        for Kc0 in Kc0s:

            logRatioCumulative = np.array([])
            cumulativeLabels = np.array([])

            for i in range(0, K):
                (DTR, LTR), (DTE, LTE) = k_fold(DPCA7, L, K, i)

                DTR0 = DTR[:, LTR == 0]
                DTR1 = DTR[:, LTR == 1]

                gmm0, _ = LBG_wrap(DTR0, n_iter=int(np.log2(Kc0) + 1))
                gmm1, _ = LBG_wrap(DTR1, n_iter=int(np.log2(Kc1) + 1))

                S0 = logpdf_GMM(DTE, gmm0)
                S1 = logpdf_GMM(DTE, gmm1)

                S = S1 - S0

                logRatioCumulative = np.append(logRatioCumulative, S)
                cumulativeLabels = np.append(cumulativeLabels, LTE)

            mindcf = DCF_min(p, 1, 1, logRatioCumulative, cumulativeLabels)
<<<<<<< HEAD
            actualdcf = DCF_actual(p, 1, 1, logRatioCumulative, cumulativeLabels)

            print(f"using Kc1={Kc1}, Kc0={Kc0}")
            print(f"min dcf: {mindcf}")
            print(f"actual dcf: {actualdcf}")
=======
            actualDCF = DCF_actual(p, 1, 1, logRatioCumulative, cumulativeLabels)

            print(f"using Kc1={Kc1}, Kc0={Kc0}")
            print(f"min dcf: {mindcf}")
            print(f"actual dcf: {actualDCF}")
>>>>>>> 088c00a14d53137dd78d2539c07ff62f714b5d14
            print("___________________________________")
 

    
