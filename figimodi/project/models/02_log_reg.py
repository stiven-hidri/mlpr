from mllib import *

if __name__ == '__main__':
    D, L = load('../Train.txt')
    
    # DPCA9 = PCA(D, L, 9)
    # DPCA8 = PCA(D, L, 8)
    DPCA7 = PCA(D, L, 7)
    # DPCA6 = PCA(D, L, 6)

    Dc = centering(DPCA7)
    Ds = std_variances(Dc)
    Dw = whitening(Ds, DPCA7)
    Dl = l2(Dw)
    expD = expand_feature_space(Dl)

    # folds
    K = 10

    # lambda
    l = [1e-4]
    
    # threshold
    p = 0.9

    for li in l:
        logRatioCumulative = np.array([])
        cumulativeLabels = np.array([])

        for i in range(0, K):
            (DTR, LTR), (DTE, LTE) = k_fold(expD, L, K, i)

            # use maxfun=[>1500], maxiter[>30], factr=[<10**7] to increment precision
            x0 = np.zeros(DTR.shape[0] + 1)
<<<<<<< HEAD
            x, f, d = sp.optimize.fmin_l_bfgs_b(logreg_obj_wrap(DTR, LTR, li), x0)
=======
            x, f, d = sp.optimize.fmin_l_bfgs_b(logreg_obj_weight_wrap(DTR, LTR, li, 0.8), x0)
>>>>>>> 088c00a14d53137dd78d2539c07ff62f714b5d14

            w, b = x[0:-1], x[-1]
            S = np.dot(w, DTE) + b

            logRatioCumulative = np.append(logRatioCumulative, S)
            cumulativeLabels = np.append(cumulativeLabels, LTE)

        mindcf = DCF_min(p, 1, 1, logRatioCumulative, cumulativeLabels)
        actualdcf = DCF_actual(p, 1, 1, logRatioCumulative, cumulativeLabels)

        print(f"using lambda={li}")
        print(f"min dcf: {mindcf}")
        print(f"actual dcf: {actualdcf}")
        print("___________________________________")