from mllib import *

if __name__ == '__main__':
    DTR, LTR = load('../Train.txt')
    DTE, LTE = load('../Test.txt')

    P = PCA_directions(DTR, 7)
    DTR = np.dot(P.T, DTR)
    DTE = np.dot(P.T, DTE)

    Dc = centering(DTR)
    Ds = std_variances(Dc)
    Dw = whitening(Ds, DTR)
    Dl = l2(Dw)
    expD = expand_feature_space(Dl)

    DcT = centering(DTE)
    DsT = std_variances(DcT)
    DwT = whitening(DsT, DTE)
    DlT = l2(DwT)
    expDT = expand_feature_space(DlT)
    
    # lambda
    l = [1e-4]

    # threshold
    p = 1/11

    for li in l:
        x0 = np.zeros(expD.shape[0] + 1)
        x, f, d = sp.optimize.fmin_l_bfgs_b(logreg_obj_wrap(expD, LTR, li), x0)

        w, b = x[0:-1], x[-1]

        S = np.dot(w, expDT) + b

        pemp = LTR.sum()/LTR.shape[0]
        logOdds = np.log(pemp/(1-pemp))

        S -= logOdds

        print(f'for lambda={li}')
        print(DCF_min(p, 1, 1, S, LTE))
        print(DCF_actual(p, 1, 1, S, LTE))

    FNR, FPR = DET_curve(p, 1, 1, S, LTE)
    np.save('..\\data\\01_log-reg_eval_FNR.npy', FNR)
    np.save('..\\data\\01_log-reg_eval_FPR.npy', FPR)