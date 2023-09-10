
from mllib import *

if __name__ == '__main__':
    DTR, LTR = load('../Train.txt')
    DTE, LTE = load('../Test.txt')

    P = PCA_directions(DTR, 7)
    DTR = np.dot(P.T, DTR)
    DTE = np.dot(P.T, DTE)

    Kc1 = [1]
    Kc0 = [4]
 
    # effective prior
    p = 1/11

    print(p)
    for Kc1i in Kc1:
        for Kc0i in Kc0:
            DTR0 = DTR[:, LTR == 0]
            DTR1 = DTR[:, LTR == 1]

            gmm0, _ = LBG_wrap(DTR0, n_iter=int(np.log2(Kc0i) + 1))
            gmm1, _ = LBG_wrap(DTR1, n_iter=int(np.log2(Kc1i) + 1))

            S0 = logpdf_GMM(DTE, gmm0)
            S1 = logpdf_GMM(DTE, gmm1)

            S = S1 - S0

            print(f'for Kc1={Kc1i}, Kc0={Kc0i}')
            print(f'min {DCF_min(p, 1, 1, S, LTE)}')
            print(f'actual {DCF_actual(p, 1, 1, S, LTE)}')
    
    # print(DCF_actual(p, 1, 1, S, LTE))

    FNR, FPR = DET_curve(p, 1, 1, S, LTE)
    np.save('..\\data\\03_gmm_eval_FNR.npy', FNR)
    np.save('..\\data\\03_gmm_eval_FPR.npy', FPR)