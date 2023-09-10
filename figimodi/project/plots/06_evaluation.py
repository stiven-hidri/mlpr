import matplotlib.pyplot as plt 

if __name__ == '__main__':
    l = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    log_training = [0.26743852459016393, 0.251, 0.27713114754098356, 0.3280327868852459, 0.3527459016393443, 0.4052254098360656, 0.9256147540983607]
    log_evaluation = [0.2673623680241327, 0.2594871794871795, 0.25709276018099547, 0.26946644042232276, 0.3002469834087481, 0.3098717948717949, 0.31237179487179484]

    c = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    svm_pca7_training = [1.0, 0.3658606557377049, 0.36805327868852455, 0.3333196721311475, 0.2821311475409836, 0.248, 0.27807377049180326, 0.2774590163934426]
    svm_pca7_evaluation = [1.0, 0.3247473604826546, 0.3075603318250377, 0.26918552036199095, 0.25183069381598794, 0.2519136500754148, 0.2700810708898944, 0.27510180995475114]
    svm_pca6_evaluation = [1.0, 1.0, 0.305039592760181, 0.26805995475113126, 0.253142911010558, 0.2500075414781297, 0.2553205128205128, 0.25363310708898945]
    svm_pca8_evaluation = [1.0, 0.3269966063348416, 0.30984162895927597, 0.2724981146304676, 0.257539592760181, 0.25589366515837103, 0.26297699849170436, 0.281487556561086]

    Kc0 = [4, 8]
    gmm_1_training = [0.24491803278688526, 0.25397540983606554]
    gmm_2_training = [0.25647540983606554, 0.24991803278688526]
    gmm_1_evaluation = [0.2171021870286576, 0.2179147812971342]
    gmm_2_evaluation = [0.2106127450980392, 0.21872737556561084]

    plt.figure()
    plt.plot(l, log_training, label='Q-Log-Reg (training)')
    plt.plot(l, log_evaluation, label='Q-Log-Reg (evaluation)')
    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('minDCF')
    plt.legend()
    plt.ylim([0.2, 0.4])
    plt.grid()
    plt.savefig('06_evaluation\\log_train_vs_eval.png')
    plt.close

    plt.figure()
    plt.plot(c, svm_pca7_training, label='SVM Poly(2) (training)')
    plt.plot(c, svm_pca7_evaluation, label='SVM Poly(2) (evaluation)')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend()
    plt.ylim([0.2, 0.4])
    plt.grid()
    plt.savefig('06_evaluation\\svm_train_vs_eval.png')
    plt.close

    plt.figure()
    plt.plot(c, svm_pca8_evaluation, label='SVM Poly(2) (evaluation PCA=8)')
    plt.plot(c, svm_pca7_evaluation, label='SVM Poly(2) (evaluation PCA=7)')
    plt.plot(c, svm_pca6_evaluation, label='SVM Poly(2) (evaluation PCA=6)')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend()
    plt.ylim([0.2, 0.4])
    plt.grid()
    plt.savefig('06_evaluation\\svm_train_vs_eval_pcas.png')
    plt.close

    plt.figure()
    plt.plot(Kc0, gmm_1_training, label='GMM Kc1=1 (training)')
    plt.plot(Kc0, gmm_2_training, label='GMM Kc1=2 (training)')
    plt.plot(Kc0, gmm_1_evaluation, label='GMM Kc1=1 (evaluation)')
    plt.plot(Kc0, gmm_2_evaluation, label='GMM Kc1=2 (evaluation)')
    plt.xlabel('Kc0')
    plt.ylabel('minDCF')
    plt.legend()
    plt.grid()
    plt.savefig('06_evaluation\\gmm_train_vs_eval.png')
    plt.close
    
