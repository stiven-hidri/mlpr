import matplotlib.pyplot as plt 

if __name__ == '__main__':
    Kc0 = [1, 2, 4, 8, 16]
    Kc0prime = [4, 8]
    gmm_nopca_k1 = [0.33522540983606564, 0.2858606557377049, 0.25209016393442624, 0.2677049180327869, 0.31395491803278686]
    gmm_nopca_k2 = [0.34772540983606554, 0.29899590163934425, 0.2686680327868852, 0.2770901639344262, 0.30612704918032785]
    gmm_nopca_k4 = [0.3558401639344262, 0.3086475409836065, 0.2845901639344262, 0.28770491803278686, 0.32422131147540983]
    
    gmm_pca8_k1 = [0.26178278688524587, 0.26647540983606555]
    gmm_pca8_k2 = [0.25584016393442627, 0.26428278688524587]
    gmm_pca7_k1 = [0.24491803278688526, 0.25397540983606554]
    gmm_pca7_k2 = [0.25647540983606554, 0.24991803278688526]
    gmm_pca6_k1 = [0.2514754098360656, 0.25584016393442627]
    gmm_pca6_k2 = [0.25397540983606554, 0.25241803278688524]

    plt.figure()
    plt.plot(Kc0, gmm_nopca_k1, label='GMM - Kc1=1 (No PCA)')
    plt.plot(Kc0, gmm_nopca_k2, label='GMM - Kc1=2 (No PCA)')
    plt.plot(Kc0, gmm_nopca_k4, label='GMM - Kc1=4 (No PCA)')
    plt.xlabel('Kc0')
    plt.ylabel('minDCF')
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig('04_gmm\\gmm_nopca.png')
    plt.close()

    
    plt.figure()
    plt.plot(Kc0prime, gmm_nopca_k1[2:4], label='GMM - Kc1=1 (No PCA)', linewidth=3)
    plt.plot(Kc0prime, gmm_pca8_k1, label='GMM - Kc1=1 (PCA=8)')
    plt.plot(Kc0prime, gmm_pca7_k1, label='GMM - Kc1=1 (PCA=7)', linewidth=3, linestyle='--')
    plt.plot(Kc0prime, gmm_pca6_k1, label='GMM - Kc1=1 (PCA=6)')
    plt.xlabel('Kc0')
    plt.ylabel('minDCF')
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig('04_gmm\\gmm_pca_k1.png')
    plt.close()

    plt.figure()
    plt.plot(Kc0prime, gmm_nopca_k2[2:4], label='GMM - Kc1=2 (No PCA)', linewidth=3)
    plt.plot(Kc0prime, gmm_pca8_k2, label='GMM - Kc1=2 (PCA=8)')
    plt.plot(Kc0prime, gmm_pca7_k2, label='GMM - Kc1=2 (PCA=7)', linewidth=3, linestyle='--')
    plt.plot(Kc0prime, gmm_pca6_k2, label='GMM - Kc1=2 (PCA=6)')
    plt.xlabel('Kc0')
    plt.ylabel('minDCF')
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig('04_gmm\\gmm_pca_k2.png')
    plt.close()




