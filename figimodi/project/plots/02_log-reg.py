import matplotlib.pyplot as plt 

if __name__ == '__main__':
    x = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10]
    lin_nopca = [0.4836885245901639, 0.4836885245901639, 0.4836885245901639, 0.4836885245901639, 0.4821311475409836, 0.481188524590164, 0.5393032786885246]
    lin_nopca_z = [0.4836885245901639, 0.4836885245901639, 0.4836885245901639, 0.48243852459016395, 0.5080532786885246, 0.5564549180327869, 0.6883811475409836]
    lin_pca7 = [0.46961065573770494, 0.46961065573770494, 0.46961065573770494, 0.46961065573770494, 0.4746106557377049, 0.4824180327868852, 0.5330327868852459]
    lin_pca7_z = [0.46961065573770494, 0.46961065573770494, 0.46836065573770497, 0.4589959016393443, 0.4674180327868852, 0.485860655737705, 0.7385860655737705]

    quad_nopca = [0.2924180327868852, 0.29366803278688525, 0.29366803278688525, 0.28991803278688527, 0.29303278688524587, 0.31614754098360653, 0.3314139344262295]
    quad_pca9 = [0.2842622950819672, 0.28301229508196724, 0.28680327868852457, 0.2864754098360655, 0.2880327868852459, 0.3214344262295082, 0.33360655737704914]
    quad_pca8 = [0.28397540983606556, 0.2864754098360655, 0.28522540983606554, 0.28489754098360653, 0.2864549180327869, 0.3164344262295082, 0.33799180327868855]
    quad_pca7 = [0.2655532786885246, 0.26680327868852455, 0.2643032786885246, 0.2618032786885246, 0.28428278688524594, 0.32112704918032786, 0.3311065573770492]
    quad_pca6 = [0.2689959016393443, 0.2689959016393443, 0.2689959016393443, 0.26961065573770493, 0.2889549180327869, 0.3104918032786885, 0.32735655737704916]

    quad_pca7_z = [0.2643032786885246, 0.2643032786885246, 0.2593032786885246, 0.2636680327868853, 0.30645491803278685, 0.3402254098360656, 0.4333196721311475]
    quad_pca7_zw = [0.26680327868852455, 0.2655532786885246, 0.2643032786885246, 0.26680327868852455, 0.27366803278688523, 0.3024180327868852, 0.46209016393442626]
    quad_pca7_zwl = [0.26743852459016393, 0.25213114754098365, 0.27713114754098356, 0.3280327868852459, 0.3527459016393443, 0.4052254098360656, 0.9256147540983607]
    quad_pca7_cwl = [0.2908196721311475, 0.27084016393442617, 0.30522540983606555, 0.34676229508196715, 0.4086270491803279, 0.46081967213114755, 0.916844262295082]

    quad_pca7_z_weight = [0.26961065573770493, 0.27241803278688526, 0.2618032786885246, 0.26491803278688525, 0.39836065573770485, 0.5574180327868853, 0.5586680327868853]
    quad_pca7_zw_weight = [0.3083606557377049, 0.3083606557377049, 0.3083606557377049, 0.31366803278688526, 0.2755532786885246, 0.31614754098360653, 0.5113934426229507]
    quad_pca7_zwl_weight = [0.2574385245901639, 0.25618852459016395, 0.2955737704918033, 0.32897540983606555, 0.35209016393442627, 0.40647540983606556, 0.9081147540983607]

    plt.figure()
    plt.plot(x, lin_nopca, label='Log-Reg (No PCA)')
    plt.plot(x, lin_nopca_z, label='Log-Reg (z-norm)')
    plt.plot(x, lin_pca7, label='Log-Reg (PCA=7)', linestyle='--', linewidth=2)
    plt.plot(x, lin_pca7_z, label='Log-Reg (PCA=7, z-norm)', linestyle='--', linewidth=2)
    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('minDCF')
    plt.legend()
    plt.grid()
    plt.savefig('02_log-reg\\linear_logreg.png')
    plt.close

    plt.figure()
    plt.plot(x, quad_nopca, label='Q-Log-Reg (No PCA)')
    plt.plot(x, quad_pca9, label='Q-Log-Reg (PCA=9)')
    plt.plot(x, quad_pca8, label='Q-Log-Reg (PCA=8)')
    plt.plot(x, quad_pca7, label='Q-Log-Reg (PCA=7)')
    plt.plot(x, quad_pca6, label='Q-Log-Reg (PCA=6)')
    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('minDCF')
    plt.legend()
    plt.grid()
    plt.savefig('02_log-reg\\quad_diffPCA.png')
    plt.close()

    plt.figure()
    plt.plot(x, quad_pca7, label='Q-Log-Reg', linewidth=3)
    plt.plot(x, quad_pca7_z, label='Q-Log-Reg (z-norm)')
    plt.plot(x, quad_pca7_zw, label='Q-Log-Reg (z-norm, whitening)')
    plt.plot(x, quad_pca7_zwl, label='Q-Log-Reg (z-norm, whitening, l2-norm)')
    plt.plot(x, quad_pca7_cwl, label='Q-Log-Reg (centering, whitnening, l2-norm)')
    plt.ylim(0.2, 0.6)
    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('minDCF')
    plt.legend()
    plt.grid()
    plt.savefig('02_log-reg\\quad_preproc.png')
    plt.close()

    plt.figure()
    plt.plot(x, quad_pca7_z, label='Q-Log-Reg (z-norm)')
    plt.plot(x, quad_pca7_zw, label='Q-Log-Reg (z-norm, whitening)')
    plt.plot(x, quad_pca7_zwl, label='Q-Log-Reg (z-norm, whitening, l2-norm)')
    plt.plot(x, quad_pca7_z_weight, label='Q-Log-Reg (z-norm) prior weighted', linestyle='--')
    plt.plot(x, quad_pca7_zw_weight, label='Q-Log-Reg (z-norm, whitening) prior weighted', linestyle='--')
    plt.plot(x, quad_pca7_zwl_weight, label='Q-Log-Reg (z-norm, whitnening, l2-norm) prior weighted', linestyle='--')
    plt.ylim(0.2, 0.6)
    plt.xscale('log')
    plt.xlabel('λ')
    plt.ylabel('minDCF')
    plt.legend()
    plt.grid()
    plt.savefig('02_log-reg\\quad_priorweight.png')
    plt.close()

