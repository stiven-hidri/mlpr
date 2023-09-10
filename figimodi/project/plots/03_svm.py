import matplotlib.pyplot as plt 

if __name__ == '__main__':
    x = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    lin_nopca = [1.0, 0.9239959016393442, 0.7420286885245901, 0.5104508196721311, 0.47307377049180327, 0.4721311475409837, 0.4946311475409836, 0.97125]
    lin_nopca_z = [1.0, 0.5661270491803279, 0.4980327868852459, 0.47305327868852454, 0.4721311475409837, 0.4721311475409837, 0.4746106557377049, 0.46838114754098364]
    lin_nopca_zwl = [1.0, 0.41743852459016395, 0.4089344262295082, 0.3814549180327869, 0.38114754098360654, 0.37862704918032786, 0.37895491803278686, 0.3796106557377049]
    lin_pca7 = [1.0, 0.929344262295082, 0.7676024590163935, 0.5204918032786885, 0.4758811475409836, 0.4674385245901639, 0.4948975409836066, 0.9865573770491805]
    lin_pca7_z = [1.0, 0.4686680327868853, 0.4771311475409836, 0.4755532786885246, 0.4649385245901639, 0.4674385245901639, 0.47305327868852454, 0.473688524590164]
    lin_pca7_zwl = [1.0, 0.3942622950819672, 0.4011270491803278, 0.3861680327868853, 0.3848975409836065, 0.37930327868852454, 0.38241803278688524, 0.38116803278688527]

    kern_nopca_p2_c1 = [1.0, 0.31739754098360656, 0.3020901639344262, 0.3005327868852459, 0.31704918032786883, 0.5029713114754099, 1.0, 0.9962499999999999]
    kern_nopca_p2_c1_z = [1.0, 0.3061475409836065, 0.29114754098360657, 0.2777254098360656, 0.2752254098360656, 0.30213114754098364, 0.3052459016393443, 0.31491803278688524]
    kern_nopca_p2_c1_zwl = [1.0, 0.3958811475409836, 0.3845696721311475, 0.34516393442622956, 0.29430327868852457, 0.2911270491803279, 0.3048975409836065, 0.31459016393442624]
    
    kern_pca7_p2_c1 = [1.0, 0.30676229508196723, 0.2989754098360655, 0.30086065573770493, 0.2846106557377049, 0.688217213114754, 0.99875, 1.0]
    kern_pca7_p2_c1_z = [1.0, 0.36959016393442623, 0.2989549180327869, 0.26678278688524587, 0.2555532786885246, 0.27834016393442623, 0.27864754098360656, 0.2773975409836065]
    kern_pca7_p2_c1_zwl = [1.0, 0.3658606557377049, 0.36805327868852455, 0.3333196721311475, 0.2821311475409836, 0.2508811475409836, 0.27807377049180326, 0.2774590163934426]

    kern_nopca_p3_c1 = [1.0, 0.590983606557377, 0.7116188524590165, 0.686577868852459, 0.6887295081967214, 0.6803073770491803, 0.6803073770491803, 0.6803073770491803]
    kern_nopca_p3_c1_z = [1.0, 0.30643442622950817, 0.29084016393442624, 0.35516393442622957, 0.4313934426229509, 0.5619467213114754, 0.6544467213114753, 0.6544467213114753]
    kern_nopca_p3_c1_zwl = [1.0, 0.3880122950819672, 0.3529713114754099, 0.3064754098360656, 0.28616803278688524, 0.29555327868852455, 0.33834016393442623, 0.39606557377049184]
    kern_pca7_p3_c1 = [1.0, 0.39547131147540987, 0.3907991803278688, 0.8031967213114753, 1.0, 0.9940573770491805, 1.0, 0.99875]
    kern_pca7_p3_c1_z = [1.0, 0.34327868852459015, 0.29551229508196725, 0.2993237704918033, 0.3726844262295082, 0.4076024590163934, 0.44702868852459016, 0.4301024590163934]
    kern_pca7_p3_c1_zwl = [1.0, 0.3345901639344262, 0.3392622950819672, 0.28864754098360657, 0.2639959016393443, 0.2805532786885246, 0.3461065573770492, 0.3661270491803279]

    C = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100]
    kern_nopca_rbf_g5 = [0.4948975409836066, 0.4727254098360656, 0.45495901639344266]
    kern_nopca_rbf_g4 = [1.0, 0.99375, 0.99375, 0.5411475409836065, 0.45586065573770496, 0.4202254098360656, 0.3836475409836066, 0.28799180327868856]
    kern_nopca_rbf_g3 = [1.0, 0.7789754098360656, 0.7173975409836066, 0.3857991803278688, 0.3333196721311475, 0.29334016393442625, 0.27711065573770494, 0.3048770491803279]
    kern_nopca_rbf_g2 = [1.0, 0.5786065573770491, 0.5786065573770491, 0.4582172131147541, 0.3820081967213115, 0.3464139344262295, 0.41172131147540986, 0.4035655737704918]
    kern_nopca_rbf_g1 = [1.0, 0.9768647540983606, 0.9768647540983606, 0.7952254098360655, 0.7930327868852459, 0.8505532786885247, 0.7614344262295083, 0.7614344262295083]
    kern_nopca_rbf_g05 = [1.0, 0.9865573770491805, 0.9865573770491805, 0.9865573770491805, 0.9865573770491805, 0.9865573770491805, 0.9865573770491805, 0.9865573770491805]

    Cprime = [1, 10, 100]
    kern_nopca_rbf_g5_z = [0.9606147540983606, 0.5036270491803279, 0.48678278688524584]
    kern_nopca_rbf_g4_z = [0.5945286885245901, 0.4786680327868853, 0.4733811475409836]
    kern_nopca_rbf_g3_z = [0.46524590163934426, 0.44805327868852457, 0.4064549180327869]
    kern_nopca_rbf_g5_zwl = [0.9468647540983606, 0.9303073770491803, 0.41614754098360646]
    kern_nopca_rbf_g4_zwl = [0.3995696721311475, 0.3873975409836065, 0.387110655737705]
    kern_nopca_rbf_g3_zwl = [0.3839549180327869, 0.3777049180327869, 0.38930327868852455]

    kern_pca7_rbf_g5 = [0.5055532786885246, 0.4733606557377049, 0.4464959016393443]
    kern_pca7_rbf_g4 = [0.41680327868852457, 0.3802663934426229, 0.2974180327868853]
    kern_pca7_rbf_g3 = [0.2792827868852459, 0.2833606557377049, 0.2843032786885246]
    kern_pca7_rbf_g5_z = [0.92875, 0.8148360655737704, 0.4852254098360656]
    kern_pca7_rbf_g4_z = [0.5033196721311475, 0.4818032786885245, 0.46461065573770494]
    kern_pca7_rbf_g3_z = [0.4724180327868852, 0.4583401639344262, 0.4530737704918033]
    kern_pca7_rbf_g5_zwl = [0.8843442622950819, 0.49151639344262293, 0.4836065573770492]
    kern_pca7_rbf_g4_zwl = [0.39993852459016394, 0.3989549180327869, 0.39801229508196717]
    kern_pca7_rbf_g3_zwl = [0.39866803278688523, 0.37055327868852456, 0.3805327868852459]

    kern_pca7_p2_zwl_weighted = [0.28274590163934427, 0.27834016393442623, 0.2720901639344262, 0.27118852459016396]

    plt.figure()
    plt.plot(x, lin_nopca, label='SVM (No PCA)')
    plt.plot(x, lin_nopca_z, label='SVM (No PCA, z-norm)')
    plt.plot(x, lin_nopca_zwl, label='SVM (No PCA, z-norm + whitening + l2-norm)')
    plt.plot(x, lin_pca7, label='SVM (PCA = 7)', linestyle='--')
    plt.plot(x, lin_pca7_z, label='SVM (PCA = 7, z-norm)', linestyle='--')
    plt.plot(x, lin_pca7_zwl, label='SVM (PCA = 7, z-norm + whitening + l2-norm)', linestyle='--')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend(loc='upper center')
    plt.grid()
    plt.savefig('03_svm\\linear_svm.png')
    plt.close()

    plt.figure()
    plt.plot(x, kern_nopca_p2_c1, label='SVM - poly(2) (No PCA)')
    plt.plot(x, kern_nopca_p2_c1_z, label='SVM - poly(2) (No PCA, z-norm)')
    plt.plot(x, kern_nopca_p2_c1_zwl, label='SVM - poly(2) (No PCA, z-norm + whitening + l2-norm)')
    plt.plot(x, kern_pca7_p2_c1, label='SVM - poly(2) (PCA=7)', linestyle='--')
    plt.plot(x, kern_pca7_p2_c1_z, label='SVM - poly(2) (PCA=7, z-norm)', linestyle='--')
    plt.plot(x, kern_pca7_p2_c1_zwl, label='SVM - poly(2) (PCA=7, z-norm + whitening + l2-norm)', linestyle='--')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend(loc='upper center')
    plt.grid()
    plt.savefig('03_svm\\kern_svm_poly2.png')
    plt.close()

    plt.figure()
    plt.plot(x, kern_nopca_p3_c1, label='SVM - poly(3) (No PCA)')
    plt.plot(x, kern_nopca_p3_c1_z, label='SVM - poly(3) (No PCA, z-norm)')
    plt.plot(x, kern_nopca_p3_c1_zwl, label='SVM - poly(3) (No PCA, z-norm + whitening + l2-norm)')
    plt.plot(x, kern_pca7_p3_c1, label='SVM - poly(3) (PCA=7)', linestyle='--')
    plt.plot(x, kern_pca7_p3_c1_z, label='SVM - poly(3) (PCA=7, z-norm)', linestyle='--')
    plt.plot(x, kern_pca7_p3_c1_zwl, label='SVM - poly(3) (PCA=7, z-norm + whitening + l2-norm)', linestyle='--')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend(loc='upper center')
    plt.grid()
    plt.savefig('03_svm\\kern_svm_poly3.png')
    plt.close()

    plt.figure()
    plt.plot(C, kern_nopca_rbf_g4, label='SVM - RBF (No PCA, log(γ)=-4)')
    plt.plot(C, kern_nopca_rbf_g3, label='SVM - RBF (No PCA, log(γ)=-3)')
    plt.plot(C, kern_nopca_rbf_g2, label='SVM - RBF (No PCA, log(γ)=-2)')
    plt.plot(C, kern_nopca_rbf_g1, label='SVM - RBF (No PCA, log(γ)=-1)')
    plt.plot(C, kern_nopca_rbf_g05, label='SVM - RBF (No PCA, γ=0.5)')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend(loc="lower left")
    plt.grid()
    plt.savefig('03_svm\\kern_rbf_nopca.png')
    plt.close()

    plt.figure()
    plt.plot(Cprime, kern_nopca_rbf_g5, label='SVM - RBF (log(γ)=-5)')
    plt.plot(Cprime, kern_nopca_rbf_g4[5:], label='SVM - RBF (log(γ)=-4)')
    plt.plot(Cprime, kern_nopca_rbf_g3[5:], label='SVM - RBF (log(γ)=-3)')
    plt.plot(Cprime, kern_nopca_rbf_g5_z, label='SVM - RBF (z-norm, log(γ)=-5)', linestyle='--')
    plt.plot(Cprime, kern_nopca_rbf_g4_z, label='SVM - RBF (z-norm, log(γ)=-4)', linestyle='--')
    plt.plot(Cprime, kern_nopca_rbf_g3_z, label='SVM - RBF (z-norm, log(γ)=-3)', linestyle='--')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig('03_svm\\kern_rbf_nopca_preproc1.png')
    plt.close()

    plt.figure()
    plt.plot(Cprime, kern_nopca_rbf_g5, label='SVM - RBF (log(γ)=-5)')
    plt.plot(Cprime, kern_nopca_rbf_g4[5:], label='SVM - RBF (log(γ)=-4)')
    plt.plot(Cprime, kern_nopca_rbf_g3[5:], label='SVM - RBF (log(γ)=-3)')
    plt.plot(Cprime, kern_nopca_rbf_g5_zwl, label='SVM - RBF (z-norm + whitening + l2-norm, log(γ)=-5)', linestyle='--')
    plt.plot(Cprime, kern_nopca_rbf_g4_zwl, label='SVM - RBF (z-norm + whitening + l2-norm, log(γ)=-4)', linestyle='--')
    plt.plot(Cprime, kern_nopca_rbf_g3_zwl, label='SVM - RBF (z-norm + whitening + l2-norm, log(γ)=-3)', linestyle='--')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend(loc="upper right")
    plt.grid()
    plt.savefig('03_svm\\kern_rbf_nopca_preproc2.png')
    plt.close()

    plt.figure()    
    plt.plot(Cprime, kern_pca7_rbf_g5, label='SVM - RBF (log(γ)=-5)')
    plt.plot(Cprime, kern_pca7_rbf_g4, label='SVM - RBF (log(γ)=-4)')
    plt.plot(Cprime, kern_pca7_rbf_g3, label='SVM - RBF (log(γ)=-3)')
    plt.plot(Cprime, kern_pca7_rbf_g5_z, label='SVM - RBF (z-norm, log(γ)=-5)', linestyle='--')
    plt.plot(Cprime, kern_pca7_rbf_g4_z, label='SVM - RBF (z-norm, log(γ)=-4)', linestyle='--')
    plt.plot(Cprime, kern_pca7_rbf_g3_z, label='SVM - RBF (z-norm, log(γ)=-3)', linestyle='--')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig('03_svm\\kern_rbf_pca7_preproc1.png')
    plt.close()

    plt.figure()    
    plt.plot(Cprime, kern_pca7_rbf_g5, label='SVM - RBF (log(γ)=-5)')
    plt.plot(Cprime, kern_pca7_rbf_g4, label='SVM - RBF (log(γ)=-4)')
    plt.plot(Cprime, kern_pca7_rbf_g3, label='SVM - RBF (log(γ)=-3)')
    plt.plot(Cprime, kern_pca7_rbf_g5_zwl, label='SVM - RBF (z-norm + whitening + l2-norm, log(γ)=-5)', linestyle='--')
    plt.plot(Cprime, kern_pca7_rbf_g4_zwl, label='SVM - RBF (z-norm + whitening + l2-norm, log(γ)=-4)', linestyle='--')
    plt.plot(Cprime, kern_pca7_rbf_g3_zwl, label='SVM - RBF (z-norm + whitening + l2-norm, log(γ)=-3)', linestyle='--')
    plt.xscale('log')
    plt.xlabel('C')
    plt.ylabel('minDCF')
    plt.legend(loc='upper right')
    plt.grid()
    plt.savefig('03_svm\\kern_rbf_pca7_preproc2.png')
    plt.close()



