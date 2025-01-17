import matplotlib.pyplot as plt
doub = [1,2,4,8]
"""
#LINEAR
a12_1 = [ 0.28869047619047616, 0.27261904761904765, 0.26626984126984127, 0.2683035714285714 ] 
a12_2 = [ 0.12003968253968254, 0.10029761904761905, 0.0914021164021164, 0.0918154761904762 ] 
a12_3 = [ 0.3460317460317461, 0.2814484126984127, 0.2548280423280424, 0.2541170634920635 ]
a11_1 = [ 0.2869047619047619, 0.2586309523809524, 0.25277777777777777, 0.27232142857142855 ]
a11_2 = [ 0.11686507936507937, 0.10009920634920635, 0.09074074074074073, 0.09236111111111112 ] 
a11_3 = [ 0.3511904761904763, 0.27509920634920637, 0.25363756613756616, 0.2514384920634921 ]

plt.figure()
plt.title("GMM FULL")
plt.plot(doub, a12_1, label="pi = 0.1 - RAW", c="red")
plt.plot(doub, a12_2, label="pi = 0.5 - RAW", c="green")
plt.plot(doub, a12_3, label="pi = 0.9 - RAW", c="blue")
plt.plot(doub, a11_1, label="pi = 0.1 - PCA(11)", linestyle="dashed", c="red")
plt.plot(doub, a11_2, label="pi = 0.5 - PCA(11)", linestyle="dashed", c="green")
plt.plot(doub, a11_3, label="pi = 0.9 - PCA(11)", linestyle="dashed", c="blue")
plt.legend()
plt.savefig("./GMM_FULL") 

"""
"""
#tied

a12_1 = [ 0.29345238095238096, 0.29345238095238096, 0.2875, 0.27619047619047615 ]
a12_2 = [ 0.11825396825396825, 0.11825396825396825, 0.1041005291005291, 0.09573412698412698 ]
a12_3 = [ 0.357936507936508, 0.357936507936508, 0.31719576719576725, 0.292609126984127 ]

a11_1 = [ 0.2982142857142857, 0.2982142857142857, 0.28253968253968254, 0.2738095238095238 ]
a11_2 = [ 0.11984126984126985, 0.11984126984126985, 0.10456349206349205, 0.09742063492063492 ]
a11_3 = [ 0.34722222222222227, 0.34722222222222227, 0.3016534391534392, 0.2844246031746032 ]

plt.figure()
plt.title("GMM TIED")
plt.plot(doub, a12_1, label="pi = 0.1 - RAW", c="red")
plt.plot(doub, a12_2, label="pi = 0.5 - RAW", c="green")
plt.plot(doub, a12_3, label="pi = 0.9 - RAW", c="blue")
plt.plot(doub, a11_1, label="pi = 0.1 - PCA(11)", linestyle="dashed", c="red")
plt.plot(doub, a11_2, label="pi = 0.5 - PCA(11)", linestyle="dashed", c="green")
plt.plot(doub, a11_3, label="pi = 0.9 - PCA(11)", linestyle="dashed", c="blue")
plt.legend()
plt.savefig("./GMM_TIED")  
"""
a12_1 = [ 0.29523809523809524, 0.5133928571428571, 0.501984126984127, 0.48705357142857136 ]
a12_2 = [ 0.11646825396825397, 0.2228174603174603, 0.21388888888888888, 0.20813492063492064 ] 
a12_3 = [ 0.34801587301587306, 0.557936507936508, 0.541468253968254, 0.5279761904761905 ]
a11_1 = [ 0.27678571428571425, 0.33839285714285716, 0.3384920634920635, 0.34553571428571433 ] 
a11_2 = [ 0.11488095238095239, 0.12291666666666667, 0.12156084656084656, 0.12038690476190475 ] 
a11_3 = [ 0.3398809523809524, 0.39533730158730157, 0.36084656084656086, 0.3542162698412699 ]

plt.figure()
plt.title("GMM DIAGONAL")
plt.plot(doub, a12_1, label="pi = 0.1 - RAW", c="red")
plt.plot(doub, a12_2, label="pi = 0.5 - RAW", c="green")
plt.plot(doub, a12_3, label="pi = 0.9 - RAW", c="blue")
plt.plot(doub, a11_1, label="pi = 0.1 - PCA(11)", linestyle="dashed", c="red")
plt.plot(doub, a11_2, label="pi = 0.5 - PCA(11)", linestyle="dashed", c="green")
plt.plot(doub, a11_3, label="pi = 0.9 - PCA(11)", linestyle="dashed", c="blue")
plt.legend()
plt.savefig("./GMM_DIAGONAL") 
