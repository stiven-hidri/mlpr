from mllib import *

if __name__ == '__main__':
    (D, L) = load('../Train.txt')

    # Showing for each feature the distribution of this with respect to the two different classes
    for i in range(D.shape[0]):
        feature_plot_binary(i, D, L, ['spoofed', 'authentic'])

    # for i in range(D.shape[0]):
    #     for j in range(D.shape[0]):
    #         if i != j:
    #             feature_scatter_binary(i, j, D, L, ['spoofed', 'authentic'])

    PCA_plot(D, L)
    LDA_plot(D, L)

    PCA_data_variance(D)

    heatmaps_binary(D, L)
