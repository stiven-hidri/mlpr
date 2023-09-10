from ml import pca, lda
import numpy as np

def main():
    sepls = []
    sepws = []
    petls = []
    petws = []
    labels = []

    f=open("../sources/iris.csv", "r")

    for line in f:
        parts=line.split(",")

        #sepal_length, sepal_width, petal_length, petal_width, family
        sepls.append(float(parts[0]))
        sepws.append(float(parts[1]))
        petls.append(float(parts[2]))
        petws.append(float(parts[3]))
        labels.append(parts[4].strip())

    table = np.array([sepls, sepws, petls, petws], dtype=float)
    classes = np.array(labels, dtype=str)

    pca(table, classes, 2, True)
    lda(table, classes, 2, True)

if __name__ == '__main__':
    main()
