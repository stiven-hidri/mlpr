import numpy as np
import matplotlib.pyplot as plt

classes = {'Iris-setosa': 0, 'Iris-versicolor': 1, 'Iris-virginica': 2}
attributes = {'sepalLength': 0, 'sepalWidth': 1, 'petalLength': 2, 'petalWidth': 3}

def load(fileName):
    mat = np.zeros(4*150).reshape((4, 150))
    labels = np.zeros(150)
    with open(fileName, 'r') as f:
        for i, line in enumerate(f):
            sepalLength, sepalWidth, ptealLength, petalWidth, label = line.rstrip().split(',')
            mat[:, i] = [sepalLength, sepalWidth, ptealLength, petalWidth]
            labels[i] = classes[label]
            
    return mat, labels

def stats(attribute1, mat, labels):
    fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(10,10))
    fig.suptitle(f'Stats for attribute: {attribute1}')
    
    vettAttr = mat[attributes[attribute1], :]

    for keyClass, valueClass in classes.items():
        mask = (labels == valueClass)
        data = vettAttr[mask]
        axs[0, 0].hist(data, bins = 10, density=True, alpha=0.3, label=f"{keyClass}")
    
    axs[0, 0].set_xlabel(f'{attribute1}')
    axs[0, 0].set_title(f'histogram: {attribute1}')
    axs[0, 0].legend()
    #set title

    i = 1
    for keyAttr, attribute2 in attributes.items():
        if keyAttr != attribute1:
            for keyClass, valueClass in classes.items():
                mask = (labels == valueClass)
                data1 = mat[attributes[attribute1], mask]
                data2 = mat[attribute2, mask]
                axs[i//2, i%2].scatter(data1, data2, label=f'{keyClass}')
            axs[i//2, i%2].set_xlabel(f'{attribute1}')
            axs[i//2, i%2].set_ylabel(f'{keyAttr}')
            axs[i//2, i%2].set_title(f'({attribute1}, {keyAttr})')
            axs[i//2, i%2].legend()
            #set title
            i += 1

    return

if __name__ == '__main__' :
    fileName = 'iris.csv'
    mat, labels = load(fileName)
    
    for attribute in attributes.keys():
        stats(attribute, mat, labels)

    plt.show()