# coding='utf-8'
from time import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from sklearn import datasets
from sklearn.manifold import TSNE

colors=list(mcolors.CSS4_COLORS.keys())
def plot_embedding(data, label, title):
    x_min, x_max = np.min(data, 0), np.max(data, 0)
    data = (data - x_min) / (x_max - x_min)

    fig = plt.figure()
    c = [mcolors.CSS4_COLORS[colors[label[i]]] for i in range(len(data))]
    plt.scatter(data[:, 0], data[:, 1], c=c)
    plt.xticks([])
    plt.yticks([])
    plt.title(title)
    plt.savefig("tsne.png", bbox_inches='tight')


def main():
    data = np.load("p_raw_data.npy")
    label = np.load("l_raw_data.npy")
    print('Computing t-SNE embedding')
    tsne = TSNE(n_components=2, init='pca', random_state=0)
    t0 = time()
    result = tsne.fit_transform(data)
    plot_embedding(result, label,
                         't-SNE embedding (time %.2fs)'
                         % (time() - t0))

if __name__ == '__main__':
    main()