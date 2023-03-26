import pickle

import numpy as np
import torch
import matplotlib.pyplot as plt
import umap


def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)
class UMAPLatent:
    def __init__(self):
        self.umap = umap.UMAP()
    def visualize(self,X,classes,N, prototype_matrix=None):
        trans = self.umap.fit(X.detach().numpy())
        X = trans.embedding_
        x_min, x_max = np.min(X, 0), np.max(X, 0)
        X = (X - x_min) / (x_max - x_min)

        plt.figure()
        colors = ["red","blue","green","yellow","purple"]
        ax = plt.subplot(111)
        cmap = get_cmap(6)
        for i in range(X.shape[0]):
            classif = classes[i]
            plt.plot(X[i, 0], X[i, 1], 'o', color=colors[int(classif.item())], alpha=0.5 if classif!=4 else 1.0)
        if prototype_matrix is not None:
          prototype_embedding = trans.transform(prototype_matrix)
          for i in range(len(prototype_embedding)):
            prototype = prototype_embedding[i] / (x_max - x_min)
            plt.scatter(prototype[0], prototype[1], marker="*", c="pink", alpha=1.0)


