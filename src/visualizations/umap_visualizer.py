import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import umap

from ..data.data import get_ds
from ..models.single_variables import SingleVariableModulesWrapper

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
        # colors = plt.cm.rainbow(np.linspace(0, 1, N+1))
        colors = ["red", "blue", "yellow", "green", "magenta"]
        for i in range(X.shape[0]):
            classif = classes[i]
            plt.plot(X[i, 0], X[i, 1], marker='o'if classif!=N else '*', color=colors[int(classif.item())] if classif!=N else 'magenta', alpha=0.3 if classif!=N else 1.0,
                     markeredgecolor="none" if classif!=N else "k")
        if prototype_matrix is not None:
          prototype_embedding = trans.transform(prototype_matrix)
          for i in range(len(prototype_embedding)):
            prototype = prototype_embedding[i] / (x_max - x_min)
            plt.scatter(prototype[0], prototype[1], marker="*", c="pink", alpha=1.0)

def sv_prototypes_visualization_basicmotions():
    class_to_index={"standing":0, "running":1, "walking":2,"badminton":3}
    train_ds, test_ds = get_ds("data/basicmotions/BasicMotions_TRAIN.ts", class_to_index), get_ds("data/basicmotions/BasicMotions_TEST.ts", class_to_index)

    sv_modules_wrapper = SingleVariableModulesWrapper(6, 4, 10, 4)
    sv_modules_wrapper.load_state_dict(torch.load("models/basicmotions/sv_modules_wrapper.dat"))

    fig, axes = plt.subplots(2, 3)
    colors = plt.cm.rainbow(np.linspace(0,1,5))

    standing = Line2D([], [], color="white", marker='o', markerfacecolor=colors[0], markersize=10.0)
    running = Line2D([], [], color="white", marker='o', markerfacecolor=colors[1], markersize=10.0)
    walking = Line2D([], [], color="white", marker='o', markerfacecolor=colors[2], markersize=10.0)
    badminton = Line2D([], [], color="white", marker='o', markerfacecolor=colors[3], markersize=10.0)
    prototype = Line2D([], [], color="white", marker='*', markerfacecolor=colors[4], markeredgecolor="black", markeredgewidth=0.5, markersize=10.0)
    leg = fig.legend(handles=[standing, running, walking, badminton, prototype],
               labels=["Standing", "Running", "Walking", "Badminton", "Prototype"], loc="lower center", ncol=5)
    leg.get_frame().set_edgecolor('k')

    visualize_moment = torch.utils.data.DataLoader(test_ds, len(test_ds), True)
    for test_sample in visualize_moment:
        inp, out = test_sample[0].detach(), test_sample[1].detach()
        for i in range(6):
            embeddings = sv_modules_wrapper.single_variable_modules[i].encoder(inp[:,:,i].unsqueeze(2).float())
            embeddings = torch.concat([embeddings, sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix], dim=0)
            out = torch.concat([out, 4*torch.ones((sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix.shape[0],))], dim=0)
            visualizer = UMAPLatent()
            trans = visualizer.umap.fit(embeddings.detach().numpy())
            X = trans.embedding_
            x_min, x_max = np.min(X, 0), np.max(X, 0)
            X = (X - x_min) / (x_max - x_min)
            ax = axes[(i // 3)][(i % 3)]
            ax.set_xlabel("UMAP-1", fontsize=7.5)
            ax.set_ylabel("UMAP-2", fontsize=7.5)
            ax.set_title("Variable " + str(i+1), y=0.0, pad=-35, fontsize=10.0)
            for j in range(X.shape[0]):
                classif = out[j]
                if classif != 4:
                    ax.plot(X[j, 0], X[j, 1], 'o', color=colors[int(classif.item())], alpha=0.5)
                else:
                    ax.plot(X[j, 0], X[j, 1], '*', color=colors[int(classif.item())], alpha=1.0, markersize=10.0, markeredgecolor="black", markeredgewidth=0.5)
            ax.xaxis.set_tick_params(labelsize=7.5)
            ax.yaxis.set_tick_params(labelsize=7.5)
            ax.locator_params(axis="x", nbins=3)
            ax.locator_params(axis="y", nbins=3)
    fig.tight_layout()
    fig.align_ylabels(axes)
    plt.show()

def sv_prototypes_visualization_epilepsy():
    class_to_index={"epilepsy":0, "walking":1, "running":2,"sawing":3}
    train_ds, test_ds = get_ds("data/epilepsy/Epilepsy_TRAIN.ts", class_to_index), get_ds("data/epilepsy/Epilepsy_TEST.ts", class_to_index)
    # Load in models
    sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=3, num_classes=4, hidden=40, num_prototypes=5)
    sv_modules_wrapper.load_state_dict(torch.load("models/epilepsy/sv_modules_wrapper_5_40.dat"))

    fig, axes = plt.subplots(1, 3)

    red_line = Line2D([], [], color="white", marker='o', markerfacecolor="red", markersize=10.0)
    blue_line = Line2D([], [], color="white", marker='o', markerfacecolor="blue", markersize=10.0)
    green_line = Line2D([], [], color="white", marker='o', markerfacecolor="green", markersize=10.0)
    yellow_line = Line2D([], [], color="white", marker='o', markerfacecolor="yellow", markersize=10.0)
    magenta_line = Line2D([], [], color="white", marker='*', markerfacecolor="magenta", markeredgecolor="black", markeredgewidth=0.5, markersize=10.0)
    leg = fig.legend(handles=[red_line, blue_line, green_line, yellow_line, magenta_line],
               labels=["Epilepsy", "Walking", "Running", "Standing", "Prototype"], loc="lower center", ncol=5)
    leg.get_frame().set_edgecolor('k')

    visualize_moment = torch.utils.data.DataLoader(test_ds, len(test_ds), True)

    for test_sample in visualize_moment:
        inp, out = test_sample[0].detach(), test_sample[1].detach()
        for i in range(3):
            embeddings = sv_modules_wrapper.single_variable_modules[i].encoder(inp[:,:,i].unsqueeze(2).float())
            embeddings = torch.concat([embeddings, sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix], dim=0)
            out = torch.concat([out, 4*torch.ones((sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix.shape[0],))], dim=0)
            visualizer = UMAPLatent()
            trans = visualizer.umap.fit(embeddings.detach().numpy())
            X = trans.embedding_
            x_min, x_max = np.min(X, 0), np.max(X, 0)
            X = (X - x_min) / (x_max - x_min)

            colors = ["red", "blue", "green", "yellow", "magenta"]
            ax = axes[i]
            ax.set_xlabel("UMAP-1", fontsize=7.5)
            ax.set_ylabel("UMAP-2", fontsize=7.5)
            ax.set_title("Variable " + str(i+1), y=0.0, pad=-35, fontsize=10.0)
            for j in range(X.shape[0]):
                classif = out[j]
                if classif != 4:
                    ax.plot(X[j, 0], X[j, 1], 'o', color=colors[int(classif.item())], alpha=0.5)
                else:
                    ax.plot(X[j, 0], X[j, 1], '*', color=colors[int(classif.item())], alpha=1.0, markersize=10.0, markeredgecolor="black", markeredgewidth=0.5)
            ax.xaxis.set_tick_params(labelsize=7.5)
            ax.yaxis.set_tick_params(labelsize=7.5)
            ax.locator_params(axis="x", nbins=3)
            ax.locator_params(axis="y", nbins=3)
    fig.tight_layout()
    fig.align_ylabels(axes)
    plt.show()

if __name__ == "__main__":
    sv_prototypes_visualization_basicmotions()
