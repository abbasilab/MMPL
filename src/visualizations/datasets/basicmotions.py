import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import torch
import umap

from src.data.data import get_ds
from src.utils.utils import get_config_from_dataset, get_test_path_from_dataset, load_encoders, load_single_variable_prototypes_wrapper, load_multivariable_prototypes

def basicmotions_visualize(dataset, type, save):
    config = get_config_from_dataset(dataset)
    test_ds = get_ds(get_test_path_from_dataset(dataset), config['class_to_index'])
    if type == "latent-space":
        visualize_latent_space(config, test_ds, save)
    elif type == "single-var":
        visualize_single_variable_prototypes(config, test_ds, save)
    elif type == "multi-var":
        visualize_multivariable_prototypes(config, save)
    

def visualize_latent_space(config, test_ds, save):
    encoders = load_encoders(config)
    for encoder in encoders:
        encoder.eval()
    test_loader = torch.utils.data.DataLoader(test_ds, len(test_ds), shuffle=False)
    with torch.no_grad():
        classes = [0, 1, 2, 3]
        colors = ['red', 'blue', 'green', 'orange']
        variable_names = ["Acc (x)", "Acc (y)", "Acc (z)", "Gyro (x)", "Gyro (y)", "Gyro (z)"]

        fig, axs = plt.subplots(2, 3, figsize=(9, 6), sharex=True, sharey=True)

        for i in range(2):
            for j in range(3):
                ax = axs[i, j]
                variable = i*3 + j
                encoder = encoders[variable]
                for data_matrix, labels, in test_loader:
                    single_variable_data = data_matrix[:, :, variable].unsqueeze(2).float()
                    embeddings = encoder(single_variable_data)
                    reducer = umap.UMAP()
                    embeddings_2d = reducer.fit_transform(embeddings)

                    for k, label in enumerate(classes):
                        idx = np.where(labels == label)[0]
                        ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label, c=colors[k])

                        ax.set_title(variable_names[variable])
                        ax.set_xticks([])
                        ax.set_yticks([])
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=config['classes'][c],
                       markersize=10, markerfacecolor=colors[c]) for c in classes]
        fig.legend(handles=handles, ncol=4, loc='lower center')
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.1)

    if save:
        save_name = "visualizations/basicmotions/embeddings.pdf"
        plt.savefig(save_name, dpi=300)
    
    plt.show()

    


def visualize_single_variable_prototypes(config, test_ds, save):
    encoders = load_encoders(config)
    wrapper = load_single_variable_prototypes_wrapper(config)
    for encoder in encoders:
        encoder.eval()
    wrapper.eval()
    test_loader = torch.utils.data.DataLoader(test_ds, len(test_ds), shuffle=False)
    with torch.no_grad():
        classes = [0, 1, 2, 3]
        classes_with_prototype = classes + [4]
        colors = ['red', 'blue', 'green', 'orange', 'magenta']
        variable_names = ["Acc (x)", "Acc (y)", "Acc (z)", "Gyro (x)", "Gyro (y)", "Gyro (z)"]

        fig, axs = plt.subplots(2, 3, figsize=(9, 6), sharex=True, sharey=True)

        for i in range(2):
            for j in range(3):
                variable = i*3 + j

                ax = axs[i, j]
                ax.set_title(variable_names[variable])
                ax.set_xticks([])
                ax.set_yticks([])

                encoder = encoders[variable]
                for data_matrix, labels, in test_loader:
                    single_variable_data = data_matrix[:, :, variable].unsqueeze(2).float()
                    embeddings = encoder(single_variable_data)
                    embeddings = torch.concat([embeddings, wrapper.single_variable_prototype_modules[variable].prototypes], dim=0)
                    labels = torch.concat([labels, len(classes)*torch.ones((wrapper.single_variable_prototype_modules[variable].prototypes.shape[0],))], dim=0)
                    reducer = umap.UMAP()
                    embeddings_2d = reducer.fit_transform(embeddings)

                    for k, label in enumerate(classes):
                        idx = np.where(labels == label)[0]
                        ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label, c=colors[k])

                    idx = np.where(labels == len(classes))[0]
                    ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=len(classes), marker="*", edgecolor='black', s=50, c=colors[-1])

        legend_names = config['classes'] + ["Prototype"]
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=legend_names[c],
                       markersize=10, markerfacecolor=colors[c]) for c in classes_with_prototype]
        fig.legend(handles=handles, ncol=5, loc='lower center')
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.1)


    if save:
        save_name = "visualizations/basicmotions/single_variable_prototypes.pdf"
        plt.savefig(save_name, dpi=300)

    plt.show()

def visualize_multivariable_prototypes(config, save):
    plt.figure()

    multivariable_prototypes = load_multivariable_prototypes(config)
    sns.heatmap(multivariable_prototypes.prototypes.detach().numpy())

    if save:
        save_name = "visualizations/basicmotions/multivariable_prototypes.pdf"
        plt.savefig(save_name, dpi=300)
    plt.show()

                
