import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
import torch
import umap

from src.data.data import get_ds
from src.utils.utils import *

def simulated_6400_visualize(dataset, type, save):
    config = get_config_from_dataset(dataset)
    train_ds = get_ds(get_train_path_from_dataset(dataset), config['class_to_index'])
    test_ds = get_ds(get_test_path_from_dataset(dataset), config['class_to_index'])
    if type == "latent-space":
        visualize_latent_space(config, test_ds, save)
    elif type == "single-var":
        visualize_single_variable_prototypes(config, test_ds, save)
    elif type == "multi-var":
        visualize_multivariable_prototypes(config, save)
    elif type == "project":
        visualize_projected_prototypes(config, test_ds, save)
    

def visualize_latent_space(config, test_ds, save):
    encoders = load_encoders(config)
    for encoder in encoders:
        encoder.eval()
    test_loader = torch.utils.data.DataLoader(test_ds, len(test_ds), shuffle=False, pin_memory=True)
    class_to_pattern_map = get_class_to_pattern_map()
    with torch.no_grad():
        classes = [i for i in range(64)]
        colors = ['red', 'blue', 'green', 'orange']
        variable_names = ["Variable 1", "Variable 2", "Variable 3", "Variable 4"]
        pattern_labels = ["Pattern 1", "Pattern 2", "Pattern 3", "Pattern 4"]

        fig, axs = plt.subplots(2, 2, figsize=(7, 7), sharex=True, sharey=True)

        for i in range(2):
            for j in range(2):
                variable = i*2 + j

                ax = axs[i, j]
                ax.set_title(variable_names[variable])
                ax.set_xticks([])
                ax.set_yticks([])

                encoder = encoders[variable]
                for data_matrix, labels, in test_loader:
                    single_variable_data = data_matrix[:, :, variable].unsqueeze(2).float()
                    embeddings = encoder(single_variable_data)
                    reducer = umap.UMAP()
                    embeddings_2d = reducer.fit_transform(embeddings.cpu())

                    if variable == 3:
                        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='grey')
                    else:
                        for k, label in enumerate(classes):
                            idx = np.where(labels == label)[0]
                            pattern = int(class_to_pattern_map[label][variable])
                            ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=pattern_labels[pattern], c=colors[pattern], alpha=0.2)

        handles = [plt.Line2D([0], [0], marker='o', color='w', label=pattern_labels[c],
                       markersize=10, markerfacecolor=colors[c]) for c in range(4)]
        fig.legend(handles=handles, ncol=5, loc='lower center')
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.1)


    if save:
        save_name = "visualizations/simulated_6400/single_variable_prototypes.pdf"
        plt.savefig(save_name, dpi=300)

    plt.show()


def visualize_single_variable_prototypes(config, test_ds, save):
    encoders = load_encoders(config)
    wrapper = load_single_variable_prototypes_wrapper(config)
    for encoder in encoders:
        encoder.eval()
    wrapper.eval()
    test_loader = torch.utils.data.DataLoader(test_ds, len(test_ds), shuffle=False, pin_memory=True)
    class_to_pattern_map = get_class_to_pattern_map()
    with torch.no_grad():
        classes = [i for i in range(64)]
        colors = ['red', 'blue', 'green', 'orange', 'black']
        variable_names = ["Variable 1", "Variable 2", "Variable 3", "Variable 4"]
        pattern_labels = ["Pattern 1", "Pattern 2", "Pattern 3", "Pattern 4", "Prototype"]

        fig, axs = plt.subplots(2, 2, figsize=(7, 7), sharex=True, sharey=True)

        for i in range(2):
            for j in range(2):
                variable = i*2 + j

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
                    embeddings_2d = reducer.fit_transform(embeddings.cpu())

                    if variable == 3:
                        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='grey')
                    else:
                        for k, label in enumerate(classes):
                            idx = np.where(labels == label)[0]
                            pattern = int(class_to_pattern_map[label][variable])
                            ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=pattern_labels[pattern], c=colors[pattern], alpha=0.2)

                        idx = np.where(labels == len(classes))[0]
                        ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label="Prototype", marker="*", edgecolor='black', s=50, c='black')

        handles = [plt.Line2D([0], [0], marker='o', color='w', label=pattern_labels[c],
                       markersize=10, markerfacecolor=colors[c]) for c in range(5)]
        fig.legend(handles=handles, ncol=5, loc='lower center')
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.1)


    if save:
        save_name = "visualizations/simulated_6400/single_variable_prototypes.pdf"
        plt.savefig(save_name, dpi=300)

    plt.show()


def visualize_multivariable_prototypes(config, save):
    plt.figure()

    multivariable_module = load_multivariable_prototypes(config)
    sorted_prototypes = torch.zeros_like(multivariable_module.prototypes)
    for var in range(multivariable_module.num_variables - 1):
        start_idx = var*multivariable_module.num_sv_prototypes
        end_idx = (var+1)*multivariable_module.num_sv_prototypes
        blocks = multivariable_module.prototypes[:, start_idx:end_idx]
        max_indices = blocks.argmax(dim=1)
        sorted_indices = max_indices.argsort()
        sorted_blocks = blocks[sorted_indices]
        sorted_prototypes[:, start_idx:end_idx] = sorted_blocks

    sns.heatmap(sorted_prototypes.cpu().detach().numpy())

    if save:
        save_name = "visualizations/simulated_6400/multivariable_prototypes.pdf"
        plt.savefig(save_name, dpi=300)
    plt.show()

def visualize_projected_prototypes(config, train_ds, save):
    multivariable_module = load_multivariable_prototypes(config)
    multivariable_module.eval()
    train_loader = torch.utils.data.DataLoader(train_ds, len(train_ds), shuffle=False, pin_memory=True)
    fig, axs = plt.subplots(4, 3, figsize=(6.75, 9))
    colors = ['red', 'green', 'blue', 'orange']
    classes = ['Epilepsy', 'Running', 'Walking', "Sawing"]
    with torch.no_grad():
        prototype_matrix = multivariable_module.prototypes
        wrapper = multivariable_module.wrapper
        for i in range(multivariable_module.num_classes):
            prototype = prototype_matrix[i]
            chunks = prototype.split(wrapper.num_prototypes)
            for j in range(len(chunks)):
                index = torch.argmax(chunks[j])
                sv_prototype = wrapper.single_variable_prototype_modules[j].prototypes[index]

                for data_matrix, labels in train_loader:
                    single_variable_data = data_matrix[:, :, j].unsqueeze(2).float()
                    embeddings = wrapper.single_variable_prototype_modules[j].encoder(single_variable_data)
                    distances = torch.norm(embeddings - sv_prototype, dim=1)
                    closest_index = torch.argmin(distances).item()
                    closest_point = single_variable_data[closest_index].squeeze(1)
                    ax = axs[i, j]
                    ax.plot(closest_point, c=colors[i])
                    if j == 0:
                        ax.set_ylabel(classes[i])
    fig.align_ylabels()
    plt.show()

                
