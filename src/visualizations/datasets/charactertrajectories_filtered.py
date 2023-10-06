import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import seaborn as sns
from sktime.datasets import load_from_tsfile_to_dataframe
import torch
import umap

from src.data.data import get_ds
from src.utils.utils import get_config_from_dataset, get_train_path_from_dataset, get_test_path_from_dataset, load_encoders, load_single_variable_prototypes_wrapper, load_multivariable_prototypes

def charactertrajectories_filtered_visualize(dataset, type, save):
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
        visualize_projected_prototypes(config, train_ds, save)
    elif type == "chars":
        visualize_characters(config, train_ds, save)
    

def visualize_latent_space(config, test_ds, save):
    encoders = load_encoders(config)
    for encoder in encoders:
        encoder.eval()
    test_loader = torch.utils.data.DataLoader(test_ds, len(test_ds), shuffle=False, pin_memory=True)
    with torch.no_grad():
        classes = [0, 1, 2, 3]
        colors = ['red', 'blue', 'green', 'orange']
        variable_names = ["x", "y", "Pen Tip Force"]

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

        for j in range(3):
            ax = axs[j]
            encoder = encoders[j]
            for data_matrix, labels, in test_loader:
                single_variable_data = data_matrix[:, :, j].unsqueeze(2).float()
                embeddings = encoder(single_variable_data)
                reducer = umap.UMAP()
                embeddings_2d = reducer.fit_transform(embeddings.cpu())
                e_min, e_max = np.min(embeddings_2d, 0), np.max(embeddings_2d, 0)
                embeddings_2d = (embeddings_2d - e_min) / (e_max - e_min)

                for k, label in enumerate(classes):
                    idx = np.where(labels == label)[0]
                    ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label, c=colors[k], alpha=0.5)

                    ax.set_title(variable_names[j])
                    ax.set_xticks([])
                    ax.set_yticks([])
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=config['classes'][c],
                       markersize=10, markerfacecolor=colors[c]) for c in classes]
        fig.legend(handles=handles, ncol=4, loc='lower center')
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.1)

    if save:
        save_name = "visualizations/charactertrajectories_filtered/embeddings.pdf"
        plt.savefig(save_name, dpi=300)
    plt.show()


def visualize_single_variable_prototypes(config, test_ds, save):
    encoders = load_encoders(config)
    wrapper = load_single_variable_prototypes_wrapper(config)
    for encoder in encoders:
        encoder.eval()
    wrapper.eval()
    test_loader = torch.utils.data.DataLoader(test_ds, len(test_ds), shuffle=False, pin_memory=True)
    with torch.no_grad():
        classes = [0, 1, 2, 3]
        classes_with_prototype = classes + [4]
        colors = ['red', 'blue', 'green', 'orange', 'magenta']
        variable_names = ["Acc (x)", "Acc (y)", "Acc (z)"]

        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)

        for j in range(3):

            ax = axs[j]
            ax.set_title(variable_names[j])
            ax.set_xticks([])
            ax.set_yticks([])

            encoder = encoders[j]
            for data_matrix, labels, in test_loader:
                single_variable_data = data_matrix[:, :, j].unsqueeze(2).float()
                embeddings = encoder(single_variable_data)
                embeddings = torch.concat([embeddings, wrapper.single_variable_prototype_modules[j].prototypes], dim=0)
                labels = torch.concat([labels, len(classes)*torch.ones((wrapper.single_variable_prototype_modules[j].prototypes.shape[0],))], dim=0)
                reducer = umap.UMAP()
                embeddings_2d = reducer.fit_transform(embeddings.cpu())
                e_min, e_max = np.min(embeddings_2d, 0), np.max(embeddings_2d, 0)
                embeddings_2d = (embeddings_2d - e_min) / (e_max - e_min)

                for k, label in enumerate(classes):
                    idx = np.where(labels == label)[0]
                    ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label, c=colors[k], alpha=0.5)

                idx = np.where(labels == len(classes))[0]
                ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=len(classes), marker="*", edgecolor='black', s=75, c=colors[-1])

        legend_names = config['classes'] + ["Prototype"]
        handles = [plt.Line2D([0], [0], marker='o', color='w', label=legend_names[c],
                       markersize=10, markerfacecolor=colors[c]) for c in classes_with_prototype]
        fig.legend(handles=handles, ncol=5, loc='lower center')
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.1)

    if save:
        save_name = "visualizations/charactertrajectories_filtered/single_variable_prototypes.pdf"
        plt.savefig(save_name, dpi=300)
    plt.show()


def visualize_multivariable_prototypes(config, save):
    plt.figure()

    multivariable_module = load_multivariable_prototypes(config)
    sorted_prototypes = torch.zeros_like(multivariable_module.prototypes)
    for var in range(multivariable_module.num_variables):
        start_idx = var*multivariable_module.num_sv_prototypes
        end_idx = (var+1)*multivariable_module.num_sv_prototypes
        blocks = multivariable_module.prototypes[:, start_idx:end_idx]
        max_indices = blocks.argmax(dim=1)
        sorted_indices = max_indices.argsort()
        sorted_blocks = blocks[sorted_indices]
        sorted_prototypes[:, start_idx:end_idx] = sorted_blocks

    sorted_prototypes_tensor = sorted_prototypes.cpu().detach().numpy()
    ax = sns.heatmap(sorted_prototypes_tensor)
    cbar = ax.collections[0].colorbar
    min_val = round(np.min(sorted_prototypes_tensor), 2)
    max_val = round(np.max(sorted_prototypes_tensor), 2)
    mid_val = round((min_val + max_val) / 2, 2)
    cbar.set_ticks([min_val, mid_val, max_val])

    ax.set_xticks([])
    ax.set_yticks([])

    num_variables = multivariable_module.num_variables
    num_sv_prototypes = multivariable_module.num_sv_prototypes
    for i in range(num_sv_prototypes, num_variables*num_sv_prototypes, num_sv_prototypes):
        ax.axvline(x=i, color='white', linewidth=3)

    if save:
        save_name = "visualizations/charactertrajectories_filtered/multivariable_prototypes.pdf"
        plt.savefig(save_name, dpi=300)
    plt.show()

def visualize_projected_prototypes(config, train_ds, save):
    multivariable_module = load_multivariable_prototypes(config)
    multivariable_module.eval()
    train_loader = torch.utils.data.DataLoader(train_ds, len(train_ds), shuffle=False, pin_memory=True)
    fig, axs = plt.subplots(4, 3, figsize=(6.75, 9))
    colors = ['red', 'blue', 'green', 'orange']
    classes = ['b', 'd', 'p', "q"]
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
                    ax.plot(closest_point, c=colors[int(labels[closest_index])])
                    if j == 0:
                        ax.set_ylabel(classes[int(labels[closest_index])])
    fig.align_ylabels()
    plt.show()

def visualize_characters(config, train_ds, save):
    multivariable_module = load_multivariable_prototypes(config)
    multivariable_module.eval()
    train_loader = torch.utils.data.DataLoader(train_ds, len(train_ds), shuffle=False, pin_memory=True)
    colors = ['red', 'blue', 'green', 'orange']
    classes = ['b', 'd', 'p', "q"]
    fig, axs = plt.subplots(1, 4, figsize=(8, 2))
    with torch.no_grad():
        prototype_matrix = multivariable_module.prototypes
        wrapper = multivariable_module.wrapper
        for data_matrix, labels in train_loader:
            for i in range(multivariable_module.num_classes):
                prototype = prototype_matrix[i]
                chunks = prototype.split(wrapper.num_prototypes)

                x_chunk = chunks[0]
                x_index = torch.argmax(x_chunk)
                x_prototype = wrapper.single_variable_prototype_modules[0].prototypes[x_index]
                x_data = data_matrix[:, :, 0].unsqueeze(2).float()
                x_embeddings = wrapper.single_variable_prototype_modules[0].encoder(x_data)
                x_distances = torch.norm(x_embeddings - x_prototype, dim=1)
                x_closest_index = torch.argmin(x_distances).item()
                x_closest_point = x_data[x_closest_index].squeeze(1)

                y_chunk = chunks[1]
                y_index = torch.argmax(y_chunk)
                y_prototype = wrapper.single_variable_prototype_modules[1].prototypes[y_index]
                y_data = data_matrix[:, :, 1].unsqueeze(2).float()
                y_embeddings = wrapper.single_variable_prototype_modules[1].encoder(y_data)
                y_distances = torch.norm(y_embeddings - y_prototype, dim=1)
                y_closest_index = torch.argmin(y_distances).item()
                y_closest_point = y_data[y_closest_index].squeeze(1)


                x_int, y_int = torch.cumsum(x_closest_point, dim=0), torch.cumsum(y_closest_point, dim=0)
                label = int(labels[x_closest_index])
                ax = axs[label]
                ax.plot(x_int, y_int, c=colors[label])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_title(classes[label])
    plt.show()



                


                
