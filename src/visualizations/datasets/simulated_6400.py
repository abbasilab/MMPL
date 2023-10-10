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
    elif type == "sims":
        visualize_random_sims(config, test_ds)
    elif type == "multi-var":
        visualize_multivariable_prototypes(config, save)
    elif type == "project":
        visualize_projected_prototypes(config, train_ds, save)
    

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
        colors = ['red', 'blue', 'green', 'orange', 'magenta']
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
                        ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label="Prototype", marker="*", edgecolor='black', s=75, c='magenta')

        handles = [plt.Line2D([0], [0], marker='o', color='w', label=pattern_labels[c],
                       markersize=10, markerfacecolor=colors[c]) for c in range(5)]
        fig.legend(handles=handles, ncol=5, loc='lower center')
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.1)


    if save:
        save_name = "visualizations/simulated_6400/single_variable_prototypes.pdf"
        plt.savefig(save_name, dpi=300)

    plt.show()

def visualize_random_sims(config, test_ds):
    wrapper = load_single_variable_prototypes_wrapper(config).to(device)
    wrapper.eval()
    test_loader = torch.utils.data.DataLoader(test_ds, len(test_ds), shuffle=False, pin_memory=True)
    with torch.no_grad():
        for data_matrix, labels in test_loader:
            data_matrix, labels = data_matrix.to(device).float(), labels.to(device).float()
            selected_data = torch.empty((64, 100, 4)).to(device)
            selected_labels = torch.empty((64,)).to(device)
            for i in range(64):
                start_idx = i*100
                end_idx = (i+1)*100
                rand_idx = torch.randint(start_idx, end_idx, (1,))
                selected_data[i] = data_matrix[rand_idx]
                selected_labels[i] = labels[rand_idx]
            sims, _ = wrapper(selected_data)
            sims = sims[:, :-4]
            x1 = sims.unsqueeze(0)
            x2 = sims.unsqueeze(1)
            distances = torch.linalg.norm(x1-x2, dim=2)
            print(distances)
            print(torch.topk(distances, k=2, dim=0, largest=False).values)
            print(torch.max(distances, dim=0).values)
            plt.figure()
            sns.heatmap(sims.to("cpu").detach().numpy())
            plt.show()


def sorting_key(prototype):
    indices = []
    for i in range(3):
        single_variable_prototype = prototype[i*4:(i+1)*4]
        high_value_index = torch.argmax(single_variable_prototype).item()
        indices.append(high_value_index)
    return tuple(indices)

def visualize_multivariable_prototypes(config, save):
    plt.figure()

    multivariable_module = load_multivariable_prototypes(config)
    prototypes = multivariable_module.prototypes#[:, :-4]
    prototypes_list = [prototypes[i, :] for i in range(prototypes.shape[0])]
    sorted_prototypes = sorted(prototypes_list, key=sorting_key)
    sorted_prototypes_tensor = torch.stack(sorted_prototypes).cpu().detach().numpy()
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
    for i in range(num_variables, num_variables*num_sv_prototypes, num_variables):
        ax.axvline(x=i, color='white', linewidth=3)

    if save:
        save_name = "visualizations/simulated_6400/multivariable_prototypes.pdf"
        plt.savefig(save_name, dpi=300)
    plt.show()

def generate_prototype_to_class_mapping(multivariable_module, train_ds, indices_to_use):
    multivariable_module.eval()
    prototype_matrix = multivariable_module.prototypes
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=len(indices_to_use), shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices_to_use))
    prototype_to_class_mapping = {}
    classes_of_interest = [0, 21, 42, 63]  # The classes you're interested in
    
    with torch.no_grad():
        for data_matrix, labels in train_loader:
            data_matrix, labels = data_matrix.to(device).float(), labels.to(device).float()
            _, sv_sims = multivariable_module(data_matrix)
            
            for class_id in classes_of_interest:
                # Extract sv_sims corresponding to the current class_id
                sv_sims_class = sv_sims[labels == class_id]
                
                # Average sv_sims across all instances of the current class
                avg_sv_sims = torch.mean(sv_sims_class, dim=0)
                
                # Calculate distances between avg_sv_sims and each prototype
                distances = torch.norm(prototype_matrix - avg_sv_sims, dim=1)
                
                # Identify the index of the closest prototype
                closest_prototype_index = torch.argmin(distances).item()
                
                # Map the class_id to the closest prototype index
                prototype_to_class_mapping[class_id] = closest_prototype_index
                
    return prototype_to_class_mapping

def visualize_projected_prototypes(config, train_ds, save):    
    multivariable_module = load_multivariable_prototypes(config).to(device)
    multivariable_module.eval()
    indices_to_use = list(range(0, 100)) + list(range(2100, 2200)) + list(range(4200, 4300)) + list(range(6300, 6400))
    prototype_to_class_mapping = generate_prototype_to_class_mapping(multivariable_module, train_ds, indices_to_use)
    prototype_matrix = multivariable_module.prototypes
    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=len(indices_to_use), shuffle=False, sampler=torch.utils.data.SubsetRandomSampler(indices_to_use))
    
    fig, axs = plt.subplots(4, 3, figsize=(12, 9))
    colors = ['red', 'blue', 'green', 'orange']
    classes_of_interest = [0, 21, 42, 63]
    variable_names = ['Variable 1', 'Variable 2', 'Variable 3', 'Variable 4']

    global_min = float('inf')
    global_max = float('-inf')

    with torch.no_grad():
        wrapper = multivariable_module.wrapper
        for c, class_id in enumerate(classes_of_interest):
            prototype_index = prototype_to_class_mapping[class_id]
            prototype = prototype_matrix[prototype_index]
            chunks = prototype.split(wrapper.num_prototypes)
            
            for j, chunk in enumerate(chunks):
                if j == len(chunks) - 1:
                    break
                index = torch.argmax(chunk)
                sv_prototype = wrapper.single_variable_prototype_modules[j].prototypes[index]

                for data_matrix, labels in train_loader:
                    data_matrix, labels = data_matrix.to(device), labels.to(device)
                    single_variable_data = data_matrix[:, :, j].unsqueeze(2).float()
                    embeddings = wrapper.single_variable_prototype_modules[j].encoder(single_variable_data)
                    distances = torch.norm(embeddings - sv_prototype, dim=1)
                    closest_index = torch.argmin(distances).item()
                    closest_point = single_variable_data[closest_index].squeeze(1)
                    ax = axs[c, j]

                    ax.plot(closest_point.cpu(), c=colors[j])
                    
                    local_min = closest_point.min().item()
                    local_max = closest_point.max().item()
                    global_min = min(global_min, local_min)
                    global_max = max(global_max, local_max)

                    if j == 0:
                        ax.set_ylabel(f'Class {class_id}')
                    if c < len(classes_of_interest) - 1:
                        ax.set_xticks([])
                    if j > 0:
                        ax.set_yticks([])
                    if c == 0:
                        ax.set_title(variable_names[j])

        for ax in axs.flat:
            ax.set_ylim(global_min, global_max)

    fig.align_ylabels()
    if save:
        save_name = "visualizations/simulated_6400/projected.pdf"
        plt.savefig(save_name, dpi=300)
    plt.show()
                
