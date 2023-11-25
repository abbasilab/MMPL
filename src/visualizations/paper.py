import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import torch
import umap

from src.data.data import get_ds
from src.utils.utils import *

basicmotions_config = get_config_from_dataset("basicmotions")
basicmotions_train_ds = get_ds(get_train_path_from_dataset("basicmotions"), basicmotions_config['class_to_index'])
basicmotions_test_ds = get_ds(get_test_path_from_dataset("basicmotions"), basicmotions_config['class_to_index'])
basicmotions_train_dl = torch.utils.data.DataLoader(basicmotions_train_ds, len(basicmotions_train_ds), shuffle=True)
basicmotions_test_dl = torch.utils.data.DataLoader(basicmotions_test_ds, len(basicmotions_test_ds), shuffle=True)
basicmotions_encoders = load_encoders(basicmotions_config)
basicmotions_sv_prototype_modules = load_single_variable_prototypes_wrapper(basicmotions_config)
basicmotions_mv_module = load_multivariable_prototypes(basicmotions_config)

epilepsy_config = get_config_from_dataset("epilepsy")
epilepsy_train_ds = get_ds(get_train_path_from_dataset("epilepsy"), epilepsy_config['class_to_index'])
epilepsy_test_ds = get_ds(get_test_path_from_dataset("epilepsy"), epilepsy_config['class_to_index'])
epilepsy_train_dl = torch.utils.data.DataLoader(epilepsy_train_ds, len(epilepsy_train_ds), shuffle=True)
epilepsy_test_dl = torch.utils.data.DataLoader(epilepsy_test_ds, len(epilepsy_test_ds), shuffle=True)
epilepsy_encoders = load_encoders(epilepsy_config)
epilepsy_sv_prototype_modules = load_single_variable_prototypes_wrapper(epilepsy_config)
epilepsy_mv_module = load_multivariable_prototypes(epilepsy_config)

charactertrajectories_filtered_config = get_config_from_dataset("charactertrajectories_filtered")
charactertrajectories_filtered_train_ds = get_ds(get_train_path_from_dataset("charactertrajectories_filtered"), charactertrajectories_filtered_config['class_to_index'])
charactertrajectories_filtered_test_ds = get_ds(get_test_path_from_dataset("charactertrajectories_filtered"), charactertrajectories_filtered_config['class_to_index'])
charactertrajectories_filtered_train_dl = torch.utils.data.DataLoader(charactertrajectories_filtered_train_ds, len(charactertrajectories_filtered_train_ds), shuffle=True)
charactertrajectories_filtered_test_dl = torch.utils.data.DataLoader(charactertrajectories_filtered_test_ds, len(charactertrajectories_filtered_test_ds), shuffle=True)
charactertrajectories_filtered_encoders = load_encoders(charactertrajectories_filtered_config)
charactertrajectories_filtered_sv_prototype_modules = load_single_variable_prototypes_wrapper(charactertrajectories_filtered_config)
charactertrajectories_filtered_mv_module = load_multivariable_prototypes(charactertrajectories_filtered_config)

simulated_config = get_config_from_dataset("simulated_6400")
simulated_train_ds = get_ds(get_train_path_from_dataset("simulated_6400"), simulated_config['class_to_index'])
simulated_test_ds = get_ds(get_test_path_from_dataset("simulated_6400", train=False), simulated_config['class_to_index'])
simulated_train_dl = torch.utils.data.DataLoader(simulated_train_ds, len(simulated_train_ds), shuffle=True)
simulated_test_dl = torch.utils.data.DataLoader(simulated_test_ds, len(simulated_test_ds), shuffle=True)
simulated_encoders = load_encoders(simulated_config)
simulated_sv_prototype_modules = load_single_variable_prototypes_wrapper(simulated_config)
simulated_mv_module = load_multivariable_prototypes(simulated_config)

one_stage_config = get_comparison_config_from_dataset("one_stage", "simulated_640")
one_stage_train_ds = get_ds(get_train_path_from_dataset("simulated_640"), one_stage_config['class_to_index'])
one_stage_test_ds = get_ds(get_test_path_from_dataset("simulated_640", train=False), one_stage_config['class_to_index'])
one_stage_train_dl = torch.utils.data.DataLoader(one_stage_train_ds, len(one_stage_train_ds), shuffle=True)
one_stage_test_dl = torch.utils.data.DataLoader(one_stage_test_ds, len(one_stage_test_ds), shuffle=True)
one_stage_model = load_one_stage_model(one_stage_config)

tab10 = plt.cm.get_cmap("tab10")
all_colors = list(tab10.colors)

dark2 = plt.cm.get_cmap("Dark2")
other_colors = list(dark2.colors)

tab20 = plt.cm.get_cmap('tab20', 20)
tab20b = plt.cm.get_cmap('tab20b', 20)
tab20c = plt.cm.get_cmap('tab20c', 20)
sixty_colors = list(tab20.colors) + list(tab20b.colors) + list(tab20c.colors)

plt.rcParams['font.family'] = 'Arial'

def simulated_dataset_generation(save=False):
    return

def simulated_single_variable_prototypes(save=False):
    class_to_pattern_map = get_class_to_pattern_map()
    with torch.no_grad():
        classes = [i for i in range(64)]
        colors = all_colors[:4] + [all_colors[6]]
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

                encoder = simulated_encoders[variable]
                for data_matrix, labels, in simulated_test_dl:
                    single_variable_data = data_matrix[:, :, variable].unsqueeze(2).float()
                    embeddings = encoder(single_variable_data)
                    embeddings = torch.concat([embeddings, simulated_sv_prototype_modules.single_variable_prototype_modules[variable].prototypes], dim=0)
                    labels = torch.concat([labels, len(classes)*torch.ones((simulated_sv_prototype_modules.single_variable_prototype_modules[variable].prototypes.shape[0],))], dim=0)
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
                        ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label="Prototype", marker="*", edgecolor='black', linewidth=1.5, s=120, c=colors[-1])

        legend_names = pattern_labels + ["Prototype"]
        handles = [plt.Line2D([0], [0], marker='o' if c != 4 else '*', color='w', label=legend_names[c],
                       markersize=10 if c != 4 else 11, markerfacecolor=colors[c], markeredgecolor='None' if c != 4 else 'black') for c in range(5)]
        fig.legend(handles=handles, ncol=5, loc='lower center')
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.1)


    if save:
        save_name = "visualizations/paper/simulated_sv.pdf"
        plt.savefig(save_name, dpi=300)

    plt.show()

def sorting_key(prototype):
    indices = []
    for i in range(3):
        single_variable_prototype = prototype[i*4:(i+1)*4]
        high_value_index = torch.argmax(single_variable_prototype).item()
        indices.append(high_value_index)
    return tuple(indices)

def simulated_multivariable_prototypes(save):
    plt.figure()
    prototypes = simulated_mv_module.prototypes
    prototypes_list = [prototypes[i, :] for i in range(prototypes.shape[0])]
    sorted_prototypes = sorted(prototypes_list, key=sorting_key)
    sorted_prototypes_tensor = torch.stack(sorted_prototypes)
    ax = sns.heatmap(sorted_prototypes_tensor.cpu().detach().numpy())

    for col in range(0, sorted_prototypes_tensor.shape[1], 4):
        ax.axvline(x=col, color='white', lw=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    min_val = round(sorted_prototypes_tensor.min().item(), 2)
    max_val = round(sorted_prototypes_tensor.max().item(), 2)
    mid_val = round((min_val + max_val) / 2, 2)

    # Set the colorbar ticks and labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([min_val, mid_val, max_val])

    if save:
        save_name = "visualizations/paper/simulated_mv.svg"
        plt.savefig(save_name, dpi=300)
    plt.show()

def simulated_projected(save): 
    prototype_matrix = simulated_mv_module.prototypes
    prototype_indices = [39, 27, 49, 54]
    fig, axs = plt.subplots(4, 4, figsize=(8, 8))
    classes_of_interest = [0, 21, 42, 63]
    variable_names = ['Variable 1', 'Variable 2', 'Variable 3', 'Variable 4']

    global_min = float('inf')
    global_max = float('-inf')

    with torch.no_grad():
        wrapper = simulated_mv_module.wrapper
        for i in range(len(prototype_indices)):
            prototype = prototype_matrix[prototype_indices[i]]
            chunks = prototype.split(wrapper.num_prototypes)
            
            for j, chunk in enumerate(chunks):
                index = torch.argmax(chunk)
                sv_prototype = wrapper.single_variable_prototype_modules[j].prototypes[index]

                for data_matrix, labels in simulated_train_dl:
                    data_matrix, labels = data_matrix.to(device), labels.to(device)
                    single_variable_data = data_matrix[:, :, j].unsqueeze(2).float()
                    embeddings = wrapper.single_variable_prototype_modules[j].encoder(single_variable_data)
                    distances = torch.norm(embeddings - sv_prototype, dim=1)
                    closest_index = torch.argmin(distances).item()
                    closest_point = single_variable_data[closest_index].squeeze(1)
                    ax = axs[i, j]

                    ax.plot(closest_point.cpu(), c=all_colors[j])
                    
                    local_min = closest_point.min().item()
                    local_max = closest_point.max().item()
                    global_min = min(global_min, local_min)
                    global_max = max(global_max, local_max)

                    if j == 0:
                        ax.set_ylabel(f'Class {classes_of_interest[i]}', fontsize=12)
                    if i < len(classes_of_interest) - 1:
                        ax.set_xticks([])
                    if j > 0:
                        ax.set_yticks([])
                    if i == 0:
                        ax.set_title(variable_names[j])

        for ax in axs.flat:
            ax.set_ylim(global_min, global_max)

    fig.align_ylabels()
    if save:
        save_name = "visualizations/paper/simulated_projected.pdf"
        plt.savefig(save_name, dpi=300)
    plt.show()

def simulated_no_contrastive_single_variable_prototypes(save=False):
    return

def simulated_silhouette_score_vs_number_of_clusters(save=False):
    variables = ["Variable 1", "Variable 1", "Variable 3", "Variable 4"]
    with torch.no_grad():
        for data_matrix, labels in simulated_train_dl:
            data_matrix, labels = data_matrix.to(device), labels.to(device)
            k_values = range(2, 11)
            all_silhouette_scores = np.zeros((len(k_values), 4))
            for var in range(4):
                single_variable_data = data_matrix[:, :, var].unsqueeze(2).float()
                encoder = simulated_encoders[var].to(device)
                embeddings = encoder(single_variable_data)
                embeddings = embeddings.cpu().detach().numpy()

                silhouette_scores = []

                for k in k_values:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    labels = kmeans.fit_predict(embeddings)
                    score = silhouette_score(embeddings, labels)
                    silhouette_scores.append(score)
                all_silhouette_scores[:, var] = silhouette_scores
                plt.plot(k_values, all_silhouette_scores[:, var], marker='o', label=variables[var], color=other_colors[var])

    plt.title('Silhouette Score vs. Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.xticks(k_values)
    plt.legend()
    
    if save:
        save_name = "visualizations/paper/simulated_silhouette.pdf"
        plt.savefig(save_name, dpi=300)
    plt.show()

def simulated_one_stage(save=False):
    for data_matrix, labels in one_stage_train_dl:
        data_matrix, labels = data_matrix.to(device), labels.to(device)
        plt.figure()
        with torch.no_grad():
            _, _, embeddings = one_stage_model(data_matrix.float())
            reducer = umap.UMAP()
            embeddings_2d = reducer.fit_transform(embeddings.cpu())

            string_labels = np.array([one_stage_config['classes'][label.item()] for label in labels])
            handles, lbls = [], []
            for label in one_stage_config['classes']:
                idx = np.where(string_labels == label)[0]
                scatter = plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label,
                                       c=sixty_colors[int(label) % 60])
                handles.append(scatter)
                lbls.append(label)
                
            plt.title("Latent Space")
            plt.show()

def simulated_one_stage_one_pattern(save=False):
    class_to_pattern_map = get_class_to_pattern_map()
    for data_matrix, labels in one_stage_train_dl:
        data_matrix, labels = data_matrix.to(device), labels.to(device)
        plt.figure()
        with torch.no_grad():
            _, _, embeddings = one_stage_model(data_matrix.float())
            reducer = umap.UMAP()
            embeddings_2d = reducer.fit_transform(embeddings.cpu())
            e_min, e_max = np.min(embeddings_2d, 0), np.max(embeddings_2d, 0)
            embeddings_2d = (embeddings_2d - e_min) / (e_max - e_min)

            string_labels = np.array([one_stage_config['classes'][label.item()] for label in labels])
            handles, lbls = [], []
            for label in one_stage_config['classes']:
                pattern = class_to_pattern_map[int(label)]
                sv_pattern = int(pattern[0])
                idx = np.where(string_labels == label)[0]
                scatter = plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label,
                                       c=all_colors[sv_pattern], alpha=0.5)
                handles.append(scatter)
                lbls.append(label)
                
            plt.title("Latent Space")
            plt.show()

def epilepsy_single_variable_prototypes(save=False):
    with torch.no_grad():
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
        variable_names = ["Acc (x)", "Acc (y)", "Acc (z)"]
        colors = all_colors[:4] + [all_colors[6]]
        for j in range(3):
            ax = axs[j]
            ax.set_title(variable_names[j])
            ax.set_xticks([])
            ax.set_yticks([])

            encoder = epilepsy_encoders[j]
            sv_module = epilepsy_sv_prototype_modules.single_variable_prototype_modules[j]
            for data_matrix, labels in epilepsy_test_dl:
                single_variable_data = data_matrix[:, :, j].unsqueeze(2).float()
                embeddings = encoder(single_variable_data)
                embeddings_with_protos = torch.concat([embeddings, sv_module.prototypes], dim=0)
                labels_with_protos = torch.concat([labels, epilepsy_config["num_classes"]*torch.ones((sv_module.prototypes.shape[0],))], dim=0)

                reducer = umap.UMAP()
                embeddings_2d = reducer.fit_transform(embeddings_with_protos)
                e_min, e_max = np.min(embeddings_2d, 0), np.max(embeddings_2d, 0)
                embeddings_2d = (embeddings_2d - e_min) / (e_max - e_min)

                for k in range(4):
                    idx = np.where(labels_with_protos == k)[0]
                    ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=k, color=colors[k], s=50, alpha=0.5)

                idx = np.where(labels_with_protos == 4)[0]
                ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=4, marker="*", edgecolor='black', linewidth=1.5, s=120, c=colors[-1])

        legend_names = epilepsy_config['classes'] + ["Prototype"]
        handles = [plt.Line2D([0], [0], marker='o' if c != 4 else '*', color='w', label=legend_names[c],
                       markersize=10 if c != 4 else 11, markerfacecolor=colors[c], markeredgecolor='None' if c != 4 else 'black') for c in range(5)]
        fig.legend(handles=handles, ncol=5, loc='lower center')
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.1)

        if save:
            plt.savefig("visualizations/paper/epilepsy_sv.pdf", dpi=300)
        plt.show()

def epilepsy_multivariable_prototypes(save=False):
    plt.figure()
    
    sorted_prototypes = torch.zeros_like(epilepsy_mv_module.prototypes)
    for var in range(epilepsy_mv_module.num_variables):
        start_idx = var*epilepsy_mv_module.num_sv_prototypes
        end_idx = (var+1)*epilepsy_mv_module.num_sv_prototypes
        blocks = epilepsy_mv_module.prototypes[:, start_idx:end_idx]
        max_indices = blocks.argmax(dim=1)
        sorted_indices = max_indices.argsort()
        sorted_blocks = blocks[sorted_indices]
        sorted_prototypes[:, start_idx:end_idx] = sorted_blocks

    ax = sns.heatmap(sorted_prototypes.cpu().detach().numpy())

    for col in range(0, sorted_prototypes.shape[1], 4):
        ax.axvline(x=col, color='white', lw=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    min_val = round(sorted_prototypes.min().item(), 2)
    max_val = round(sorted_prototypes.max().item(), 2)
    mid_val = round((min_val + max_val) / 2, 2)

    # Set the colorbar ticks and labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([min_val, mid_val, max_val])

    if save:
        save_name = "visualizations/paper/epilepsy_mv.svg"
        plt.savefig(save_name, dpi=300)
    plt.show()

def epilepsy_projected(save=False):
    fig, axs = plt.subplots(4, 3, figsize=(7, 7))
    classes = ['Epilepsy', 'Walking', 'Running', 'Sawing']
    variable_names = ["Acc (x)", "Acc (y)", "Acc (z)"]
    global_min = float('inf')
    global_max = float('-inf')
    with torch.no_grad():
        prototype_matrix = epilepsy_mv_module.prototypes
        wrapper = epilepsy_mv_module.wrapper
        for i in range(epilepsy_mv_module.num_classes):
            prototype = prototype_matrix[i]
            chunks = prototype.split(wrapper.num_prototypes)
            for j in range(3):
                index = torch.argmax(chunks[j])
                sv_prototype = wrapper.single_variable_prototype_modules[j].prototypes[index]

                for data_matrix, labels in epilepsy_train_dl:
                    single_variable_data = data_matrix[:, :, j].unsqueeze(2).float()
                    embeddings = wrapper.single_variable_prototype_modules[j].encoder(single_variable_data)
                    distances = torch.norm(embeddings - sv_prototype, dim=1)
                    closest_index = torch.argmin(distances).item()
                    closest_point = single_variable_data[closest_index].squeeze(1)
                    ax = axs[int(labels[closest_index]), j]
                    ax.plot(closest_point, c=all_colors[int(labels[closest_index])])
                    if j == 0:
                        ax.set_ylabel(classes[int(labels[closest_index])])
                    if int(labels[closest_index]) < 3:
                        ax.set_xticks([])
                    if j > 0:
                        ax.set_yticks([])
                    if int(labels[closest_index]) == 0:
                        ax.set_title(variable_names[j])

                    local_min = closest_point.min().item()
                    local_max = closest_point.max().item()
                    global_min = min(global_min, local_min)
                    global_max = max(global_max, local_max)
    for ax in axs.flat:
        ax.set_ylim(global_min, global_max)
    fig.align_ylabels()

    if save:
        save_name = "visualizations/paper/epilepsy_projected.pdf"
        plt.savefig(save_name, dpi=300)
    plt.show()

def charactertrajectories_filtered_single_variable_prototypes(save=False):
    with torch.no_grad():
        fig, axs = plt.subplots(1, 3, figsize=(12, 4), sharex=True, sharey=True)
        variable_names = ["x", "y", "Pen Tip Force"]
        colors = all_colors[:4] + [all_colors[6]]
        for j in range(3):
            ax = axs[j]
            ax.set_title(variable_names[j])
            ax.set_xticks([])
            ax.set_yticks([])

            encoder = charactertrajectories_filtered_encoders[j]
            sv_module = charactertrajectories_filtered_sv_prototype_modules.single_variable_prototype_modules[j]
            for data_matrix, labels in charactertrajectories_filtered_test_dl:
                single_variable_data = data_matrix[:, :, j].unsqueeze(2).float()
                embeddings = encoder(single_variable_data)
                embeddings_with_protos = torch.concat([embeddings, sv_module.prototypes], dim=0)
                labels_with_protos = torch.concat([labels, charactertrajectories_filtered_config["num_classes"]*torch.ones((sv_module.prototypes.shape[0],))], dim=0)

                reducer = umap.UMAP()
                embeddings_2d = reducer.fit_transform(embeddings_with_protos)
                e_min, e_max = np.min(embeddings_2d, 0), np.max(embeddings_2d, 0)
                embeddings_2d = (embeddings_2d - e_min) / (e_max - e_min)

                for k in range(4):
                    idx = np.where(labels_with_protos == k)[0]
                    ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=k, color=colors[k], s=50, alpha=0.5)

                idx = np.where(labels_with_protos == 4)[0]
                ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=4, marker="*", edgecolor='black', linewidth=1.5, s=120, c=colors[-1])

        legend_names = charactertrajectories_filtered_config['classes'] + ["Prototype"]
        handles = [plt.Line2D([0], [0], marker='o' if c != 4 else '*', color='w', label=legend_names[c],
                       markersize=10 if c != 4 else 11, markerfacecolor=colors[c], markeredgecolor='None' if c != 4 else 'black') for c in range(5)]
        fig.legend(handles=handles, ncol=5, loc='lower center')
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.1)

        if save:
            plt.savefig("visualizations/paper/charactertrajectories_filtered_sv.pdf", dpi=300)
        plt.show()

def charactertrajectories_filtered_multivariable_prototypes(save=False):
    plt.figure()
    
    sorted_prototypes = torch.zeros_like(charactertrajectories_filtered_mv_module.prototypes)
    for var in range(charactertrajectories_filtered_mv_module.num_variables):
        start_idx = var*charactertrajectories_filtered_mv_module.num_sv_prototypes
        end_idx = (var+1)*charactertrajectories_filtered_mv_module.num_sv_prototypes
        blocks = charactertrajectories_filtered_mv_module.prototypes[:, start_idx:end_idx]
        max_indices = blocks.argmax(dim=1)
        sorted_indices = max_indices.argsort()
        sorted_blocks = blocks[sorted_indices]
        sorted_prototypes[:, start_idx:end_idx] = sorted_blocks

    ax = sns.heatmap(sorted_prototypes.cpu().detach().numpy())

    for col in range(0, sorted_prototypes.shape[1], 4):
        ax.axvline(x=col, color='white', lw=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    min_val = round(sorted_prototypes.min().item(), 2)
    max_val = round(sorted_prototypes.max().item(), 2)
    mid_val = round((min_val + max_val) / 2, 2)

    # Set the colorbar ticks and labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([min_val, mid_val, max_val])

    if save:
        save_name = "visualizations/paper/charactertrajectories_filtered_mv.svg"
        plt.savefig(save_name, dpi=300)
    plt.show()

def charactertrajectories_filtered_projected(save=False):
    classes = ['b', 'd', 'p', "q"]
    fig, axs = plt.subplots(1, 4, figsize=(8, 2))
    with torch.no_grad():
        prototype_matrix = charactertrajectories_filtered_mv_module.prototypes
        wrapper = charactertrajectories_filtered_mv_module.wrapper
        for data_matrix, labels in charactertrajectories_filtered_train_dl:
            for i in range(charactertrajectories_filtered_mv_module.num_classes):
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
                ax.plot(x_int, y_int, c=all_colors[label])
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)
                ax.set_title(classes[label])
    if save:
        save_name = "visualizations/paper/charactertrajectories_filtered_projected.pdf"
        plt.savefig(save_name, dpi=300)
    plt.show()

def charactertrajectories_filtered_silhouette_score_vs_number_of_clusters(save=False):
    variables = ["x", "y", "Pen Tip Force"]
    with torch.no_grad():
        for data_matrix, labels in charactertrajectories_filtered_train_dl:
            data_matrix, labels = data_matrix.to(device), labels.to(device)
            k_values = range(2, 11)
            all_silhouette_scores = np.zeros((len(k_values), 3))
            for var in range(3):
                single_variable_data = data_matrix[:, :, var].unsqueeze(2).float()
                encoder = charactertrajectories_filtered_encoders[var].to(device)
                embeddings = encoder(single_variable_data)
                embeddings = embeddings.cpu().detach().numpy()

                silhouette_scores = []

                for k in k_values:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    labels = kmeans.fit_predict(embeddings)
                    score = silhouette_score(embeddings, labels)
                    silhouette_scores.append(score)
                all_silhouette_scores[:, var] = silhouette_scores
                plt.plot(k_values, all_silhouette_scores[:, var], marker='o', label=variables[var], color=other_colors[var])

    plt.title('Silhouette Score vs. Number of Clusters')
    plt.xlabel('Number of Clusters')
    plt.ylabel('Silhouette Score')
    plt.xticks(k_values)
    plt.legend()
    
    if save:
        save_name = "visualizations/paper/charactertrajectories_filtered_silhouette.pdf"
        plt.savefig(save_name, dpi=300)
    plt.show()

def charactertrajectories_filtered_no_contrastive_single_variable_prototypes(save=False):
    return

def basicmotions_single_variable_prototypes(save=False):
    with torch.no_grad():
        fig, axs = plt.subplots(2, 3, figsize=(12, 8), sharex=True, sharey=True)
        variable_names = ["Acc (x)", "Acc (y)", "Acc (z)", "Gyro (x)", "Gyro (y)", "Gyro (z)"]
        colors = all_colors[:4] + [all_colors[6]]
        for i in range(2):
            for j in range(3):
                ax = axs[i][j]
                ax.set_title(variable_names[3*i + j])
                ax.set_xticks([])
                ax.set_yticks([])

                encoder = basicmotions_encoders[j]
                sv_module = basicmotions_sv_prototype_modules.single_variable_prototype_modules[j]
                for data_matrix, labels in basicmotions_test_dl:
                    single_variable_data = data_matrix[:, :, j].unsqueeze(2).float()
                    embeddings = encoder(single_variable_data)
                    embeddings_with_protos = torch.concat([embeddings, sv_module.prototypes], dim=0)
                    labels_with_protos = torch.concat([labels, basicmotions_config["num_classes"]*torch.ones((sv_module.prototypes.shape[0],))], dim=0)

                    reducer = umap.UMAP()
                    embeddings_2d = reducer.fit_transform(embeddings_with_protos)
                    e_min, e_max = np.min(embeddings_2d, 0), np.max(embeddings_2d, 0)
                    embeddings_2d = (embeddings_2d - e_min) / (e_max - e_min)

                    for k in range(4):
                        idx = np.where(labels_with_protos == k)[0]
                        ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=k, color=colors[k], s=50, alpha=0.5)

                    idx = np.where(labels_with_protos == 4)[0]
                    ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=4, marker="*", edgecolor='black', linewidth=1.5, s=120, c=colors[-1])

        legend_names = basicmotions_config['classes'] + ["Prototype"]
        handles = [plt.Line2D([0], [0], marker='o' if c != 4 else '*', color='w', label=legend_names[c],
                       markersize=10 if c != 4 else 11, markerfacecolor=colors[c], markeredgecolor='None' if c != 4 else 'black') for c in range(5)]
        fig.legend(handles=handles, ncol=5, loc='lower center')
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.05)

        if save:
            plt.savefig("visualizations/paper/basicmotions_sv.pdf", dpi=300)
        plt.show()

def basicmotions_multivariable_prototypes(save=False):
    plt.figure()
    
    sorted_prototypes = torch.zeros_like(basicmotions_mv_module.prototypes)
    for var in range(basicmotions_mv_module.num_variables):
        start_idx = var*basicmotions_mv_module.num_sv_prototypes
        end_idx = (var+1)*basicmotions_mv_module.num_sv_prototypes
        blocks = basicmotions_mv_module.prototypes[:, start_idx:end_idx]
        max_indices = blocks.argmax(dim=1)
        sorted_indices = max_indices.argsort()
        sorted_blocks = blocks[sorted_indices]
        sorted_prototypes[:, start_idx:end_idx] = sorted_blocks

    ax = sns.heatmap(sorted_prototypes.cpu().detach().numpy())

    for col in range(0, sorted_prototypes.shape[1], 4):
        ax.axvline(x=col, color='white', lw=2)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    min_val = round(sorted_prototypes.min().item(), 2)
    max_val = round(sorted_prototypes.max().item(), 2)
    mid_val = round((min_val + max_val) / 2, 2)

    # Set the colorbar ticks and labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([min_val, mid_val, max_val])

    if save:
        save_name = "visualizations/paper/basicmotions_mv.svg"
        plt.savefig(save_name, dpi=300)
    plt.show()

def basicmotions_projected(save=False):
    fig, axs = plt.subplots(4, 3, figsize=(5.25, 7))
    classes = ['Standing', 'Walking', 'Running', 'Badminton']
    variable_names = ["Acc (x)", "Acc (y)", "Acc (z)"]
    global_min = float('inf')
    global_max = float('-inf')
    with torch.no_grad():
        prototype_matrix = basicmotions_mv_module.prototypes
        wrapper = basicmotions_mv_module.wrapper
        for i in range(basicmotions_mv_module.num_classes):
            prototype = prototype_matrix[i]
            chunks = prototype.split(wrapper.num_prototypes)
            for j in range(3):
                index = torch.argmax(chunks[j])
                sv_prototype = wrapper.single_variable_prototype_modules[j].prototypes[index]

                for data_matrix, labels in basicmotions_train_dl:
                    single_variable_data = data_matrix[:, :, j].unsqueeze(2).float()
                    embeddings = wrapper.single_variable_prototype_modules[j].encoder(single_variable_data)
                    distances = torch.norm(embeddings - sv_prototype, dim=1)
                    closest_index = torch.argmin(distances).item()
                    closest_point = single_variable_data[closest_index].squeeze(1)
                    ax = axs[int(labels[closest_index]), j]
                    ax.plot(closest_point, c=all_colors[int(labels[closest_index])])
                    if j == 0:
                        ax.set_ylabel(classes[int(labels[closest_index])])
                    if int(labels[closest_index]) < 3:
                        ax.set_xticks([])
                    if j > 0:
                        ax.set_yticks([])
                    if int(labels[closest_index]) == 0:
                        ax.set_title(variable_names[j])

                    local_min = closest_point.min().item()
                    local_max = closest_point.max().item()
                    global_min = min(global_min, local_min)
                    global_max = max(global_max, local_max)
    for ax in axs.flat:
        ax.set_ylim(global_min, global_max)
    fig.align_ylabels()

    if save:
        save_name = "visualizations/paper/basicmotions_projected.pdf"
        plt.savefig(save_name, dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the figure or not")
    args = parser.parse_args()

    simulated_one_stage_one_pattern(save=args.save)