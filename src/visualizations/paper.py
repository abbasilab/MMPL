import argparse
import math
import random

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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
simulated_train_dl_unshuffled = torch.utils.data.DataLoader(simulated_train_ds, len(simulated_train_ds), shuffle=False)
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

no_contrastive_config = get_comparison_config_from_dataset("no_contrastive", "simulated_6400")
no_contrastive_sv_prototype_modules = load_no_contrastive_single_variable_prototypes(no_contrastive_config)
no_contrastive_encoders = []
for module in no_contrastive_sv_prototype_modules.single_variable_prototype_modules:
    no_contrastive_encoders.append(module.encoder)
# no_contrastive_mv_module = load_no_contrastive_multivariable_prototypes(no_contrastive_config)

tab10 = plt.cm.get_cmap("tab10")
all_colors = list(tab10.colors)

dark2 = plt.cm.get_cmap("Dark2")
other_colors = list(dark2.colors)

tab20 = plt.cm.get_cmap('tab20', 20)
tab20b = plt.cm.get_cmap('tab20b', 20)
tab20c = plt.cm.get_cmap('tab20c', 20)
sixty_colors = list(tab20.colors) + list(tab20b.colors) + list(tab20c.colors)

plt.rcParams['font.family'] = 'Arial'
plt.rcParams["font.size"] = "10"

def model_input_time_series(save=False):
    fig, axs = plt.subplots(3, figsize=(8, 8))
    for data_matrix, labels in epilepsy_train_dl:
        for i in range(3):
            sv_data = data_matrix[:, :, i]
            index = random.randint(0, len(sv_data) - 1)
            ax = axs[i]
            ax.plot(sv_data[index, :].detach().numpy(), c=other_colors[i])
            ax.set_xticks([])
            ax.set_yticks([])
    if save:
        plt.savefig("visualizations/paper/model_input_time_series.svg")
    plt.show()


def simulated_dataset_generation(save=False):
    fig, axs = plt.subplots(4, 3, figsize=(9, 8))

    for i in range(4):
        for j in range(3):
            ax = axs[i, j]
            ax.set_ylim(-1.5, 1.5)
            # if j == 0:
            #     ax.set_yticks([-1, 0, 1])
            # else:
            #     ax.set_yticks([])
            # if i == 3:
            #     ax.set_xticks([0, 50, 100])
            # else:
            #     ax.set_xticks([])
            ax.set_yticks([-1, 0, 1])

            if i == 3:
                ax.set_xlabel(f"Variable {j+1}", fontsize=16)
            if j == 0:
                ax.set_ylabel(f"Pattern {i+1}", rotation=0, fontsize=16, labelpad=30)
    
    # Variable 1
    g1 = np.sin(5 * np.linspace(0, 2 * math.pi, 200))[0:20]
    g2 = np.sin(5 * np.linspace(0, 2 * math.pi, 200))[20:40]

    ax = axs[0, 0]
    shiftone = np.random.randint(0, 55)
    series = np.concatenate([np.zeros(shiftone), g1, np.zeros(5), g1, np.zeros(100 - (shiftone + 45))])
    ax.plot(series, c=all_colors[0])

    ax = axs[1, 0]
    shiftone = np.random.randint(0, 55)
    series = np.concatenate([np.zeros(shiftone), g1, np.zeros(5), g2, np.zeros(100 - (shiftone + 45))])
    ax.plot(series, c=all_colors[1])

    ax = axs[2, 0]
    shiftone = np.random.randint(0, 55)
    series = np.concatenate([np.zeros(shiftone), g2, np.zeros(5), g1, np.zeros(100 - (shiftone + 45))])
    ax.plot(series, c=all_colors[2])

    ax = axs[3, 0]
    shiftone = np.random.randint(0, 55)
    series = np.concatenate([np.zeros(shiftone), g2, np.zeros(5), g2, np.zeros(100 - (shiftone + 45))])
    ax.plot(series, c=all_colors[3])


    # Variable 2
    g1 = np.sin(5 * np.linspace(0, 2 * math.pi, 200))[0:20]

    ax = axs[0, 1]
    shiftone = np.random.randint(0, 10)
    series = np.concatenate([np.zeros(shiftone), 1.0*g1, np.zeros(100-(shiftone+20))])
    ax.plot(series, c=all_colors[0])

    ax = axs[1, 1]
    shiftone = np.random.randint(25, 35)
    series = np.concatenate([np.zeros(shiftone), 1.0*g1, np.zeros(100-(shiftone+20))])
    ax.plot(series, c=all_colors[1])

    ax = axs[2, 1]
    shiftone = np.random.randint(50, 60)
    series = np.concatenate([np.zeros(shiftone), 1.0*g1, np.zeros(100-(shiftone+20))])
    ax.plot(series, c=all_colors[2])

    ax = axs[3, 1]
    shiftone = np.random.randint(75, 80)
    series = np.concatenate([np.zeros(shiftone), 1.0*g1, np.zeros(100-(shiftone+20))])
    ax.plot(series, c=all_colors[3])

    # Variable 3
    ax = axs[0, 2]
    series = np.sin(0*np.linspace(0, 2 * math.pi,100))
    ax.plot(series, c=all_colors[0])

    ax = axs[1, 2]
    series = np.sin(1*np.linspace(0, 2 * math.pi,100) + 2*math.pi)
    ax.plot(series, c=all_colors[1])

    ax = axs[2, 2]
    series = np.sin(2*np.linspace(0, 2 * math.pi,100) + 2*math.pi)
    ax.plot(series, c=all_colors[2])

    ax = axs[3, 2]
    series = np.sin(3*np.linspace(0, 2 * math.pi,100) + 2*math.pi)
    ax.plot(series, c=all_colors[3])

    fig.subplots_adjust(hspace=0.5)

    if save:
        save_name = "visualizations/paper/simulated_generation.eps"
        plt.savefig(save_name, dpi=300)

    plt.show()

def simulated_dataset_example(save=False):
    fig = plt.figure(figsize=(8, 8))
    gs = gridspec.GridSpec(5, 4, height_ratios=[1, 1, 0.01, 1, 1])  # Adjust the 0.1 to reduce the ellipsis row height
    grid_shape = (5, 4)  # 5 rows (4 for data, 1 for ellipsis), 4 columns
    display_classes = [0, 21, 42, 63]
    row_labels = [f"Class {i+1}" for i in display_classes]
    col_labels = [f"Variable {i+1}" for i in range(4)]
    ellipsis_row = 2
    with torch.no_grad():
        for data_matrix, labels in simulated_train_dl_unshuffled:
            axes = []
            for i, class_idx in enumerate(display_classes):
                row = i if i < ellipsis_row else i + 1  # Adjust row to skip ellipsis row
                for j in range(4):

                    ax = plt.subplot(gs[row, j])  # Use 'row' instead of 'i'
                    ax.set_ylim(-1.5, 1.5)
                    ax.set_yticks([-1, 0, 1])
                    ax.set_xticks([0, 50, 100])
                    axes.append(ax)

                    if j == 3:
                        ax.plot(data_matrix[class_idx*100, :, j], c="grey")
                    else:
                        patterns = get_single_variable_patterns_from_labels(labels, j)
                        pattern = int(patterns[class_idx*100])
                        ax.plot(data_matrix[class_idx*100, :, j], c=all_colors[pattern])
                    
                    if j == 0:
                        ax.set_ylabel(row_labels[i], fontsize=16)
                    if row == grid_shape[0] - 1:
                        ax.set_xlabel(col_labels[j], fontsize=16)
                    
                    # if row < grid_shape[0] - 1:
                    #     ax.set_xticks([])
                    
                    # if j > 0:
                    #     ax.set_yticks([])

            # Add vertical ellipsis in the gap
            for i in range(0, len(axes), 4):
                min_val = min(ax.get_ylim()[0] for ax in axes[i:i + 4])
                max_val = max(ax.get_ylim()[1] for ax in axes[i:i + 4])
                for ax in axes[i:i + 4]:
                    ax.set_ylim(min_val, max_val)
            for j in range(4):
                ax = plt.subplot(gs[ellipsis_row, j])
                ax.annotate('...', xy=(0.5, 0.5), xycoords='axes fraction', fontsize=20, ha='center')
                ax.axis('off')
            plt.tight_layout()
            if save:
                save_name = "visualizations/paper/simulated_example.eps"
                plt.savefig(save_name, dpi=300)
            plt.show()


def simulated_single_variable_prototypes(save=False):
    class_to_pattern_map = get_class_to_pattern_map()
    with torch.no_grad():
        classes = [i for i in range(64)]
        colors = all_colors[:4] + [all_colors[6]]
        variable_names = ["Variable 1", "Variable 2", "Variable 3", "Variable 4"]
        pattern_labels = ["Pattern 1", "Pattern 2", "Pattern 3", "Pattern 4", "Prototype"]

        # Change here: Modified to 1x4 layout
        fig, axs = plt.subplots(1, 4, figsize=(16, 4), sharex=True, sharey=True)

        for i in range(4):
            # Change here: Adjusted indexing
            ax = axs[i]
            ax.set_title(variable_names[i], fontsize=16)
            ax.set_xticks([])
            ax.set_yticks([])

            encoder = simulated_encoders[i]
            for data_matrix, labels, in simulated_test_dl:
                single_variable_data = data_matrix[:, :, i].unsqueeze(2).float()
                embeddings = encoder(single_variable_data)
                embeddings = torch.concat([embeddings, simulated_sv_prototype_modules.single_variable_prototype_modules[i].prototypes], dim=0)
                labels = torch.concat([labels, len(classes)*torch.ones((simulated_sv_prototype_modules.single_variable_prototype_modules[i].prototypes.shape[0],))], dim=0)
                reducer = umap.UMAP()
                embeddings_2d = reducer.fit_transform(embeddings.cpu())
                e_min, e_max = np.min(embeddings_2d, 0), np.max(embeddings_2d, 0)
                embeddings_2d = (embeddings_2d - e_min) / (e_max - e_min)

                if i == 3:
                    ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='grey', s=20.0)
                else:
                    for k, label in enumerate(classes):
                        idx = np.where(labels == label)[0]
                        pattern = int(class_to_pattern_map[label][i])
                        ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=pattern_labels[pattern], c=colors[pattern], alpha=0.2, s=20.0)

                    idx = np.where(labels == len(classes))[0]
                    ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label="Prototype", marker="*", edgecolor='black', linewidth=1.5, s=200, c=colors[-1])

        legend_names = pattern_labels + ["Prototype"]
        handles = [plt.Line2D([0], [0], marker='o' if c != 4 else '*', color='w', label=legend_names[c],
                       markersize=10 if c != 4 else 11, markerfacecolor=colors[c], markeredgecolor='None' if c != 4 else 'black') for c in range(5)]
        fig.legend(handles=handles, ncol=5, loc='lower center')
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.1)

    if save:
        save_name = "visualizations/paper/simulated_sv.eps"
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
    plt.figure(figsize=(8, 8))
    prototypes = simulated_mv_module.prototypes
    prototypes_list = [prototypes[i, :] for i in range(prototypes.shape[0])]
    sorted_prototypes = sorted(prototypes_list, key=sorting_key)
    sorted_prototypes_tensor = torch.stack(sorted_prototypes)
    ax = sns.heatmap(sorted_prototypes_tensor.cpu().detach().numpy())

    for col in range(0, sorted_prototypes_tensor.shape[1], 4):
        ax.axvline(x=col, color='white', lw=2)

    # Set the x-ticks for variables
    x_ticks = [2, 6, 10, 14]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(['Variable 1', 'Variable 2', 'Variable 3', 'Variable 4'], fontsize=16)
    ax.tick_params(axis='x', length=0)

    # Set the y-ticks for groups
    y_ticks = [8, 24, 40, 56]
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(['Group 1', 'Group 2', 'Group 3', 'Group 4'], fontsize=16)
    ax.tick_params(axis='y', length=0)

    min_val = round(sorted_prototypes_tensor.min().item(), 2)
    max_val = round(sorted_prototypes_tensor.max().item(), 2)
    mid_val = round((min_val + max_val) / 2, 2)

    # Set the colorbar ticks and labels
    cbar = ax.collections[0].colorbar
    cbar.set_ticks([min_val, mid_val, max_val])

    if save:
        save_name = "visualizations/paper/simulated_mv.eps"
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

                    ax.plot(closest_point.cpu(), c=other_colors[j])
                    
                    # local_min = closest_point.min().item()
                    # local_max = closest_point.max().item()
                    # global_min = min(global_min, local_min)
                    # global_max = max(global_max, local_max)
                    ax.set_ylim(-1.5, 1.5)

                    if j == 0:
                        ax.set_ylabel(f'Class {classes_of_interest[i]}', fontsize=16)
                    # if i < len(classes_of_interest) - 1:
                    #     ax.set_xticks([])
                    # if j > 0:
                    #     ax.set_yticks([])
                    if i == 3:
                        ax.set_xlabel(variable_names[j], fontsize=16)

        # for ax in axs.flat:
        #     ax.set_ylim(global_min, global_max)

    plt.tight_layout()
    fig.align_ylabels()
    if save:
        save_name = "visualizations/paper/simulated_projected.eps"
        plt.savefig(save_name, dpi=300)
    plt.show()

def simulated_no_contrastive_single_variable_prototypes(save=False):
    class_to_pattern_map = get_class_to_pattern_map()
    with torch.no_grad():
        classes = [i for i in range(64)]
        colors = all_colors[:4] + [all_colors[6]]
        variable_names = ["Variable 1", "Variable 2", "Variable 3", "Variable 4"]
        pattern_labels = ["Pattern 1", "Pattern 2", "Pattern 3", "Pattern 4", "Prototype"]

        fig, axs = plt.subplots(2, 2, figsize=(8, 8), sharex=True, sharey=True)

        for i in range(2):
            for j in range(2):
                variable = i*2 + j

                ax = axs[i, j]
                ax.set_title(variable_names[variable], fontsize=16)
                ax.set_xticks([])
                ax.set_yticks([])

                encoder = no_contrastive_encoders[variable]
                for data_matrix, labels, in simulated_test_dl:
                    single_variable_data = data_matrix[:, :, variable].unsqueeze(2).float()
                    embeddings = encoder(single_variable_data)
                    embeddings = torch.concat([embeddings, no_contrastive_sv_prototype_modules.single_variable_prototype_modules[variable].prototypes], dim=0)
                    labels = torch.concat([labels, len(classes)*torch.ones((no_contrastive_sv_prototype_modules.single_variable_prototype_modules[variable].prototypes.shape[0],))], dim=0)
                    reducer = umap.UMAP()
                    embeddings_2d = reducer.fit_transform(embeddings.cpu())
                    e_min, e_max = np.min(embeddings_2d, 0), np.max(embeddings_2d, 0)
                    embeddings_2d = (embeddings_2d - e_min) / (e_max - e_min)

                    if variable == 3:
                        ax.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1], c='grey', s=5.0)
                    else:
                        for k, label in enumerate(classes):
                            idx = np.where(labels == label)[0]
                            pattern = int(class_to_pattern_map[label][variable])
                            ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=pattern_labels[pattern], c=colors[pattern], alpha=0.2, s=5.0)

                        idx = np.where(labels == len(classes))[0]
                        ax.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label="Prototype", marker="*", edgecolor='black', linewidth=1.5, s=200, c=colors[-1])

        legend_names = pattern_labels + ["Prototype"]
        handles = [plt.Line2D([0], [0], marker='o' if c != 4 else '*', color='w', label=legend_names[c],
                       markersize=10 if c != 4 else 11, markerfacecolor=colors[c], markeredgecolor='None' if c != 4 else 'black') for c in range(5)]
        fig.legend(handles=handles, ncol=5, loc='lower center')
        plt.tight_layout()
        fig.subplots_adjust(bottom=0.1)


    if save:
        save_name = "visualizations/paper/no_contrastive_simulated_sv.eps"
        plt.savefig(save_name, dpi=300)

    plt.show()

def simulated_no_contrastive_multivariable_prototypes(save=False):
    plt.figure()
    prototypes = no_contrastive_mv_module.prototypes
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
        save_name = "visualizations/paper/no_contrastive_mv.eps"
        plt.savefig(save_name, dpi=300)
    plt.show()

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

    plt.xlabel('Number of Clusters', fontsize=16)
    plt.ylabel('Silhouette Score', fontsize=16)
    plt.xticks(k_values, fontsize=12)
    plt.yticks(fontsize=12)
    plt.legend()
    
    if save:
        save_name = "visualizations/paper/simulated_silhouette.eps"
        plt.savefig(save_name, dpi=300)
    plt.show()

def simulated_accuracy_vs_number_of_clusters(save=False):
    num_protos = range(2, 11)
    accuracies = [0.96953125, 0.99296875, 0.9984375, 0.99875, 0.9990625, 0.99828125, 0.99515625, 0.99765625, 0.9928125]
    plt.plot(num_protos, accuracies, c="black")
    plt.xlabel("Number of Univariate Prototypes", fontsize=16)
    plt.ylabel("Classification Accuracy", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if save:
        save_name = "visualizations/paper/simulated_accuracy.eps"
        plt.savefig(save_name, dpi=300)
    plt.show()

def simulated_number_of_clusters(save=False):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

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
                ax1.plot(k_values, all_silhouette_scores[:, var], marker='o', label=variables[var], c=other_colors[var])

    ax1.set_xlabel('Number of Clusters', fontsize=16)
    ax1.set_ylabel('Silhouette Score', fontsize=16)
    ax1.legend()

    # Plot 2: Simulated Accuracy vs Number of Clusters
    num_protos = range(2, 11)
    accuracies = [0.96953125, 0.99296875, 0.9984375, 0.99875, 0.9990625, 0.99828125, 0.99515625, 0.99765625, 0.9928125]
    ax2.plot(num_protos, accuracies, c="black", marker='o')
    ax2.set_xlabel("Number of Univariate Prototypes", fontsize=16)
    ax2.set_ylabel("Accuracy", fontsize=16)

    # Save or show
    if save:
        plt.savefig("visualizations/paper/simulated_number_of_clusters.eps", dpi=300)
    plt.show()

def simulated_one_stage(save=False):
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
            plt.savefig("visualizations/paper/epilepsy_sv.eps", dpi=300)
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
        save_name = "visualizations/paper/epilepsy_mv.eps"
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
        save_name = "visualizations/paper/epilepsy_projected.eps"
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
            plt.savefig("visualizations/paper/charactertrajectories_filtered_sv.eps", dpi=300)
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
        save_name = "visualizations/paper/charactertrajectories_filtered_projected.eps"
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

    plt.xlabel('Number of Clusters', fontsize=16)
    plt.ylabel('Silhouette Score', fontsize=16)
    plt.xticks(k_values)
    plt.legend()
    
    if save:
        save_name = "visualizations/paper/charactertrajectories_filtered_silhouette.eps"
        plt.savefig(save_name, dpi=300)
    plt.show()

def charactertrajectories_filtered_accuracy_vs_number_of_clusters(save=False):
    num_protos = range(2, 11)
    accuracies = [0.7636363636363637, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]
    plt.plot(num_protos, accuracies, c="black")
    plt.xlabel("Number of Univariate Prototypes", fontsize=16)
    plt.ylabel("Classification Accuracy", fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    if save:
        save_name = "visualizations/paper/charactertrajectories_filtered_accuracy.eps"
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
            plt.savefig("visualizations/paper/basicmotions_sv.eps", dpi=300)
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
        save_name = "visualizations/paper/basicmotions_mv.eps"
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
        save_name = "visualizations/paper/basicmotions_projected.eps"
        plt.savefig(save_name, dpi=300)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the figure or not")
    args = parser.parse_args()

    simulated_single_variable_prototypes(save=args.save)