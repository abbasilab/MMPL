import argparse

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
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
simulated_test_ds = get_ds(get_test_path_from_dataset("simulated_6400"), simulated_config['class_to_index'])
simulated_train_dl = torch.utils.data.DataLoader(simulated_train_ds, len(simulated_train_ds), shuffle=True)
simulated_test_dl = torch.utils.data.DataLoader(simulated_test_ds, len(simulated_test_ds), shuffle=True)
simulated_encoders = load_encoders(simulated_config)
simulated_sv_prototype_modules = load_single_variable_prototypes_wrapper(simulated_config)
simulated_mv_module = load_multivariable_prototypes(simulated_config)

tab10 = plt.cm.get_cmap("tab10")
all_colors = list(tab10.colors)

plt.rcParams['font.family'] = 'Arial'

def simulated_dataset_generation(save=False):
    return

def simulated_single_variable_prototypes(save=False):
    return

def simulated_multivariable_prototypes(save=False):
    return

def simulated_projections(save=False):
    return

def simulated_no_contrastive_single_variable_prototypes(save=False):
    return

def simulated_silhouette_score_vs_number_of_clusters(save=False):
    return

def simulated_one_stage(save=False):
    return

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
    return

def epilepsy_projections(save=False):
    return

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
    return

def charactertrajectories_filtered_projected(save=False):
    # Show actual characters
    return

def charactertrajectories_filtered_silhouette_score_vs_number_of_clusters(save=False):
    return

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
    return

def basicmotions_projection(save=False):
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the figure or not")
    args = parser.parse_args()

    basicmotions_single_variable_prototypes(save=args.save)