import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms
from matplotlib.lines import Line2D
import seaborn as sns

from ..data.data import get_ds, filter_classes
from ..models.charactertrajectories_filtered.single_variable_encoding import LSTMEncoder
from ..models.single_variables import EncodingModule, SingleVariableModulesWrapper
from ..models.multivariable import MultivariableModule
from .umap_visualizer import UMAPLatent
from .utils import get_multivariable_prototype_classes

def sv_prototypes_visualization():
    class_to_index = {
        "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
        "9":9, "10":10, "11":11, "12":12, "13":13, "14":14, "15":15,
        "16":16, "17":17, "18":18, "19":19, "20":20
    }
    index_to_class = {}
    for i in range(1, 21):
        index_to_class[i] = str(i)
    train_ds, test_ds = get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TRAIN.ts", class_to_index), get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TEST.ts", class_to_index)

    encoding_module = EncodingModule(torch.nn.ModuleList([LSTMEncoder(119, 10) for _ in range(3)]))
    encoding_module.load_state_dict(torch.load("models/charactertrajectories_filtered/enc.dat"))

    sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=3, num_classes=4, hidden=10, num_prototypes=4)
    for i in range(len(sv_modules_wrapper.single_variable_modules)):
        module = sv_modules_wrapper.single_variable_modules[i]
        module.encoder = encoding_module.module_list[i]
    sv_modules_wrapper.load_state_dict(torch.load("models/charactertrajectories_filtered/sv_modules_wrapper.dat"))

    fig, axes = plt.subplots(1, 3, figsize=(5,2.5))
    colors = ["red", "blue", "green", "orange", "magenta"]
    titles = ["x", "y", "Pen Tip Force"]

    b = Line2D([], [], color="white", marker='o', markerfacecolor=colors[0], markersize=7.5)
    d = Line2D([], [], color="white", marker='o', markerfacecolor=colors[1], markersize=7.5)
    p = Line2D([], [], color="white", marker='o', markerfacecolor=colors[2], markersize=7.5)
    q = Line2D([], [], color="white", marker='o', markerfacecolor=colors[3], markersize=7.5)
    prototype = Line2D([], [], color="white", marker='*', markerfacecolor=colors[4], markeredgecolor="black", markeredgewidth=0.5, markersize=7.5)
    leg = fig.legend(handles=[b, d, p, q, prototype],
               labels=["b", "d", "p", "q", "Prototype"], loc="lower center", ncol=5, fontsize=7.5)
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
            ax = axes[i]
            ax.set_xlabel("UMAP-1", fontsize=6)
            ax.set_ylabel("UMAP-2", fontsize=6)
            ax.set_title(titles[i], y=0.0, pad=-35, fontsize=7.5)
            for j in range(X.shape[0]):
                classif = out[j]
                if classif != 4:
                    ax.plot(X[j, 0], X[j, 1], 'o', color=colors[int(classif.item())], alpha=0.3)
                else:
                    ax.plot(X[j, 0], X[j, 1], '*', color=colors[int(classif.item())], alpha=1.0, markersize=7.5, markeredgecolor="black", markeredgewidth=0.5)
            ax.xaxis.set_tick_params(labelsize=6)
            ax.yaxis.set_tick_params(labelsize=6)
            ax.locator_params(axis="x", nbins=3)
            ax.locator_params(axis="y", nbins=3)
            ax.set_box_aspect(1)
    fig.tight_layout()
    fig.subplots_adjust(bottom=0.165)
    fig.align_ylabels(axes)
    plt.savefig("figures/charactertrajectories_filtered/sv_latent_space.pdf", dpi=300)

def heatmap():
    class_to_index = {
        "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
        "9":9, "10":10, "11":11, "12":12, "13":13, "14":14, "15":15,
        "16":16, "17":17, "18":18, "19":19, "20":20
    }
    index_to_class = {}
    for i in range(1, 21):
        index_to_class[i] = str(i)
    train_ds, test_ds = get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TRAIN.ts", class_to_index), get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TEST.ts", class_to_index)
    filtered_train, filtered_test = filter_classes(train_ds, [2, 4, 12, 13]), filter_classes(test_ds, [2, 4, 12, 13])
    index_to_class = {
        0:"b", 1:"d", 2:"p", 3:"q"
    }
    encoding_module = EncodingModule(torch.nn.ModuleList([LSTMEncoder(119, 10) for _ in range(3)]))
    encoding_module.load_state_dict(torch.load("models/charactertrajectories_filtered/enc.dat"))

    sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=3, num_classes=4, hidden=10, num_prototypes=4)
    for i in range(len(sv_modules_wrapper.single_variable_modules)):
        module = sv_modules_wrapper.single_variable_modules[i]
        module.encoder = encoding_module.module_list[i]
    sv_modules_wrapper.load_state_dict(torch.load("models/charactertrajectories_filtered/sv_modules_wrapper.dat"))

    model = MultivariableModule(single_variable_modules=sv_modules_wrapper.single_variable_modules, \
                                 num_variables=3, hidden=12, num_classes=4, num_prototypes=4)
    model.load_state_dict(torch.load("models/charactertrajectories_filtered/multivariable_module.dat"))

    protos = model.aggregate_prototype_layer.protos
    min_val = protos.min()
    max_val = protos.max()
    scaled_protos = (protos - min_val) / (max_val - min_val)

    classes = get_multivariable_prototype_classes(model, filtered_train)
    classes, scaled_protos = zip(*sorted(zip(classes, scaled_protos)))
    scaled_protos = torch.stack(scaled_protos)
    class_names = [index_to_class[i] for i in classes]

    xticklabels = [
        "x", "", "", "",
        "y", "", "", "",
        "Pen Tip Force", "", "", ""
    ]
    fig = plt.figure(figsize=(6,3))
    sns.set(font_scale=0.75)
    ax = sns.heatmap(scaled_protos.detach().numpy(), xticklabels=xticklabels, yticklabels=class_names, cbar_kws={"ticks":[0,0.5,1]}, )
    plt.yticks(rotation=0)
    plt.xticks(rotation=0, ha="center")
    dx = 33/72.; dy = 0/72. 
    offset = transforms.ScaledTranslation(dx, dy, fig.dpi_scale_trans)
    for label in ax.xaxis.get_majorticklabels():
        label.set_transform(label.get_transform() + offset)

    for i in range(model.num_variables):
        ax.axvline(i*model.single_variable_modules[i].num_prototypes, color="white", lw=2.0)

    plt.savefig("figures/charactertrajectories_filtered/heatmap.pdf", dpi=300)

def multivariable_prototypes_closest_training_point():
    class_to_index = {
        "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
        "9":9, "10":10, "11":11, "12":12, "13":13, "14":14, "15":15,
        "16":16, "17":17, "18":18, "19":19, "20":20
    }
    index_to_class = {}
    for i in range(1, 21):
        index_to_class[i] = str(i)
    train_ds, test_ds = get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TRAIN.ts", class_to_index), get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TEST.ts", class_to_index)
    filtered_train, filtered_test = filter_classes(train_ds, [2, 4, 12, 13]), filter_classes(test_ds, [2, 4, 12, 13])
    index_to_class = {
        0:"b", 1:"d", 2:"p", 3:"q"
    }
    encoding_module = EncodingModule(torch.nn.ModuleList([LSTMEncoder(119, 10) for _ in range(3)]))
    encoding_module.load_state_dict(torch.load("models/charactertrajectories_filtered/enc.dat"))

    sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=3, num_classes=4, hidden=10, num_prototypes=4)
    for i in range(len(sv_modules_wrapper.single_variable_modules)):
        module = sv_modules_wrapper.single_variable_modules[i]
        module.encoder = encoding_module.module_list[i]
    sv_modules_wrapper.load_state_dict(torch.load("models/charactertrajectories_filtered/sv_modules_wrapper.dat"))

    model = MultivariableModule(single_variable_modules=sv_modules_wrapper.single_variable_modules, \
                                 num_variables=3, hidden=12, num_classes=4, num_prototypes=4)
    model.load_state_dict(torch.load("models/charactertrajectories_filtered/multivariable_module.dat"))

    protos = model.aggregate_prototype_layer.protos
    min_val = protos.min()
    max_val = protos.max()
    scaled_protos = (protos - min_val) / (max_val - min_val)

    classes = get_multivariable_prototype_classes(model, filtered_train)
    classes, scaled_protos = zip(*sorted(zip(classes, scaled_protos)))
    scaled_protos = torch.stack(scaled_protos)
    class_names = [index_to_class[i] for i in classes]
    vars = ["x", "y", "Pen Tip Force"]

    colors = ["red", "blue", "green", "orange"]

    data_train = torch.utils.data.DataLoader(filtered_train, len(filtered_train), True)
    with torch.no_grad():
        for train_sample in data_train:
            inp, out = train_sample[0].detach(), train_sample[1].detach()

            fig, axes = plt.subplots(model.num_prototypes, model.num_variables)

            ylims = [[float("inf"), -float("inf")] for _ in range(model.num_variables)]
            for i in range(model.num_prototypes):
                mv_prototype = scaled_protos[i].detach().numpy()
                single_variable_blocks = np.split(mv_prototype, model.num_variables)
                for j in range(len(single_variable_blocks)):
                    block = single_variable_blocks[j]
                    index = np.argmax(block)
                    sv_prototype = model.single_variable_modules[j].protolayer.prototype_matrix[index]
                    embeddings = sv_modules_wrapper.single_variable_modules[j].encoder(inp[:,:,j].unsqueeze(2).float())
                    best_index = -1
                    best_dist = float("inf")
                    for k in range(len(embeddings)):
                        dist = torch.norm(embeddings[k] - sv_prototype)
                        if dist < best_dist:
                            best_index = k
                            best_dist = dist
                    closest_point_embedding = embeddings[best_index]
                    closest_point = inp[best_index, :, j]
                    ylims[j][0] = min(ylims[j][0], float(min(closest_point)))
                    ylims[j][1] = max(ylims[j][1], float(max(closest_point)))

                    ax = axes[i][j]
                    ax.plot(closest_point.detach().numpy(), linewidth=0.5, c=colors[i])
                    plt.subplots_adjust(hspace=0.5, wspace=0.45)
                    ax.tick_params(length=0)
                    ax.xaxis.set_tick_params(labelsize=6)
                    ax.yaxis.set_tick_params(labelsize=6)

                    if j == 0:
                        ax.set_ylabel(class_names[i], labelpad=25.0, rotation="horizontal", fontsize=7.5)

                    if i == 3:
                        ax.set_xlabel(vars[j], fontsize=7.5)

            for j in range(model.num_variables):
                for i in range(model.num_prototypes):
                    ax = axes[i][j]
                    ax.set_ylim(ylims[j][0], ylims[j][1])

            fig.align_ylabels()
            plt.savefig("figures/charactertrajectories_filtered/mv_closest_point.pdf", dpi=300)





if __name__ == "__main__":
    sv_prototypes_visualization()