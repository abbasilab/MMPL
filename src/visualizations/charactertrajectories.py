import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import seaborn as sns

from ..data.data import get_ds, filter_classes
from ..models.single_variables import SingleVariableModulesWrapper
from ..models.multivariable import MultivariableModule
from .umap_visualizer import UMAPLatent
from .utils import get_multivariable_prototype_classes

def sv_prototypes_visualization_filtered():
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

    sv_modules_wrapper = SingleVariableModulesWrapper(3, 4, 20, 4)
    sv_modules_wrapper.load_state_dict(torch.load("models/charactertrajectories/sv_modules_wrapper_bdpq.dat"))

    fig, axes = plt.subplots(1, 3, figsize=(24,4))
    colors = plt.cm.rainbow(np.linspace(0,1,5))

    b = Line2D([], [], color="white", marker='o', markerfacecolor=colors[0], markersize=10.0)
    d = Line2D([], [], color="white", marker='o', markerfacecolor=colors[1], markersize=10.0)
    p = Line2D([], [], color="white", marker='o', markerfacecolor=colors[2], markersize=10.0)
    q = Line2D([], [], color="white", marker='o', markerfacecolor=colors[3], markersize=10.0)
    prototype = Line2D([], [], color="white", marker='*', markerfacecolor=colors[4], markeredgecolor="black", markeredgewidth=0.5, markersize=10.0)
    leg = fig.legend(handles=[b, d, p, q, prototype],
               labels=["b", "d", "p", "q", "Prototype"], loc="lower center", ncol=5)
    leg.get_frame().set_edgecolor('k')

    visualize_moment = torch.utils.data.DataLoader(filtered_test, len(filtered_test), True)
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
    fig.align_ylabels(axes)
    plt.show()

def sv_prototypes_visualization():
    class_to_index = {
        "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
        "9":9, "10":10, "11":11, "12":12, "13":13, "14":14, "15":15,
        "16":16, "17":17, "18":18, "19":19, "20":20
    }
    letters = ["a", "b", "c", "d", "e", "g", "h", "l", "m", "n", "o", "p", "q", "r", "s", "u", "v", "w", "y", "z"]
    index_to_class = {}
    for i in range(1, 21):
        index_to_class[i] = str(i)
    train_ds, test_ds = get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TRAIN.ts", class_to_index), get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TEST.ts", class_to_index)

    sv_modules_wrapper = SingleVariableModulesWrapper(3, 20, 20, 20)
    sv_modules_wrapper.load_state_dict(torch.load("models/charactertrajectories/sv_modules_wrapper.dat"))

    fig, axes = plt.subplots(1, 3, figsize=(24,4))
    colors = plt.cm.rainbow(np.linspace(0,1,len(letters) + 1))

    lines = [Line2D([], [], color="white", marker='o', markerfacecolor=colors[i], markersize=10.0) for i in range(len(letters))]
    prototype = Line2D([], [], color="white", marker='*', markerfacecolor=colors[-1], markeredgecolor="black", markeredgewidth=0.5, markersize=10.0)
    leg = fig.legend(handles=lines + [prototype],
               labels=letters + ["Prototype"], loc="lower center", ncol=7)
    leg.get_frame().set_edgecolor('k')

    visualize_moment = torch.utils.data.DataLoader(test_ds, len(test_ds), True)
    for test_sample in visualize_moment:
        inp, out = test_sample[0].detach(), test_sample[1].detach()
        out = out - 1
        for i in range(3):
            embeddings = sv_modules_wrapper.single_variable_modules[i].encoder(inp[:,:,i].unsqueeze(2).float())
            embeddings = torch.concat([embeddings, sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix], dim=0)
            out = torch.concat([out, 20*torch.ones((sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix.shape[0],))], dim=0)
            visualizer = UMAPLatent()
            trans = visualizer.umap.fit(embeddings.detach().numpy())
            X = trans.embedding_
            x_min, x_max = np.min(X, 0), np.max(X, 0)
            X = (X - x_min) / (x_max - x_min)
            ax = axes[i]
            ax.set_xlabel("UMAP-1", fontsize=7.5)
            ax.set_ylabel("UMAP-2", fontsize=7.5)
            ax.set_title("Variable " + str(i+1), y=0.0, pad=-35, fontsize=10.0)
            for j in range(X.shape[0]):
                classif = out[j]
                if classif != 20:
                    ax.plot(X[j, 0], X[j, 1], 'o', color=colors[int(classif.item())], alpha=0.5)
                else:
                    ax.plot(X[j, 0], X[j, 1], '*', color=colors[int(classif.item())], alpha=1.0, markersize=10.0, markeredgecolor="black", markeredgewidth=0.5)
            ax.xaxis.set_tick_params(labelsize=7.5)
            ax.yaxis.set_tick_params(labelsize=7.5)
            ax.locator_params(axis="x", nbins=3)
            ax.locator_params(axis="y", nbins=3)
    fig.align_ylabels(axes)
    plt.show()

def heatmap_filtered():
    class_to_index = {
        "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
        "9":9, "10":10, "11":11, "12":12, "13":13, "14":14, "15":15,
        "16":16, "17":17, "18":18, "19":19, "20":20
    }
    index_to_class = {}
    for i in range(1, 21):
        index_to_class[i] = str(i)
    filtered_index_to_class = {0:"b", 1:"d", 2:"p", 3:"q"}
    train_ds, test_ds = get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TRAIN.ts", class_to_index), get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TEST.ts", class_to_index)
    filtered_train, filtered_test = filter_classes(train_ds, [2, 4, 12, 13]), filter_classes(test_ds, [2, 4, 12, 13])

    sv_modules_wrapper = SingleVariableModulesWrapper(3, 4, 20, 4)
    sv_modules_wrapper.load_state_dict(torch.load("models/charactertrajectories/sv_modules_wrapper_bdpq.dat"))

    model = MultivariableModule(sv_modules_wrapper.single_variable_modules, 3, 12, 4, 4)
    model.load_state_dict(torch.load("models/charactertrajectories/multivariable_module_bdpq.dat"))

    classes = get_multivariable_prototype_classes(model, filtered_train)
    class_names = [filtered_index_to_class[i] for i in classes]
    print(class_names)

    protos = model.aggregate_prototype_layer.protos
    min_val = protos.min()
    max_val = protos.max()
    scaled_protos = (protos - min_val) / (max_val - min_val)
    ax = sns.heatmap(scaled_protos.detach().numpy(), xticklabels=False, yticklabels=class_names)
    ax.set
    
    plt.show()

def heatmap():
    class_to_index = {
        "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
        "9":9, "10":10, "11":11, "12":12, "13":13, "14":14, "15":15,
        "16":16, "17":17, "18":18, "19":19, "20":20
    }
    index_to_class = {}
    for i in range(1, 21):
        index_to_class[i] = str(i)
    filtered_index_to_class = {0:"b", 1:"d", 2:"p", 3:"q"}
    train_ds, test_ds = get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TRAIN.ts", class_to_index), get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TEST.ts", class_to_index)
    filtered_train, filtered_test = filter_classes(train_ds, [2, 4, 12, 13]), filter_classes(test_ds, [2, 4, 12, 13])

    sv_modules_wrapper = SingleVariableModulesWrapper(3, 20, 20, 20)
    sv_modules_wrapper.load_state_dict(torch.load("models/charactertrajectories/sv_modules_wrapper.dat"))

    model = MultivariableModule(sv_modules_wrapper.single_variable_modules, 3, 60, 20, 20)
    model.load_state_dict(torch.load("models/charactertrajectories/multivariable_module.dat"))

    # classes = get_multivariable_prototype_classes(model, train_ds)
    # class_names = [index_to_class[i] for i in classes]
    # print(class_names)

    protos = model.aggregate_prototype_layer.protos
    min_val = protos.min()
    max_val = protos.max()
    scaled_protos = (protos - min_val) / (max_val - min_val)
    ax = sns.heatmap(scaled_protos.detach().numpy(), xticklabels=False, yticklabels=False)
    ax.set
    
    plt.show()

def multivariable_prototypes_filtered():
    class_to_index = {
        "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
        "9":9, "10":10, "11":11, "12":12, "13":13, "14":14, "15":15,
        "16":16, "17":17, "18":18, "19":19, "20":20
    }
    index_to_class = {}
    for i in range(1, 21):
        index_to_class[i] = str(i)
    filtered_index_to_class = {0:"b", 1:"d", 2:"p", 3:"q"}
    train_ds, test_ds = get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TRAIN.ts", class_to_index), get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TEST.ts", class_to_index)
    filtered_train, filtered_test = filter_classes(train_ds, [2, 4, 12, 13]), filter_classes(test_ds, [2, 4, 12, 13])

    sv_modules_wrapper = SingleVariableModulesWrapper(3, 4, 20, 4)
    sv_modules_wrapper.load_state_dict(torch.load("models/charactertrajectories/sv_modules_wrapper_bdpq.dat"))

    model = MultivariableModule(sv_modules_wrapper.single_variable_modules, 3, 12, 4, 4)
    model.load_state_dict(torch.load("models/charactertrajectories/multivariable_module_bdpq.dat"))

    protos = model.aggregate_prototype_layer.protos
    min_val = protos.min()
    max_val = protos.max()
    scaled_protos = (protos - min_val) / (max_val - min_val)

    classes = get_multivariable_prototype_classes(model, filtered_train)
    class_names = [filtered_index_to_class[i] for i in classes]
    print(class_names)

    cmap = plt.cm.get_cmap("Dark2")
    colors = cmap(np.linspace(0.5, 1, model.num_variables))

    data_train = torch.utils.data.DataLoader(filtered_train, len(filtered_train), True)
    data_test = torch.utils.data.DataLoader(filtered_test, len(filtered_test), True)
    with torch.no_grad():
        for train_sample in data_train:
            inp, out = train_sample[0].detach(), train_sample[1].detach()

            fig, axes = plt.subplots(model.num_prototypes, model.num_variables)

            vmin = np.min(scaled_protos.detach().numpy())
            vmax = np.max(scaled_protos.detach().numpy())
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
                    ax.plot(closest_point.detach().numpy(), linewidth=0.75, c=colors[j])
                    plt.subplots_adjust(hspace=0.75)
                    ax.tick_params(length=0)
                    ax.xaxis.set_tick_params(labelsize=6)
                    ax.yaxis.set_tick_params(labelsize=6)

                    # ax = axes[2*i + 1][j]

                    # sns.heatmap(np.expand_dims(block, axis=0),
                    #             ax=ax, cbar=False, vmin=vmin, vmax=vmax, square=True,
                    #             xticklabels=False, yticklabels=False)

                    if j == 0:
                        ax.set_ylabel(class_names[i], labelpad=25.0, rotation="horizontal", fontsize=10)


            for j in range(model.num_variables):
                for i in range(model.num_prototypes):
                    ax = axes[i][j]
                    ax.set_ylim(ylims[j][0], ylims[j][1])

            fig.align_ylabels()
            plt.show()

def split_heatmaps():
    class_to_index={"epilepsy":0, "walking":1, "running":2,"sawing":3}
    index_to_class = {0:"pilepsy", 1:"walking", 2:"running", 3:"sawing"}
    train_ds, test_ds = get_ds("data/epilepsy/Epilepsy_TRAIN.ts", class_to_index), get_ds("data/epilepsy/Epilepsy_TEST.ts", class_to_index)

    sv_modules_wrapper = SingleVariableModulesWrapper(3, 4, 40, 4)
    sv_modules_wrapper.load_state_dict(torch.load("models/epilepsy/sv_modules_wrapper.dat"))
    model = MultivariableModule(sv_modules_wrapper.single_variable_modules, 3, 12, 4, 4)
    model.load_state_dict(torch.load("models/epilepsy/multivariable_module.dat"))

    protos = model.aggregate_prototype_layer.protos
    min_val = protos.min()
    max_val = protos.max()
    scaled_protos = (protos - min_val) / (max_val - min_val)
    vmin = np.min(scaled_protos.detach().numpy())
    vmax = np.max(scaled_protos.detach().numpy())

    fig, axes = plt.subplots(model.num_prototypes, model.num_variables)

    for i in range(model.num_prototypes):
        mv_prototype = scaled_protos[i].detach().numpy()
        single_variable_blocks = np.split(mv_prototype, model.num_variables)
        for j in range(len(single_variable_blocks)):
            block = single_variable_blocks[j]
            ax = axes[i][j]
            s = sns.heatmap(np.expand_dims(block, axis=0), ax=ax, square=True, vmin=vmin, vmax=vmax,
                        xticklabels=False, yticklabels=False, cbar=False)
            s.set_xlabel("Variable " + str(j+1), fontsize=10)
            

    plt.show()




if __name__ == "__main__":
    sv_prototypes_visualization_filtered()