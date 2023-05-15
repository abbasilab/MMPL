import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns

from ..data.data import get_ds
from ..models.single_variables import SingleVariableModulesWrapper, DecodingModule
from ..models.multivariable import MultivariableModule

def generate_prototypes_heatmap(model):
    plt.figure()
    sns.heatmap(torch.relu(model.aggregate_prototype_layer.protos).detach().numpy())
    plt.show()

def generate_decoded_prototypes_heatmap_basicmotions():
    # decoding_module = DecodingModule(10, 100, 6)
    # decoding_module.load_state_dict(torch.load("models/basicmotions/dec.dat"))
    sv_modules_wrapper = SingleVariableModulesWrapper(6, 4, 10, 4)
    sv_modules_wrapper.load_state_dict(torch.load("models/basicmotions/sv_modules_wrapper.dat"))
    model = MultivariableModule(sv_modules_wrapper.single_variable_modules, 6, 24, 4, 4)
    model.load_state_dict(torch.load("models/basicmotions/multivariable_module.dat"))
    sns.heatmap(torch.relu(model.aggregate_prototype_layer.protos).detach().numpy())
    plt.show()
    exit()
    class_to_index={"standing":0, "running":1, "walking":2,"badminton":3}
    index_to_class={0:"Standing", 1:"Running", 2:"Walking", 3:"Badminton"}
    train_ds, test_ds = get_ds("data/basicmotions/BasicMotions_TRAIN.ts", class_to_index), get_ds("data/basicmotions/BasicMotions_TEST.ts", class_to_index)

    m = {0: [], 1: [], 2: [], 3: []}

    data_test = torch.utils.data.DataLoader(test_ds, len(test_ds), True)
    with torch.no_grad():
        for data_matrix, labels in data_test:
            _, concat_features = model(data_matrix.float())
            for i in range(len(concat_features)):
                point = concat_features[i]
                min_dist = float("inf")
                index = -1
                for j in range(len(model.aggregate_prototype_layer.protos)):
                    proto = model.aggregate_prototype_layer.protos[j]
                    dist = np.linalg.norm(point - proto)
                    if dist < min_dist:
                        min_dist = dist
                        index = j
                m[index].append(labels[i].item())

    classes = [max(set(arr)) for arr in m.values()]
    class_names = [index_to_class[i] for i in classes]
    print(class_names)
    with torch.no_grad():
        for test_sample in data_test:
            inp, out = test_sample[0].detach(), test_sample[1].detach()
            gs = GridSpec(nrows=8, ncols=7)
            fig = plt.figure()
            vmin = np.min(model.aggregate_prototype_layer.protos.detach().numpy())
            vmax = np.max(model.aggregate_prototype_layer.protos.detach().numpy())
            ylims = [[float("inf"), -float("inf")] for _ in range(model.num_variables)]
            for i in range(model.num_prototypes):
                mv_prototype_tensor = model.aggregate_prototype_layer.protos[i]
                mv_prototype = mv_prototype_tensor.detach().numpy()
                single_variable_blocks = np.split(mv_prototype, 6)
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
                    ax = fig.add_subplot(gs[2*i, j])
                    ax.plot(closest_point.detach().numpy())
                    ax.tick_params(length=0)
                    ax.xaxis.set_tick_params(labelsize=6)
                    ax.yaxis.set_tick_params(labelsize=6)

                    ax = fig.add_subplot(gs[2*i + 1, j])
                    s = sns.heatmap(np.expand_dims(block, axis=0),
                                ax=ax, cbar=False, vmin=vmin, vmax=vmax, xticklabels=False, yticklabels=False, square=True)
                    s.set_xlabel("Variable " + str(j+1), fontsize=7.5)

            fig.subplots_adjust(wspace=0.4, hspace=0)
            for j in range(model.num_variables):
                for i in range(model.num_prototypes):
                    ax = fig.get_axes()[model.num_variables*2*i + 2*j]
                    ax.set_ylim(ylims[j][0], ylims[j][1])
                    

            fig.align_ylabels()
            plt.show()

def generate_decoded_prototypes_heatmap_epilepsy():
    sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=3, num_classes=4, hidden=40, num_prototypes=5)
    sv_modules_wrapper.load_state_dict(torch.load("models/epilepsy/sv_modules_wrapper_5_40.dat"))

    model = MultivariableModule(single_variable_modules=sv_modules_wrapper.single_variable_modules, \
                                 num_variables=3, hidden=15, num_classes=4, num_prototypes=4)
    model.load_state_dict(torch.load("models/epilepsy/multivariable_module_5_40.dat"))
    class_to_index={"epilepsy":0, "walking":1, "running":2,"sawing":3}
    index_to_class={0:"Epilepsy", 1:"Walking", 2:"Running", 3:"Sawing"}
    train_ds, test_ds = get_ds("data/epilepsy/Epilepsy_TRAIN.ts", class_to_index), get_ds("data/epilepsy/Epilepsy_TEST.ts", class_to_index)
    m = {0: [], 1: [], 2: [], 3: []}

    data_train = torch.utils.data.DataLoader(train_ds, len(train_ds), True)
    data_test = torch.utils.data.DataLoader(test_ds, len(test_ds), True)
    with torch.no_grad():
        for data_matrix, labels in data_test:
            _, concat_features = model(data_matrix.float())
            for i in range(len(concat_features)):
                point = concat_features[i]
                min_dist = float("inf")
                index = -1
                for j in range(len(model.aggregate_prototype_layer.protos)):
                    proto = model.aggregate_prototype_layer.protos[j]
                    dist = np.linalg.norm(point - proto)
                    if dist < min_dist:
                        min_dist = dist
                        index = j
                m[index].append(labels[i].item())

    classes = [max(set(arr)) for arr in m.values()]
    class_names = [index_to_class[i] for i in classes]

    print(class_names)

    with torch.no_grad():
        for test_sample in data_test:
            inp, out = test_sample[0].detach(), test_sample[1].detach()
            gs = GridSpec(nrows=8, ncols=4)
            fig = plt.figure()
            vmin = np.min(model.aggregate_prototype_layer.protos.detach().numpy())
            vmax = np.max(model.aggregate_prototype_layer.protos.detach().numpy())
            ylims = [[float("inf"), -float("inf")] for _ in range(model.num_variables)]
            for i in range(model.num_prototypes):
                mv_prototype_tensor = model.aggregate_prototype_layer.protos[i]
                mv_prototype = mv_prototype_tensor.detach().numpy()
                single_variable_blocks = np.split(mv_prototype, 3)
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
                    ax = fig.add_subplot(gs[2*i, j])
                    ax.plot(closest_point.detach().numpy())
                    ax.tick_params(length=0)
                    ax.xaxis.set_tick_params(labelsize=6)
                    ax.yaxis.set_tick_params(labelsize=6)

                    ax = fig.add_subplot(gs[2*i + 1, j])
                    s = sns.heatmap(np.expand_dims(block, axis=0),
                                ax=ax, cbar=False, vmin=vmin, vmax=vmax, xticklabels=False, yticklabels=False, square=True)
                    s.set_xlabel("Variable " + str(j+1), fontsize=7.5)

            fig.subplots_adjust(wspace=0.4, hspace=0)
            for j in range(model.num_variables):
                for i in range(model.num_prototypes):
                    ax = fig.get_axes()[model.num_variables*2*i + 2*j]
                    ax.set_ylim(ylims[j][0], ylims[j][1])
                    

            fig.align_ylabels()
            plt.show()


if __name__ == "__main__":
    generate_decoded_prototypes_heatmap_basicmotions()
    # sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=3, num_classes=4, hidden=40, num_prototypes=5)
    # sv_modules_wrapper.load_state_dict(torch.load("models/epilepsy/sv_modules_wrapper_5_40.dat"))

    # model = MultivariableModule(single_variable_modules=sv_modules_wrapper.single_variable_modules, \
    #                              num_variables=3, hidden=15, num_classes=4, num_prototypes=5)
    # model.load_state_dict(torch.load("models/epilepsy/multivariable_module_5_40.dat"))
    # generate_prototypes_heatmap(model)
