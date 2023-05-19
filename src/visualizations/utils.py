import torch

def get_multivariable_prototype_classes(model, train_ds):
    protos = model.aggregate_prototype_layer.protos
    data_train = torch.utils.data.DataLoader(train_ds, len(train_ds), True)

    proto_class_map = {}
    for i in range(model.num_prototypes):
        proto_class_map[i] = []

    with torch.no_grad():
        for data_matrix, labels in data_train:
            _, concat_features = model(data_matrix.float())
            for i in range(len(concat_features)):
                point = concat_features[i]
                min_dist = float("inf")
                index = -1
                for j in range(len(protos)):
                    proto = protos[j]
                    dist = torch.norm(point - proto)
                    if dist < min_dist:
                        min_dist = dist
                        index = j
                proto_class_map[index].append(labels[i].item())
    classes = [max(set(arr), key=arr.count) for arr in proto_class_map.values()]
    return classes

            
