import argparse

import numpy as np
from scipy.spatial.distance import pdist, squareform

from src.data.data import get_ds
from src.utils.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def intra_cluster_distance(wrapper, dl, dataset):
    class_to_pattern = get_class_to_pattern_map().to(device)
    wrapper.eval()
    with torch.no_grad():
        for data_matrix, labels in dl:
            data_matrix, labels = data_matrix.to(device), labels.to(device)
            if dataset.startswith("simulated"):
                for var in range(wrapper.num_variables - 1):
                    patterns = class_to_pattern[labels, var].to(device)
                    encoder = wrapper.single_variable_prototype_modules[var].encoder   
                    single_variable_data = data_matrix[:, :, var].unsqueeze(2).float()
                    embeddings = encoder(single_variable_data)

                    unique_patterns = torch.unique(patterns)
                    pattern_to_distance = {}

                    for pattern in unique_patterns:
                        mask = patterns == pattern
                        cluster_points = embeddings[mask]
                        expanded_cluster_points_1 = cluster_points.unsqueeze(1)
                        expanded_cluster_points_2 = cluster_points.unsqueeze(0)
                        
                        distances = torch.norm(expanded_cluster_points_1 - expanded_cluster_points_2, dim=2)

                        avg_distance = torch.mean(distances.tril(diagonal=-1))

                        pattern_to_distance[pattern.item()] = avg_distance.item()
                    print("variable: " + str(var))
                    print(pattern_to_distance)
            else:
                for var in range(wrapper.num_variables):
                    encoder = wrapper.single_variable_prototype_modules[var].encoder   
                    single_variable_data = data_matrix[:, :, var].unsqueeze(2).float()
                    embeddings = encoder(single_variable_data)

                    unique_labels = torch.unique(labels)
                    labels_to_distance = {}

                    for label in unique_labels:
                        mask = labels == label
                        cluster_points = embeddings[mask]
                        expanded_cluster_points_1 = cluster_points.unsqueeze(1)
                        expanded_cluster_points_2 = cluster_points.unsqueeze(0)
                        
                        distances = torch.norm(expanded_cluster_points_1 - expanded_cluster_points_2, dim=2)

                        avg_distance = torch.mean(distances.tril(diagonal=-1))

                        labels_to_distance[label.item()] = avg_distance.item()
                    print("variable: " + str(var))
                    print(labels_to_distance)

def inter_cluster_distance(wrapper, dl, dataset):
    class_to_pattern = get_class_to_pattern_map().to(device)
    wrapper.eval()
    with torch.no_grad():
        for data_matrix, labels in dl:
            data_matrix, labels = data_matrix.to(device), labels.to(device)
            if dataset.startswith("simulated"):
                for var in range(wrapper.num_variables - 1):
                    patterns = class_to_pattern[labels, var].to(device)
                    encoder = wrapper.single_variable_prototype_modules[var].encoder   
                    single_variable_data = data_matrix[:, :, var].unsqueeze(2).float()
                    embeddings = encoder(single_variable_data)

                    unique_patterns = torch.unique(patterns)
                    centroids = []
                    for p in unique_patterns:
                        mask = (patterns == p)
                        cluster_points = embeddings[mask]
                        centroid = torch.mean(cluster_points, dim=0)
                        centroids.append(centroid)
                    
                    centroids = torch.stack(centroids)
                    distance_matrix = torch.cdist(centroids, centroids, p=2)
                    print(distance_matrix)
            else:
                for var in range(wrapper.num_variables):
                    encoder = wrapper.single_variable_prototype_modules[var].encoder   
                    single_variable_data = data_matrix[:, :, var].unsqueeze(2).float()
                    embeddings = encoder(single_variable_data)

                    unique_labels = torch.unique(labels)
                    centroids = []
                    for l in unique_labels:
                        mask = (labels == l)
                        cluster_points = embeddings[mask]
                        centroid = torch.mean(cluster_points, dim=0)
                        centroids.append(centroid)
                    
                    centroids = torch.stack(centroids)
                    distance_matrix = torch.cdist(centroids, centroids, p=2)
                    print(distance_matrix)

def main(args):
    dataset = args.dataset
    config = get_config_from_dataset(dataset)
    train_ds = get_ds(get_train_path_from_dataset(dataset), config['class_to_index'])
    test_ds = get_ds(get_test_path_from_dataset(dataset), config['class_to_index'])
    train_dataloader = torch.utils.data.DataLoader(train_ds, len(train_ds), shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_ds, len(test_ds), shuffle=False)

    wrapper = load_single_variable_prototypes_wrapper(config).to(device)

    if args.type == "intra":
        intra_cluster_distance(wrapper, test_dataloader, dataset)
    elif args.type == "inter":
        inter_cluster_distance(wrapper, test_dataloader, dataset)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g. <basicmotions>)")
    parser.add_argument("--type", type=str, help="What function do you want to call")

    args = parser.parse_args()
    main(args)