import numpy as np
from scipy.spatial.distance import pdist, squareform

from src.data.data import get_ds
from src.utils.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

dataset = "simulated_6400"

config = get_config_from_dataset(dataset)
train_ds = get_ds(get_train_path_from_dataset(dataset), config['class_to_index'])
test_ds = get_ds(get_test_path_from_dataset(dataset), config['class_to_index'])
train_dataloader = torch.utils.data.DataLoader(train_ds, len(train_ds), shuffle=True)
test_dataloader = torch.utils.data.DataLoader(test_ds, len(test_ds), shuffle=False)

wrapper = load_single_variable_prototypes_wrapper(config).to(device)

def intra_cluster_distance(wrapper):
    class_to_pattern = get_class_to_pattern_map()
    wrapper.eval()
    with torch.no_grad():
        for data_matrix, labels in test_dataloader:
            data_matrix, labels = data_matrix.to(device), labels.to(device)
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

intra_cluster_distance(wrapper)