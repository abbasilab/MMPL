import matplotlib.pyplot as plt
import numpy as np
import umap

from src.comparisons.ming.model import AutoencoderPrototypeModel
from src.data.data import get_ds
from src.utils.utils import *

config = get_comparison_config_from_dataset("ming", "simulated_640")
one_stage_config = config['one_stage']

model = AutoencoderPrototypeModel(
    input_dim=one_stage_config['input_dim'],
    hidden_dim=one_stage_config['hidden_dim'],
    latent_dim=one_stage_config['latent_dim'],
    num_prototypes=one_stage_config['num_prototypes'],
    seq_len=one_stage_config['seq_len'],
    num_classes=one_stage_config['num_classes'],
    num_layers=one_stage_config['num_layers']
)

model.load_state_dict(torch.load("models/comparisons/ming/simulated_640/model.pth"))

train_ds=get_ds(get_train_path_from_dataset("simulated_640"), config['class_to_index'])
test_ds=get_ds(get_test_path_from_dataset("simulated_640"), config['class_to_index'])

train_dataloader = torch.utils.data.DataLoader(train_ds, len(train_ds), shuffle=False)
test_dataloader = torch.utils.data.DataLoader(test_ds, len(test_ds), shuffle=False)

def plot_latent_space():
    for data_matrix, labels in train_dataloader:
        data_matrix, labels = data_matrix.to(device), labels.to(device)
        plt.figure()
        with torch.no_grad():
            _, _, embeddings = model(data_matrix.float())
            reducer = umap.UMAP()
            embeddings_2d = reducer.fit_transform(embeddings.cpu())

            string_labels = np.array([config['classes'][label.item()] for label in labels])
            handles, lbls = [], []
            for label in config['classes']:
                idx = np.where(string_labels == label)[0]
                scatter = plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label)
                handles.append(scatter)
                lbls.append(label)
                
            plt.title("Latent Space")
            plt.show()

if __name__ == "__main__":
    plot_latent_space()