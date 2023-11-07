import argparse

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import numpy as np

from src.data.data import get_ds
from src.models.encoding import Encoder
from src.utils.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    config = get_config_from_dataset(args.dataset)
    encoders = load_encoders(config)

    train_ds=get_ds(get_train_path_from_dataset(args.dataset), config['class_to_index'])
    train_dl = torch.utils.data.DataLoader(train_ds, len(train_ds), shuffle=True)

    with torch.no_grad():
        for data_matrix, labels in train_dl:
            data_matrix, labels = data_matrix.to(device), labels.to(device)
            k_values = range(2, 11)
            all_silhouette_scores = np.zeros((len(k_values), config['num_variables']))
            for var in range(config['num_variables']):
                single_variable_data = data_matrix[:, :, var].unsqueeze(2).float()
                encoder = encoders[var].to(device)
                embeddings = encoder(single_variable_data)
                embeddings = embeddings.cpu().detach().numpy()

                silhouette_scores = []

                for k in k_values:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    labels = kmeans.fit_predict(embeddings)
                    score = silhouette_score(embeddings, labels)
                    silhouette_scores.append(score)
                all_silhouette_scores[:, var] = silhouette_scores
                plt.plot(k_values, all_silhouette_scores[:, var], marker='o', label=f"Variable {var+1}")

    average_silhouette_scores = np.mean(all_silhouette_scores, axis=1)

    # Find the k with the highest average silhouette score
    best_k = k_values[np.argmax(average_silhouette_scores)]

    # Plotting
    plt.title('Silhouette Score vs. Number of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Score')
    plt.xticks(k_values)
    plt.grid(True)
    plt.legend()
    plt.show()

    # Print the best k
    print(f"The k with the highest average silhouette score is: {best_k}")  

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g. <basicmotions>)")
    args = parser.parse_args()
    main(args)