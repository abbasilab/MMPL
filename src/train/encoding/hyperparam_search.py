import gc

from sklearn.metrics import silhouette_score
import torch

from src.data.data import get_ds
from src.models.encoding import Encoder
from src.train.encoding.trainer import EncoderTrainer
from src.utils.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    config = get_config_from_dataset("simulated_6400")
    encoding_config = config['encoding']
    m_vals = [0.01, 0.1, 1.0, 10.0]
    hidden_dim_vals = [32, 64]
    latent_dim_vals = [16, 32]

    for i in range(3, len(m_vals)):
        for j in range(len(hidden_dim_vals)):
            torch.cuda.empty_cache()

            m = m_vals[i]
            hidden_dim = hidden_dim_vals[j]
            latent_dim = latent_dim_vals[j]             
            encoders = []
            for _ in range(4):
                encoders.append(Encoder(
                    input_dim=1,
                    hidden_dim=hidden_dim,
                    latent_dim=latent_dim
                ))
            trainer = EncoderTrainer(
                encoders=encoders,
                train_ds=get_ds(get_train_path_from_dataset("simulated_6400"), config['class_to_index']),
                test_ds=get_ds(get_test_path_from_dataset("simulated_6400"), config['class_to_index']),
                classes=config['classes'],
                num_variables=config['num_variables'],
                batch_size=encoding_config['batch_size'],
                lr=encoding_config['lr'],
                gamma=encoding_config['gamma'],
                epochs=500,
                m=m
            ).to(device)

            trainer.train()

            with torch.no_grad():
                with open("results.txt", "a") as f:
                    f.write(f"m: {m}, hidden: {hidden_dim}, latent: {latent_dim}\n")
                    for data_matrix, labels in trainer.test_dataloader:
                        data_matrix = data_matrix.to(device)
                        for var in range(3):
                            encoder = trainer.encoders[var]
                            single_variable_data = data_matrix[:, :, var].unsqueeze(2).float()
                            embeddings = encoder(single_variable_data)
                            embeddings = embeddings.cpu().detach().numpy()
                            patterns = get_single_variable_patterns_from_labels(labels, var)

                            score = silhouette_score(embeddings, patterns)
                            f.write(f"\tVariable: {var}, score: {score}\n")
                    f.close()


if __name__ == "__main__":
    main()