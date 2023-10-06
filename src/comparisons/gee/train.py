import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import umap

from src.data.data import get_ds
from src.comparisons.gee.model import AutoencoderPrototypeModel

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Trainer(torch.nn.Module):
    def __init__(self, autoencoder_prototype_model, train_ds, test_ds, classes, batch_size, lr, gamma, epochs, l1, l2, l3, l4, d_min):
        super(Trainer, self).__init__()
        self.model = autoencoder_prototype_model
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.classes = classes
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.epochs = epochs
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4
        self.d_min = d_min

        self.train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_ds, len(test_ds), shuffle=False)

        self.opt = torch.optim.Adam(self.parameters(), lr=self.lr)
        self.sched = torch.optim.lr_scheduler.ExponentialLR(self.opt, self.gamma)
        self.classification_loss_fn = torch.nn.CrossEntropyLoss()
        self.reconstruction_loss_fn = torch.nn.MSELoss()

        self.classification_losses = []
        self.reconstruction_losses = []
        self.prototype_diversity_penalties = []
        self.prototype_similarity_penalties = []
        self.encoded_space_coverage_penalties = []
        self.total_losses = []

    def prototype_diversity_penalty(self):
        """
        Penalizes prototypes for being close together.
        """
        total_penalty = 0
        prototypes = self.model.prototype_network.prototypes
        num_prototypes = prototypes.size(0)

        for j in range(num_prototypes):
            for k in range(j + 1, num_prototypes):
                distance = torch.norm(prototypes[j] - prototypes[k])
                term = torch.pow(torch.max(torch.tensor(0.0), self.d_min - distance), 2)
                total_penalty += term

        return total_penalty
    
    def prototype_similarity_penalty(self, embeddings):
        """
        Penalizes prototypes for not being similar to a training point.
        """
        prototypes = self.model.prototype_network.prototypes
        embeddings_exp = embeddings.unsqueeze(1)
        prototypes_exp = self.model.prototype_network.prototypes.unsqueeze(0)

        distances = torch.norm(embeddings_exp - prototypes_exp, dim=2)
        min_distances = torch.min(distances, dim=0).values
        return torch.sum(min_distances) / prototypes.size(0)
    
    def encoded_space_coverage_penalty(self, embeddings):
        """
        Penalizes prototypes for leaving regions of the encoded space uncovered.
        """
        embeddings_exp = embeddings.unsqueeze(1)
        prototypes_exp = self.model.prototype_network.prototypes.unsqueeze(0) 

        pairwise_distances = torch.norm(embeddings_exp - prototypes_exp, dim=2)
        closest_distances = torch.min(pairwise_distances, dim=1)[0]
        total_penalty = torch.sum(closest_distances)
        return total_penalty / len(embeddings_exp)

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            for data_matrix, labels in self.train_dataloader:
                data_matrix, labels = data_matrix.to(device), labels.to(device)
                predictions, reconstructions, embeddings = self.model(data_matrix.float())
                classification_loss = self.classification_loss_fn(predictions, labels)
                self.classification_losses.append(classification_loss.item())

                reconstruction_loss = self.reconstruction_loss_fn(reconstructions, data_matrix.float())
                self.reconstruction_losses.append(reconstruction_loss.item())

                diversity_penalty = self.prototype_diversity_penalty()
                self.prototype_diversity_penalties.append(diversity_penalty.item())

                similarity_penalty = self.prototype_similarity_penalty(embeddings)
                self.prototype_similarity_penalties.append(similarity_penalty.item())

                coverage_penalty = self.encoded_space_coverage_penalty(embeddings)
                self.encoded_space_coverage_penalties.append(coverage_penalty.item())

                total_loss = classification_loss + \
                             (self.l1)*reconstruction_loss + \
                             (self.l2)*diversity_penalty + \
                             (self.l3)*similarity_penalty + \
                             (self.l4)*coverage_penalty
                self.total_losses.append(total_loss.item())

                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()
            self.sched.step()

    def plot_classification_loss(self):
        plt.figure()
        plt.plot(self.classification_losses, label="Classification Loss")
        plt.title("Classification Loss per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_reconstruction_loss(self):
        plt.figure()
        plt.plot(self.reconstruction_losses, label="Reconstruction Loss")
        plt.title("Reconstruction Loss per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_diversity_penalties(self):
        plt.figure()
        plt.plot(self.prototype_diversity_penalties, label="Diversity Penalties")
        plt.title("Diversity Penalties per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_similarity_penalties(self):
        plt.figure()
        plt.plot(self.prototype_similarity_penalties, label="Similarity Penalties")
        plt.title("Similarity Penalties per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_coverage_penalties(self):
        plt.figure()
        plt.plot(self.encoded_space_coverage_penalties, label="Coverage Penalties")
        plt.title("Coverage Penalties per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def visualize_prototypes(self):
        with torch.no_grad():
            for data_matrix, labels in self.train_dataloader:
                data_matrix, labels = data_matrix.to(device), labels.to(device)
                plt.figure()
                with torch.no_grad():
                    _, _, embeddings = self.model(data_matrix.float())
                    prototypes = self.model.prototype_network.prototypes
                    embeddings = torch.concat([embeddings, prototypes], dim=0)

                    reducer = umap.UMAP()
                    embeddings_2d = reducer.fit_transform(embeddings.cpu())

                    string_labels = np.array([self.classes[label.item()] for label in labels])

                    handles, lbls = [], []
                    for label in self.classes:
                        idx = np.where(string_labels == label)[0]
                        scatter = plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label)
                        handles.append(scatter)
                        lbls.append(label)

                    labels = torch.concat([labels, len(self.classes)*torch.ones((prototypes.shape[0],)).to(device)], dim=0).cpu()
                    idx = np.where(labels == len(self.classes))[0]
                    scatter = plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label="Prototype", marker="*", edgecolor='black', s=75, c='magenta')
                    handles.append(scatter)
                    lbls.append("Prototype")
                        
                    plt.legend(handles, lbls)
                    plt.title("Latent Space")
                    plt.show()

    def plot_total_loss(self):
        plt.figure()
        plt.plot(self.total_losses, label="Total Loss")
        plt.title("Total Loss per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def eval(self, use_test=False):
        dl = self.train_dataloader
        if use_test:
            dl = self.test_dataloader
        with torch.no_grad():
            numerator = 0
            denominator = 0
            for data_matrix, labels in dl:
                data_matrix, labels = data_matrix.to(device), labels.to(device)
                output, _, _ = self.model(data_matrix.float())
                sof = torch.softmax(output, 1)
                prediction = torch.argmax(sof, 1)

                numerator += torch.sum(prediction.eq(labels).int())
                denominator += data_matrix.shape[0]
            accuracy = float(numerator) / float(denominator)
            print("Accuracy: " + str(accuracy))

if __name__ == "__main__":
    model = AutoencoderPrototypeModel(
        input_dim=3,
        hidden_dim=32,
        latent_dim=16,
        num_prototypes=5,
        # seq_len=100,
        seq_len=206,
        # seq_len=119,
        num_classes=4,
        num_layers=1
    ).to(device)
    model.float()

    # class_to_index={"standing":0, "running":1, "walking":2,"badminton":3}
    # train_ds, test_ds = get_ds("data/basicmotions/processed/train.ts", class_to_index), get_ds("data/basicmotions/processed/test.ts", class_to_index)
    class_to_index={"epilepsy":0, "walking":1, "running":2,"sawing":3}
    train_ds, test_ds = get_ds("data/epilepsy/processed/train.ts", class_to_index), get_ds("data/epilepsy/processed/test.ts", class_to_index)
    # class_to_index={"b":0, "d":1, "p":2,"q":3}
    # train_ds, test_ds = get_ds("data/charactertrajectories_filtered/processed/train.ts", class_to_index), get_ds("data/charactertrajectories_filtered/processed/test.ts", class_to_index)
    # class_to_index={}
    # for i in range(64):
    #     class_to_index[str(i)] = i
    # train_ds, test_ds = get_ds("data/simulated_6400/processed/train.ts", class_to_index), get_ds("data/simulated_6400/processed/val.ts", class_to_index)
    
    trainer = Trainer(
        model,
        train_ds,
        test_ds,
        # ["Standing", "Running", "Walking", "Badminton"],
        # batch_size=40,
        ["Epilepsy", "Walking", "Running", "Sawing"],
        batch_size=137,
        # ["b", "d", "p", "q"],
        # batch_size=127,
        # [str(i) for i in range(64)],
        # batch_size=640,
        lr=0.01,
        gamma=0.999,
        epochs=2000,
        l1=1.0,
        l2=100.0,
        l3=0.1,
        l4=0.1,
        d_min=2.0
    ).to(device)

    trainer.train()
    trainer.plot_classification_loss()
    trainer.plot_reconstruction_loss()
    trainer.plot_diversity_penalties()
    trainer.plot_similarity_penalties()
    trainer.plot_coverage_penalties()
    trainer.eval()
    trainer.eval(use_test=True)
    trainer.visualize_prototypes()

