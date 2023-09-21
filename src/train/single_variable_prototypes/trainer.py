import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import umap

class SingleVariablePrototypesTrainer(torch.nn.Module):
    """
    Trains single-variable prototypes modules
    """
    def __init__(self, wrapper, train_ds, test_ds, classes, num_variables, num_prototypes, num_layers, batch_size, lr, gamma, epochs, l1, l2, l3, l4):
        super(SingleVariablePrototypesTrainer, self).__init__()
        self.wrapper = wrapper
        self.train_ds = train_ds
        self.test_ds = test_ds
        self.classes = classes
        self.num_variables = num_variables
        self.num_prototypes = num_prototypes
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.epochs = epochs
        self.l1 = l1
        self.l2 = l2
        self.l3 = l3
        self.l4 = l4

        self.train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_ds, len(test_ds), shuffle=False)

        # Disable gradients for the encoders
        for module in self.wrapper.single_variable_prototype_modules:
            for param in module.encoder.parameters():
                param.requires_grad = False

        self.opt = torch.optim.Adam(filter(lambda x: x.requires_grad, self.parameters()), lr=self.lr)
        self.classification_loss_fn = torch.nn.CrossEntropyLoss()
        self.sched = torch.optim.lr_scheduler.ExponentialLR(self.opt, self.gamma)

        self.classification_losses = []
        self.prototype_diversity_penalties = []
        self.prototype_similarity_penalties = []
        self.encoded_space_coverage_penalties = []
        self.total_losses = []

    def initialize_prototypes(self):
        """
        Initializes prototypes according to k-means++.
        https://en.wikipedia.org/wiki/K-means%2B%2B
        """
        with torch.no_grad():
            # Iterate through the variables
            for data_matrix, _ in self.train_dataloader:
                for i in range(self.num_variables):
                    single_variable_data = data_matrix[:, :, i].unsqueeze(2).float()
                    encoder = self.wrapper.single_variable_prototype_modules[i].encoder
                    prototypes = self.wrapper.single_variable_prototype_modules[i].prototypes
                    encodings = encoder(single_variable_data)
                    # Step 1: Set a random element to be the first prototype
                    prototypes[0] = random.choice(encodings)
                    for j in range(1, self.num_prototypes):
                        # Step 2: For each data point calculate distance to each chosen prototype
                        # Keep distance to closest chosen prototype
                        distances = []
                        for point in encodings:
                            min_distance = float("inf")
                            for k in range(j):
                                prototype = prototypes[k]
                                min_distance = min(min_distance, torch.linalg.vector_norm(point - prototype))
                            distances.append(float(min_distance))
                        probabilities = np.array(distances)
                        probabilities = np.square(probabilities)
                        probabilities = probabilities / probabilities.sum()

                        # Step 3: Choose an element at random to be the next prototype with prob proportional to distance
                        found = False
                        while not found:
                            index = np.random.choice([ind for ind in range(len(encodings))], p=probabilities)
                            candidate = encodings[index]
                            if candidate not in prototypes:
                                found = True
                        prototypes[j] = candidate

    def prototype_diversity_penalty(self):
        """
        Penalizes prototypes for being close together.
        """
        total_penalty = 0
        for i in range(self.num_variables):
            prototypes = self.wrapper.single_variable_prototype_modules[i].prototypes
            num_prototypes = prototypes.size(0)
            penalty = 0.0

            for j in range(num_prototypes):
                for k in range(j + 1, num_prototypes):
                    distance = torch.norm(prototypes[j] - prototypes[k])
                    term = torch.pow(torch.max(torch.tensor(0.0), 1.0 - distance), 2)
                    penalty += term

            total_penalty += penalty
        return total_penalty
    
    def prototype_similarity_penalty(self, data):
        """
        Penalizes prototypes for not being similar to a training point.
        """
        total_penalty = 0
        for i in range(self.num_variables):
            single_variable_data = data[:, :, i].unsqueeze(2).float()

            embeddings = self.wrapper.single_variable_prototype_modules[i].encoder(single_variable_data)
            embeddings_exp = embeddings.unsqueeze(1)
            prototypes_exp = self.wrapper.single_variable_prototype_modules[i].prototypes.unsqueeze(0)

            distances = torch.norm(embeddings_exp - prototypes_exp, dim=2)
            min_distances = torch.min(distances, dim=0).values
            sim = torch.sum(min_distances)
            total_penalty += sim
        return total_penalty
    
    def encoded_space_coverage_penalty(self, data):
        """
        Penalizes prototypes for leaving regions of the encoded space uncovered.
        """
        total_penalty = 0
        for i in range(self.num_variables):
            single_variable_data = data[:, :, i].unsqueeze(2).float()

            embeddings = self.wrapper.single_variable_prototype_modules[i].encoder(single_variable_data)
            prototypes = self.wrapper.single_variable_prototype_modules[i].prototypes
            embeddings_expanded = embeddings.unsqueeze(1)
            prototypes_expanded = prototypes.unsqueeze(0) 

            pairwise_distances = torch.norm(embeddings_expanded - prototypes_expanded, dim=2)
            closest_distances = torch.min(pairwise_distances, dim=1)[0]
            total_distance = torch.sum(closest_distances)
            total_penalty += total_distance
        return total_penalty
    
    def train(self):
        for epoch in tqdm(range(self.epochs)):
            for data_matrix, labels in self.train_dataloader:
                _, classification_output = self.wrapper(data_matrix.float())

                classification_loss = self.classification_loss_fn(classification_output, labels)
                self.classification_losses.append(float(classification_loss))
                
                diversity_penalty = self.prototype_diversity_penalty()
                self.prototype_diversity_penalties.append(float(diversity_penalty))

                similarity_penalty = self.prototype_similarity_penalty(data_matrix)
                self.prototype_similarity_penalties.append(float(similarity_penalty))

                coverage_penalty = self.encoded_space_coverage_penalty(data_matrix)
                self.encoded_space_coverage_penalties.append(float(coverage_penalty))

                total_loss = (self.l1)*classification_loss + \
                             (self.l2)*diversity_penalty + \
                             (self.l3)*similarity_penalty + \
                             (self.l4)*coverage_penalty
                self.total_losses.append(float(total_loss))


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

    def plot_diversity_penalties(self):
        plt.figure()
        plt.plot(self.prototype_diversity_penalties, label="Diversity Penalty")
        plt.title("Diversity Penalty per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Penalty")
        plt.legend()
        plt.show()

    def plot_similarity_penalties(self):
        plt.figure()
        plt.plot(self.prototype_similarity_penalties, label="Similarity Penalty")
        plt.title("Similarity Penalty per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Penalty")
        plt.legend()
        plt.show()

    def plot_coverage_penalties(self):
        plt.figure()
        plt.plot(self.encoded_space_coverage_penalties, label="Coverage Penalty")
        plt.title("Coverage Penalty per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Penalty")
        plt.legend()
        plt.show()

    def plot_total_loss(self):
        plt.figure()
        plt.plot(self.total_losses, label="Total Loss")
        plt.title("Total Loss per Epoch")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()
        plt.show()

    def plot_all_latent_spaces_with_prototypes(self, use_test=True):
        ds = self.train_dataloader
        if use_test:
            ds = self.test_dataloader
        for data_matrix, labels in ds:
            for variable in range(self.num_variables):
                plt.figure()
                single_variable_data = data_matrix[:, :, variable].unsqueeze(2).float()
                with torch.no_grad():
                    embeddings = self.wrapper.single_variable_prototype_modules[variable].encoder(single_variable_data)
                    embeddings = torch.concat([embeddings, self.wrapper.single_variable_prototype_modules[variable].prototypes], dim=0)
                    out = torch.concat([labels, len(self.classes)*torch.ones((self.wrapper.single_variable_prototype_modules[variable].prototypes.shape[0],))], dim=0)
                    reducer = umap.UMAP()
                    embeddings_2d = reducer.fit_transform(embeddings)

                    unique_labels = self.classes + [len(self.classes)]
                    string_labels = np.array([unique_labels[int(label)] for label in out])
                    handles, lbls = [], []
                    for label in self.classes:
                        idx = np.where(string_labels == label)[0]
                        scatter = plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label)
                        handles.append(scatter)
                        lbls.append(label)

                    idx = np.where(string_labels == len(self.classes))[0]
                    scatter = plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], marker="*", edgecolor='black', label="Prototype")
                    handles.append(scatter)
                    lbls.append("Prototype")
                    plt.legend(handles, lbls)
                    plt.title("Latent Space for Variable " + str(variable + 1))
                    plt.show()
    
    def evaluate(self, use_test=True):
        with torch.no_grad():
            numerator = 0
            denominator = 0
            for data_matrix, label in self.test_dataloader:
                _, classification_output = self.wrapper(data_matrix.float())
                sof = torch.softmax(classification_output, 1)
                prediction = torch.argmax(sof, 1)

                numerator += torch.sum(prediction.eq(label).int())
                denominator += data_matrix.shape[0]
            accuracy = float(numerator) / float(denominator)
            print("Accuracy: " + str(accuracy))

    def save(self, save_dir):
        save_name = save_dir + "single_variable_prototypes.pth"
        torch.save(self.wrapper.state_dict(), save_name)

    def load(self, save_dir):
        save_name = save_dir + "single_variable_prototypes.pth"
        self.wrapper.load_state_dict(torch.load(save_name))

    