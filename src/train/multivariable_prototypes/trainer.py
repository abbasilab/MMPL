import random

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class MultivariableModuleTrainer(torch.nn.Module):
    """
    Trains multivariable module.
    """
    def __init__(self, multivariable_prototypes, train_ds, test_ds, classes, num_variables, num_prototypes, num_layers, batch_size, lr, gamma, epochs, l1, l2, l3, l4, d_min):
        super(MultivariableModuleTrainer, self).__init__()
        self.multivariable_prototypes = multivariable_prototypes
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
        self.d_min = d_min

        self.train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        self.full_train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=len(train_ds), shuffle=False, pin_memory=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_ds, len(test_ds), shuffle=False, pin_memory=True)

        # Disable gradients for single variable portions
        for param in self.multivariable_prototypes.wrapper.parameters():
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
            for data_matrix, labels in self.full_train_dataloader:
                data_matrix = data_matrix.to(device)
                wrapper = self.multivariable_prototypes.wrapper
                prototypes = self.multivariable_prototypes.prototypes
                sims, _ = wrapper(data_matrix)
                chosen_indices = []
                # Step 1: Set a random element to be the first prototype
                index = random.randint(0, len(sims) - 1)
                prototypes[0] = sims[index]
                chosen_indices.append(index)
                for j in range(1, self.num_prototypes):
                    print(j)
                    # Step 2: For each data point calculate distance to each chosen prototype
                    # Keep distance to closest chosen prototype
                    distances = []
                    for point in sims:
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
                        index = np.random.choice([ind for ind in range(len(sims))], p=probabilities)
                        candidate = sims[index]
                        if index not in chosen_indices:
                            found = True
                            chosen_indices.append(index)
                    prototypes[j] = candidate
    
    def prototype_diversity_penalty(self):
        """
        Penalizes prototypes for being close together.
        """
        # total_penalty = 0
        # prototypes = self.multivariable_prototypes.prototypes
        # num_prototypes = prototypes.size(0)

        # for j in range(num_prototypes):
        #     for k in range(j + 1, num_prototypes):
        #         distance = torch.norm(prototypes[j] - prototypes[k])
        #         term = torch.pow(torch.max(torch.tensor(0.0), self.d_min - distance), 2)
        #         total_penalty += term

        # return total_penalty
        prototypes = self.multivariable_prototypes.prototypes
        num_prototypes = prototypes.size(0)

        prototypes_expanded = prototypes.unsqueeze(1).expand(-1, num_prototypes, -1)
        pairwise_distances = torch.norm(prototypes_expanded - prototypes_expanded.transpose(0, 1), dim=2)
        penalty = torch.pow(torch.max(torch.zeros_like(pairwise_distances), self.d_min - pairwise_distances), 2)
        mask = torch.triu(torch.ones_like(penalty), diagonal=1)
        penalty = penalty * mask
        total_penalty = torch.sum(penalty)
        return total_penalty
    
    def prototype_similarity_penalty(self, sims):
        """
        Penalizes prototypes for not being similar to a training point.
        """
        sims_exp = sims.unsqueeze(1)
        prototypes_exp = self.multivariable_prototypes.prototypes.unsqueeze(0)

        distances = torch.norm(sims_exp - prototypes_exp, dim=2)
        min_distances = torch.min(distances, dim=0).values
        return torch.sum(min_distances) / self.num_prototypes
    
    def encoded_space_coverage_penalty(self, sims):
        """
        Penalizes prototypes for leaving regions of the encoded space uncovered.
        """
        sims_exp = sims.unsqueeze(1)
        prototypes_exp = self.multivariable_prototypes.prototypes.unsqueeze(0) 

        pairwise_distances = torch.norm(sims_exp - prototypes_exp, dim=2)
        closest_distances = torch.min(pairwise_distances, dim=1)[0]
        total_penalty = torch.sum(closest_distances)
        return total_penalty / len(sims)
    
    def train(self):
        for epoch in tqdm(range(self.epochs)):
            for data_matrix, labels in self.train_dataloader:
                data_matrix, labels = data_matrix.to(device), labels.to(device)
                output, sv_sims = self.multivariable_prototypes(data_matrix.float())

                classification_loss = self.classification_loss_fn(output, labels)
                self.classification_losses.append(float(classification_loss))
                
                diversity_penalty = self.prototype_diversity_penalty()
                self.prototype_diversity_penalties.append(float(diversity_penalty))

                similarity_penalty = self.prototype_similarity_penalty(sv_sims)
                self.prototype_similarity_penalties.append(float(similarity_penalty))

                coverage_penalty = self.encoded_space_coverage_penalty(sv_sims)
                self.encoded_space_coverage_penalties.append(float(coverage_penalty))

                total_loss = (self.l1)*classification_loss + \
                             (self.l2)*diversity_penalty + \
                             (self.l3)*similarity_penalty + \
                             (self.l4)*coverage_penalty
                self.total_losses.append(float(total_loss))

                self.opt.zero_grad()
                total_loss.backward()
                self.opt.step()
                with torch.no_grad():
                    for param in self.multivariable_prototypes.linear.parameters():
                        param.clamp_(min=0)
            self.sched.step()

            # if epoch % 10 == 0:
            #     self.plot_prototypes_heatmap()

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

    def sorting_key(self, prototype):
        indices = []
        for i in range(3):
            single_variable_prototype = prototype[i*4:(i+1)*4]
            high_value_index = torch.argmax(single_variable_prototype).item()
            indices.append(high_value_index)
        return tuple(indices)
    def plot_prototypes_heatmap(self):
        with torch.no_grad():
            prototypes = self.multivariable_prototypes.prototypes[:, :-4]
            prototypes_list = [prototypes[i, :] for i in range(prototypes.shape[0])]
            sorted_prototypes = sorted(prototypes_list, key=self.sorting_key)
            sorted_prototypes_tensor = torch.stack(sorted_prototypes)

            plt.figure()
            sns.heatmap(sorted_prototypes_tensor.cpu().detach().numpy())
            plt.show()

    def evaluate(self, use_test=False):
        dl = self.train_dataloader
        if use_test:
            dl = self.test_dataloader
        with torch.no_grad():
            numerator = 0
            denominator = 0
            for data_matrix, labels in dl:
                data_matrix, labels = data_matrix.to(device), labels.to(device)
                output, _ = self.multivariable_prototypes(data_matrix.float())
                sof = torch.softmax(output, 1)
                prediction = torch.argmax(sof, 1)

                numerator += torch.sum(prediction.eq(labels).int())
                denominator += data_matrix.shape[0]
            accuracy = float(numerator) / float(denominator)
            print("Accuracy: " + str(accuracy))
        return accuracy

    def save(self, save_dir):
        save_name = save_dir + "multivariable_prototypes.pth"
        torch.save(self.multivariable_prototypes.state_dict(), save_name)

    def load(self, save_dir):
        save_name = save_dir + "multivariable_prototypes.pth"
        self.multivariable_prototypes.load_state_dict(torch.load(save_name))




