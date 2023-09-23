import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
import umap

from src.models.encoding import ContrastiveLoss

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class EncoderTrainer(torch.nn.Module):
    """
    Trains single-variable encoders.
    """
    def __init__(self, encoders, train_ds, test_ds, classes, num_variables, batch_size, lr, gamma, epochs, m):
        super(EncoderTrainer, self).__init__()
        self.encoders = torch.nn.ModuleList(encoders)
        self.train_ds = train_ds   
        self.test_ds = test_ds
        self.classes = classes
        self.num_variables = num_variables
        self.batch_size = batch_size
        self.lr = lr
        self.gamma = gamma
        self.epochs = epochs
        self.m = m

        self.train_dataloader = torch.utils.data.DataLoader(train_ds, batch_size=batch_size, shuffle=True, pin_memory=True)
        self.test_dataloader = torch.utils.data.DataLoader(test_ds, len(test_ds), shuffle=False, pin_memory=True)

        self.opts = [torch.optim.Adam(self.encoders[i].parameters(), lr=lr) for i in range(num_variables)]

        self.scheds = [torch.optim.lr_scheduler.ExponentialLR(self.opts[i], gamma) for i in range(num_variables)]
        
        self.contrastive_loss_fn = ContrastiveLoss(m=m)

        self.contrastive_losses = [[] for _ in range(num_variables)]

    def train(self):
        for encoder in self.encoders:
            encoder.train()
        for epoch in tqdm(range(self.epochs)):
            for i in range(self.num_variables):
                for data_matrix, labels in self.train_dataloader:
                    data_matrix, labels = data_matrix.to(device), labels.to(device)
                    single_variable_data = data_matrix[:, :, i].unsqueeze(2).float()
                    self.opts[i].zero_grad()
                    embeddings = self.encoders[i](single_variable_data)
                    contrastive_loss = self.contrastive_loss_fn(embeddings, labels)
                    contrastive_loss.backward()
                    self.opts[i].step()
                self.scheds[i].step()
                self.contrastive_losses[i].append(contrastive_loss.item())

    def plot_contrastive_losses(self, variable=None):
        plt.figure()
        plt.title("Contrastive Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")

        if variable:
            plt.plot(self.contrastive_losses[variable], label="Variable " + str(variable + 1))
        else:
            for i in range(self.num_variables):
                plt.plot(self.contrastive_losses[i], label="Variable " + str(i + 1))
        plt.legend()
        plt.show()

    def plot_latent_spaces(self, use_test=False):
        dl = self.train_dataloader
        if use_test:
            dl = self.test_dataloader
        for data_matrix, labels in dl:
            data_matrix, labels = data_matrix.to(device), labels.to(device)
            for i in range(self.num_variables):
                plt.figure()
                single_variable_data = data_matrix[:, :, i].unsqueeze(2).float()
                with torch.no_grad():
                    embeddings = self.encoders[i](single_variable_data)

                    reducer = umap.UMAP()
                    embeddings_2d = reducer.fit_transform(embeddings.cpu())

                    string_labels = np.array([self.classes[label.item()] for label in labels])

                    handles, lbls = [], []
                    for label in self.classes:
                        idx = np.where(string_labels == label)[0]
                        scatter = plt.scatter(embeddings_2d[idx, 0], embeddings_2d[idx, 1], label=label)
                        handles.append(scatter)
                        lbls.append(label)
                        
                    plt.legend(handles, lbls)
                    plt.title("Latent Space for Variable " + str(i + 1))
                    plt.show()

    def save(self, save_dir):
        for i in range(self.num_variables):
            save_name = save_dir + "encoder" + str(i+1) + ".pth"
            torch.save(self.encoders[i].state_dict(), save_name)

    def load(self, save_dir):
        for i in range(self.num_variables):
            save_name = save_dir + "encoder" + str(i+1) + ".pth"
            self.encoders[i].load_state_dict(torch.load(save_name))

    