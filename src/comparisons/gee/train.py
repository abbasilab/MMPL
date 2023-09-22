import matplotlib.pyplot as plt
import torch
from tqdm import tqdm

from src.data.data import get_ds
from src.comparisons.gee.model import AutoencoderPrototypeModel

class Trainer(torch.nn.Module):
    def __init__(self, autoencoder_prototype_model, train_ds, test_ds, batch_size, lr, gamma, epochs, l1, l2, l3, l4):
        super(Trainer, self).__init__()
        self.model = autoencoder_prototype_model
        self.train_ds = train_ds
        self.test_ds = test_ds
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

    def train(self):
        for epoch in tqdm(range(self.epochs)):
            for data_matrix, labels in self.train_dataloader:
                predictions, reconstructions, embeddings = self.model(data_matrix.float())
                classification_loss = self.classification_loss_fn(predictions, labels)
                self.classification_losses.append(classification_loss.item())

                reconstruction_loss = self.reconstruction_loss_fn(reconstructions, data_matrix.float())
                self.reconstruction_losses.append(reconstruction_loss.item())

                total_loss = classification_loss + \
                             (self.l1)*reconstruction_loss
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
            for data_matrix, label in dl:
                output, _, _ = self.model(data_matrix.float())
                sof = torch.softmax(output, 1)
                prediction = torch.argmax(sof, 1)

                numerator += torch.sum(prediction.eq(label).int())
                denominator += data_matrix.shape[0]
            accuracy = float(numerator) / float(denominator)
            print("Accuracy: " + str(accuracy))

if __name__ == "__main__":
    model = AutoencoderPrototypeModel(
        input_dim=3,
        latent_dim=1,
        autoencoder_num_layers=2,
        num_prototypes=4,
        seq_len=206,
        num_classes=4,
        num_layers=4
    )
    model.float()

    # class_to_index={"standing":0, "running":1, "walking":2,"badminton":3}
    # train_ds, test_ds = get_ds("data/basicmotions/processed/train.ts", class_to_index), get_ds("data/basicmotions/processed/test.ts", class_to_index)
    class_to_index={"epilepsy":0, "walking":1, "running":2,"sawing":3}
    train_ds, test_ds = get_ds("data/epilepsy/processed/train.ts", class_to_index), get_ds("data/epilepsy/processed/test.ts", class_to_index)
    trainer = Trainer(
        model,
        train_ds,
        test_ds,
        batch_size=137,
        lr=0.01,
        gamma=0.999,
        epochs=1000,
        l1=1.0,
        l2=1.0,
        l3=1.0,
        l4=1.0
    )

    trainer.train()
    trainer.plot_classification_loss()
    trainer.plot_reconstruction_loss()
    trainer.eval()
    trainer.eval(use_test=True)

