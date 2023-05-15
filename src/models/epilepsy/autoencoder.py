import torch
import random
import matplotlib.pyplot as plt
from ...data.data import get_ds
from ..single_variables import EncodingModule, SiameseContrastiveLoss

class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden):
        super(LSTMEncoder, self).__init__()
        self.input_size = input_size
        self.hidden = hidden

        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.lstm1 = torch.nn.LSTM(input_size=1, hidden_size=2*hidden, num_layers=1, batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=1, hidden_size=hidden, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(in_features=hidden, out_features=hidden)

    def forward(self, x):
        x, (hidden_n, cell_n) = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x
    
class LSTMDecoder(torch.nn.Module):
    def __init__(self, input_size, hidden):
        super(LSTMDecoder, self).__init__()
        self.input_size = input_size
        self.hidden = hidden

        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=input_size, num_layers=1, batch_first=True)
        self.lstm1 = torch.nn.LSTM(input_size=1, hidden_size=2*hidden, num_layers=1, batch_first=True)
        self.lstm2 = torch.nn.LSTM(input_size=1, hidden_size=input_size, num_layers=1, batch_first=True)
        self.linear = torch.nn.Linear(in_features=input_size, out_features=input_size)

    def forward(self, x):
        x = x.unsqueeze(2)
        x, (hidden_n, cell_n) = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x


class SingleVariableAutoencoder(torch.nn.Module):
    def __init__(self, input_size, hidden):
        super(SingleVariableAutoencoder, self).__init__()
        self.input_size = input_size
        self.hidden = hidden

        self.encoder = LSTMEncoder(input_size, hidden)
        self.decoder = LSTMDecoder(input_size, hidden)

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded
    
class EncodingModule(torch.nn.Module):
    def __init__(self, module_list):
        super().__init__()
        self.module_list = module_list
        self.num_variables = len(module_list)
    
    def forward(self, data):
        encoder_results = []
        decoder_results = []
        for i in range(self.num_variables):
            data_subset = data[:, :, i]
            decoded, encoded = self.module_list[i](data_subset.unsqueeze(2))
            encoder_results.append(encoded)
            decoder_results.append(decoded)
        return decoder_results, encoder_results 

if __name__ == "__main__":
    class_to_index={"epilepsy":0, "walking":1, "running":2,"sawing":3}
    train_ds, test_ds = get_ds("data/epilepsy/Epilepsy_TRAIN.ts", class_to_index), get_ds("data/epilepsy/Epilepsy_TEST.ts", class_to_index)

    autoencoders = []
    for i in range(3):
        autoencoders.append(SingleVariableAutoencoder(206, 60))
    encoding_module = EncodingModule(torch.nn.ModuleList(autoencoders))

    data_load = torch.utils.data.DataLoader(train_ds, len(train_ds), True)
    opt = torch.optim.Adam(params=encoding_module.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
    contrastive_loss_fn = SiameseContrastiveLoss(m=1.0)
    reconstruction_loss_fn = torch.nn.MSELoss()
    batch_size = 16

    epochs = 500
    for epoch in range(epochs):
        for data_matrix, labels in data_load:
            opt.zero_grad()
            indices = torch.randperm(len(data_matrix))[:batch_size]
            decoder_output, encoder_output = encoding_module(data_matrix[indices].float())
            contrastive_loss = 0
            reconstruction_loss = 0
            for i in range(encoding_module.num_variables):
                contrastive_loss += contrastive_loss_fn(encoder_output[i], labels[indices])
                reconstruction_loss += reconstruction_loss_fn(decoder_output.float(),
                                                              data_matrix[indices][:, :, i].float())
            total_loss = (1.0)*contrastive_loss + (1.0)*reconstruction_loss
            total_loss.backward()
            opt.step()
        sched.step()
        print("Epoch: ", epoch, " Total Loss: ", float(total_loss))

    torch.save(encoding_module.state_dict(), "models/epilepsy/autoenc.dat")

    encoding_module.load_state_dict(torch.load("models/epilepsy/autoenc.dat"))

    data_test = torch.utils.data.DataLoader(test_ds, len(test_ds), True)
    with torch.no_grad():
        for data_matrix, labels in data_test:
            index = random.randint(0, len(data_matrix) - 1)
            print(labels[index])
            orig = data_matrix[index]
            
            decoded_output, _ = encoding_module(data_matrix.float())

            fig, axes = plt.subplots(1, 3, figsize=(12,4))
            for i in range(3):
                ax = axes[i]
                decoded_point = decoded_output[i][index]
                ax.plot(orig[:, i], c='b', lw=0.5)
                ax.plot(decoded_point, c='r', lw=0.5)

            plt.show()



