import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from ...data.data import get_ds, filter_classes
from ..single_variables import SingleVariableModulesWrapper
from ..multivariable import MultivariableModule, similarity_penalty1, similarity_penalty3, diversity_penalty

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
    class_to_index = {
        "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
        "9":9, "10":10, "11":11, "12":12, "13":13, "14":14, "15":15,
        "16":16, "17":17, "18":18, "19":19, "20":20
    }
    train_ds, test_ds = get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TRAIN.ts", class_to_index), get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TEST.ts", class_to_index)
    filtered_train, filtered_test = filter_classes(train_ds, [2, 4, 12, 13]), filter_classes(test_ds, [2, 4, 12, 13])

    autoencoders = []
    for i in range(3):
        autoencoders.append(SingleVariableAutoencoder(119, 10))
    encoding_module = EncodingModule(torch.nn.ModuleList(autoencoders))
    encoding_module.load_state_dict(torch.load("models/charactertrajectories_filtered/autoenc.dat"))

    sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=3, num_classes=4, hidden=10, num_prototypes=5)
    for i in range(3):
        sv_modules_wrapper.single_variable_modules[i].encoder = encoding_module.module_list[i].encoder
    sv_modules_wrapper.load_state_dict(torch.load("models/charactertrajectories_filtered/sv_modules_wrapper.dat"))

    model = MultivariableModule(single_variable_modules=sv_modules_wrapper.single_variable_modules, \
                                 num_variables=3, hidden=15, num_classes=4, num_prototypes=4)
    model.initialize_prototypes(filtered_train)
    sns.heatmap(torch.relu(model.aggregate_prototype_layer.protos).detach().numpy())
    plt.show()
    choice = input()
    if choice == "n":
        exit()

    opt = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=0.01)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
    classification_loss = torch.nn.CrossEntropyLoss()

    # data_train = torch.utils.data.DataLoader(train_ds, len(train_ds), True)
    # data_test = torch.utils.data.DataLoader(test_ds, len(test_ds), True)
    data_train = torch.utils.data.DataLoader(filtered_train, len(filtered_train), True)
    data_test = torch.utils.data.DataLoader(filtered_test, len(filtered_test), True)

    epochs = 700
    for epoch in tqdm(range(epochs)):
        for train, label in data_train:
            pred, second_degree = model(train.float())

            class_loss = classification_loss(pred, label)
            total_loss = (1.)*class_loss + \
                         (1.)*similarity_penalty1(second_degree, model.aggregate_prototype_layer.protos) + \
                         (1.)*similarity_penalty3(second_degree, model.aggregate_prototype_layer.protos) + \
                         (5.)*diversity_penalty(model.aggregate_prototype_layer.protos)

            opt.zero_grad()
            total_loss.backward()
            opt.step()
        sched.step()

        if epoch % 50 == 0:
            with torch.no_grad():
                numerator = 0
                denominator = 0
                for test, label in data_test:
                    pred, reject = model(test.float())
                    sof = torch.softmax(pred, 1)
                    prediction = torch.argmax(sof, 1)
                    numerator += torch.sum(prediction.eq(label).int())
                    denominator += test.shape[0]
                accuracy = float(numerator/denominator)
                print("Epoch: ", epoch, "Accuracy: ", accuracy, "Loss: ", float(total_loss))
    
    with torch.no_grad():
        numerator = 0
        denominator = 0
        for test, label in data_test:
            pred, reject = model(test.float())
            sof = torch.softmax(pred, 1)
            prediction = torch.argmax(sof, 1)
            numerator += torch.sum(prediction.eq(label).int())
            denominator += test.shape[0]
        accuracy = float(numerator/denominator)
        print("Final Accuracy: ", accuracy)

    torch.save(model.state_dict(), "models/charactertrajectories_filtered/multivariable_module.dat")

    model.load_state_dict(torch.load("models/charactertrajectories_filtered/multivariable_module.dat"))
    sns.heatmap(torch.relu(model.aggregate_prototype_layer.protos).detach().numpy())
    plt.show()
    