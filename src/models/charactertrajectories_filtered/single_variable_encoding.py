import torch
import matplotlib.pyplot as plt

from ...data.data import get_ds, filter_classes
from ..single_variables import SiameseContrastiveLoss, EncodingModule
from ...visualizations.umap_visualizer import UMAPLatent

class LSTMEncoder(torch.nn.Module):
    def __init__(self, input_size, hidden):
        super().__init__()
        self.input_size = input_size
        self.hidden = hidden

        self.lstm = torch.nn.LSTM(input_size=1, hidden_size=hidden, num_layers=2, batch_first=True)
        self.linear = torch.nn.Linear(in_features=hidden, out_features=hidden)

    def forward(self, x):
        x, (hidden_n, cell_n) = self.lstm(x)
        x = x[:, -1, :]
        x = self.linear(x)
        return x

if __name__ == "__main__":
    class_to_index = {
        "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
        "9":9, "10":10, "11":11, "12":12, "13":13, "14":14, "15":15,
        "16":16, "17":17, "18":18, "19":19, "20":20
    }
    
    train_ds, test_ds = get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TRAIN.ts", class_to_index), get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TEST.ts", class_to_index)
    filtered_train, filtered_test = filter_classes(train_ds, [2, 4, 12, 13]), filter_classes(test_ds, [2, 4, 12, 13])

    encoders = [LSTMEncoder(119, 10) for _  in range(3)]
    encoding_module = EncodingModule(torch.nn.ModuleList(encoders))

    data_load = torch.utils.data.DataLoader(filtered_train, len(filtered_train), True)
    opt = torch.optim.Adam(params=encoding_module.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
    loss = SiameseContrastiveLoss(m=1.0)

    epochs = 2000
    for epoch in range(epochs):
        for data_matrix, labels in data_load:
            output = encoding_module(data_matrix.float())

            total_loss = 0
            for i in range(encoding_module.num_variables):
                total_loss += loss(output[i], labels)
            opt.zero_grad()
            total_loss.backward()
            opt.step()
        sched.step()
        print("Epoch: ", epoch, " Total Loss: ", float(total_loss))

    torch.save(encoding_module.state_dict(), "models/charactertrajectories_filtered/enc.dat")

    encoding_module.load_state_dict(torch.load("models/charactertrajectories_filtered/enc.dat"))
    visualize_moment = torch.utils.data.DataLoader(filtered_test, len(filtered_test))
    for train_sample in visualize_moment:
            inp, out = train_sample[0].detach(), train_sample[1].detach()
            for i in range(3):
                embeddings = encoding_module.module_list[i](inp[:,:,i].unsqueeze(2).float())
                visualizer = UMAPLatent()
                visualizer.visualize(embeddings, out, 4)
    plt.show()
        