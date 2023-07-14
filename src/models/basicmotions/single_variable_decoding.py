import torch
import random
import matplotlib.pyplot as plt
from ...data.data import get_ds
from ..single_variables import EncodingModule, DecodingModule, SingleVariableModulesWrapper

if __name__ == "__main__":
    class_to_index={"standing":0, "running":1, "walking":2,"badminton":3}
    train_ds, test_ds = get_ds("data/basicmotions/BasicMotions_TRAIN.ts", class_to_index), get_ds("data/basicmotions/BasicMotions_TEST.ts", class_to_index)

    sv_modules_wrapper = SingleVariableModulesWrapper(6, 4, 10, 4)
    sv_modules_wrapper.load_state_dict(torch.load("models/basicmotions/enc.dat"))
    encoding_module = EncodingModule([module.encoder for module in sv_modules_wrapper.single_variable_modules])

    decoding_module = DecodingModule(10, 100, 6)

    data_load = torch.utils.data.DataLoader(train_ds, len(train_ds), True)
    opt = torch.optim.Adam(params=decoding_module.parameters(), lr=0.1)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
    loss = torch.nn.MSELoss()
    batch_size = 40

    epochs = 100
    for epoch in range(epochs):
        for data_matrix, labels in data_load:
            indices = torch.randperm(len(data_matrix))[:batch_size]
            with torch.no_grad():
                encoded = encoding_module(data_matrix[indices].float())

            decoded = decoding_module(encoded)

            total_loss = 0
            for i in range(decoding_module.num_variables):
                total_loss += loss(decoded[:, :, i].float(), data_matrix[indices][:, :, i].float())
            opt.zero_grad()
            total_loss.backward()
            opt.step()
        sched.step()
        print("Epoch: ", epoch, " Total Loss: ", float(total_loss))

    torch.save(decoding_module.state_dict(), "models/basicmotions/dec.dat")

    decoding_module.load_state_dict(torch.load("models/basicmotions/dec.dat"))

    data_test = torch.utils.data.DataLoader(test_ds, len(test_ds), True)
    with torch.no_grad():
        for data_matrix, labels in data_load:
            index = random.randint(0, len(data_matrix) - 1)
            orig = data_matrix[index]
            
            encoded = encoding_module(data_matrix.float())
            decoded = decoding_module(encoded)[index]

            fig, axes = plt.subplots(2, 3)
            for i in range(2):
                for j in range(3):
                    ax = axes[i][j]
                    decoded_point = decoded[:, 3*i + j]
                    ax.plot(orig[:, 3*i + j], c='b', lw=0.5)
                    ax.plot(decoded_point, c='r', lw=0.5)

            plt.show()

