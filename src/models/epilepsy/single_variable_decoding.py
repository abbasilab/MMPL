import torch

from ...data.data import get_ds
from ..single_variables import EncodingModule, DecodingModule, SingleVariableModulesWrapper

if __name__ == "__main__":
    class_to_index={"epilepsy":0, "walking":1, "running":2,"sawing":3}
    train_ds, test_ds = get_ds("data/epilepsy/Epilepsy_TRAIN.ts", class_to_index), get_ds("data/epilepsy/Epilepsy_TEST.ts", class_to_index)

    sv_modules_wrapper = SingleVariableModulesWrapper(3, 4, 40, 4)
    sv_modules_wrapper.load_state_dict(torch.load("models/epilepsy/enc.dat"))
    encoding_module = EncodingModule([module.encoder for module in sv_modules_wrapper.single_variable_modules])

    decoding_module = DecodingModule(40, 206, 3)

    data_load = torch.utils.data.DataLoader(train_ds, len(train_ds), True)
    opt = torch.optim.Adam(params=decoding_module.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
    loss = torch.nn.MSELoss()
    batch_size = len(train_ds)

    epochs = 1000
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

    torch.save(decoding_module.state_dict(), "models/epilepsy/dec.dat")
