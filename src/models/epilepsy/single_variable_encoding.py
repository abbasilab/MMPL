import torch
import numpy as np
from ...data.data import get_ds
from ..single_variables import SingleVariableModulesWrapper, EncodingModule, SiameseContrastiveLoss

if __name__ == "__main__":
    class_to_index={"epilepsy":0, "walking":1, "running":2,"sawing":3}
    train_ds, test_ds = get_ds("data/epilepsy/Epilepsy_TRAIN.ts", class_to_index), get_ds("data/epilepsy/Epilepsy_TEST.ts", class_to_index)
    
    sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=3, num_classes=4, hidden=20, num_prototypes=6)
    encoding_module = EncodingModule(torch.nn.ModuleList([sv_module.encoder for sv_module in sv_modules_wrapper.single_variable_modules]))

    data_load = torch.utils.data.DataLoader(train_ds, len(train_ds), True)
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

    torch.save(sv_modules_wrapper.state_dict(), "models/epilepsy/enc.dat")

