import pickle

import numpy as np
import matplotlib.pyplot as plt
import torch

from ...data.data import get_ds
from ..single_variables import SiameseContrastiveLoss, EncodingModule, SingleVariableModulesWrapper
from ...visualizations.umap_visualizer import UMAPLatent

if __name__ == "__main__":
    class_to_index={"standing":0, "running":1, "walking":2,"badminton":3}
    
    train_ds, test_ds = get_ds("data/basicmotions/BasicMotions_TRAIN.ts", class_to_index), get_ds("data/basicmotions/BasicMotions_TEST.ts", class_to_index)

    sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=6, num_classes=4, hidden=10, num_prototypes=4)

    # Initialize an encoding module for each variable
    encoding_module = EncodingModule(torch.nn.ModuleList([sv_module.encoder for sv_module in sv_modules_wrapper.single_variable_modules]))

    data_load = torch.utils.data.DataLoader(train_ds, len(train_ds), True)
    opt = torch.optim.Adam(params=encoding_module.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
    loss = SiameseContrastiveLoss()

    epochs = 2500
    for epoch in range(epochs):
        losses = []
        for data_matrix, labels in data_load:
            encoding_module.zero_grad()
            output = encoding_module(data_matrix.float())

            total_loss = 0
            for i in range(encoding_module.num_variables):
                total_loss += loss(output[i], labels)
            losses.append(float(total_loss))
            opt.zero_grad()
            total_loss.backward()
            opt.step()
        sched.step()

        avg_loss = float(sum(losses)) / float(len(losses))
        print("Epoch: ", epoch, " Average Loss: ", avg_loss)

    fileobj = open("models/basicmotions/enc.dat", "wb")
    pickle.dump(sv_modules_wrapper, fileobj)
    fileobj.close()
    
        