import torch
import matplotlib.pyplot as plt
import random
from ...data.data import get_ds, filter_classes
from ..single_variables import SiameseContrastiveLoss, EncodingModule, SingleVariableModulesWrapper
from ...visualizations.umap_visualizer import UMAPLatent

if __name__ == "__main__":
    class_to_index = {
        "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
        "9":9, "10":10, "11":11, "12":12, "13":13, "14":14, "15":15,
        "16":16, "17":17, "18":18, "19":19, "20":20
    }
    
    train_ds, test_ds = get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TRAIN.ts", class_to_index), get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TEST.ts", class_to_index)
    filtered_train, filtered_test = filter_classes(train_ds, [2, 4, 12, 13]), filter_classes(test_ds, [2, 4, 12, 13])
    
    sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=3, num_classes=4, hidden=20, num_prototypes=4)

    # Initialize an encoding module for each variable
    encoding_module = EncodingModule(torch.nn.ModuleList([sv_module.encoder for sv_module in sv_modules_wrapper.single_variable_modules]))

    data_load = torch.utils.data.DataLoader(filtered_train, len(filtered_train), True)
    # data_load = torch.utils.data.DataLoader(train_ds, len(train_ds), True)
    opt = torch.optim.Adam(params=encoding_module.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
    loss = SiameseContrastiveLoss(m=1.0)

    # epochs = 2000
    # for epoch in range(epochs):
    #     for data_matrix, labels in data_load:
    #         output = encoding_module(data_matrix.float())

    #         total_loss = 0
    #         for i in range(encoding_module.num_variables):
    #             total_loss += loss(output[i], labels)
    #         opt.zero_grad()
    #         total_loss.backward()
    #         opt.step()
    #     sched.step()
    #     print("Epoch: ", epoch, " Total Loss: ", float(total_loss))

    # torch.save(sv_modules_wrapper.state_dict(), "models/charactertrajectories/enc_bdpq.dat")

    sv_modules_wrapper.load_state_dict(torch.load("models/charactertrajectories/enc_bdpq.dat"))
    visualize_moment = torch.utils.data.DataLoader(filtered_test, len(filtered_test))
    # visualize_moment = torch.utils.data.DataLoader(test_ds, len(test_ds))
    for train_sample  in visualize_moment:
            inp, out = train_sample[0].detach(), train_sample[1].detach()
            out = out - 1
            num_vars = inp.shape[-1]
            for i in range(3):
                embeddings = sv_modules_wrapper.single_variable_modules[i].encoder(inp[:,:,i].unsqueeze(2).float())
                visualizer = UMAPLatent()
                visualizer.visualize(embeddings, out, 4)
    plt.show()
        