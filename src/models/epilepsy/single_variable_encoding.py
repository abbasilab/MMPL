import torch
import matplotlib.pyplot as plt
import numpy as np
from ...visualizations.umap_visualizer import UMAPLatent
from ...data.data import get_ds
from ..single_variables import SingleVariableModulesWrapper, EncodingModule, SiameseContrastiveLoss

if __name__ == "__main__":
    class_to_index={"epilepsy":0, "walking":1, "running":2,"sawing":3}
    train_ds, test_ds = get_ds("data/epilepsy/Epilepsy_TRAIN.ts", class_to_index), get_ds("data/epilepsy/Epilepsy_TEST.ts", class_to_index)

    sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=3, num_classes=4, hidden=40, num_prototypes=4)
    encoding_module = EncodingModule(torch.nn.ModuleList([sv_module.encoder for sv_module in sv_modules_wrapper.single_variable_modules]))

    data_load = torch.utils.data.DataLoader(train_ds, len(train_ds), True)
    opt = torch.optim.Adam(params=encoding_module.parameters(), lr=0.01)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
    contrastive_loss_fn = SiameseContrastiveLoss(m=1.0)
    batch_size = len(train_ds)

    # epochs = 350
    # for epoch in range(epochs):
    #     for data_matrix, labels in data_load:
    #         opt.zero_grad()
    #         indices = torch.randperm(len(data_matrix))[:batch_size]
    #         output = encoding_module(data_matrix[indices].float())
    #         total_loss = 0
    #         for i in range(encoding_module.num_variables):
    #             total_loss += contrastive_loss_fn(output[i], labels[indices])
    #         total_loss.backward()
    #         opt.step()
    #     sched.step()
    #     print("Epoch: ", epoch, " Total Loss: ", float(total_loss))

    sv_modules_wrapper.load_state_dict(torch.load("models/epilepsy/enc.dat"))

    fig, axes = plt.subplots(2, 3)
    colors = plt.cm.rainbow(np.linspace(0,1,5))

    visualize_moment = torch.utils.data.DataLoader(test_ds, len(test_ds), True)
    for test_sample in visualize_moment:
        inp, out = test_sample[0].detach(), test_sample[1].detach()
        for i in range(3):
            embeddings = sv_modules_wrapper.single_variable_modules[i].encoder(inp[:,:,i].unsqueeze(2).float())
            # embeddings = torch.concat([embeddings, sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix], dim=0)
            # out = torch.concat([out, 4*torch.ones((sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix.shape[0],))], dim=0)
            visualizer = UMAPLatent()
            trans = visualizer.umap.fit(embeddings.detach().numpy())
            X = trans.embedding_
            x_min, x_max = np.min(X, 0), np.max(X, 0)
            X = (X - x_min) / (x_max - x_min)
            ax = axes[(i // 3)][(i % 3)]
            ax.set_xlabel("UMAP-1", fontsize=7.5)
            ax.set_ylabel("UMAP-2", fontsize=7.5)
            ax.set_title("Variable " + str(i+1), y=0.0, pad=-35, fontsize=10.0)
            for j in range(X.shape[0]):
                classif = out[j]
                if classif != 4:
                    ax.plot(X[j, 0], X[j, 1], 'o', color=colors[int(classif.item())], alpha=0.5)
                else:
                    ax.plot(X[j, 0], X[j, 1], '*', color=colors[int(classif.item())], alpha=1.0, markersize=10.0, markeredgecolor="black", markeredgewidth=0.5)
            ax.xaxis.set_tick_params(labelsize=7.5)
            ax.yaxis.set_tick_params(labelsize=7.5)
            ax.locator_params(axis="x", nbins=3)
            ax.locator_params(axis="y", nbins=3)
    fig.tight_layout()
    fig.align_ylabels(axes)
    plt.show()
    torch.save(sv_modules_wrapper.state_dict(), "models/epilepsy/enc.dat")
