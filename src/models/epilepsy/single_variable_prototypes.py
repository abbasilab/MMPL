import torch
import numpy as np
import matplotlib.pyplot as plt
from ...visualizations.umap_visualizer import UMAPLatent
from ...data.data import get_ds
from ..single_variables import (SingleVariableModulesWrapper, initialize_prototypes,
                                prototype_diversity_penalty, prototype_similarity_penalty, encoded_space_coverage_penalty)

if __name__ == "__main__":
    class_to_index={"epilepsy":0, "walking":1, "running":2,"sawing":3}
    train_ds, test_ds = get_ds("data/epilepsy/Epilepsy_TRAIN.ts", class_to_index), get_ds("data/epilepsy/Epilepsy_TEST.ts", class_to_index)

    sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=3, num_classes=4, hidden=40, num_prototypes=4)

    # sv_modules_wrapper.load_state_dict(torch.load("models/epilepsy/enc.dat"))

    # # Initialize the single-variable prototypes
    # initialize_prototypes(sv_modules_wrapper, train_ds)

    # # Disable gradients for the encoders
    # for i in range(sv_modules_wrapper.num_variables):
    #     for param in sv_modules_wrapper.single_variable_modules[i].encoder.parameters():
    #         param.requires_grad = False

    # data_train = torch.utils.data.DataLoader(train_ds, len(train_ds), True)
    # data_test = torch.utils.data.DataLoader(test_ds, len(test_ds), True)
    # whole_data_get = torch.utils.data.DataLoader(train_ds,len(train_ds),False)
    # whole_data_iter = iter(whole_data_get)
    # whole_data_tensor = next(whole_data_iter)[0]

    # opt = torch.optim.Adam(filter(lambda x: x.requires_grad, sv_modules_wrapper.parameters()), lr=0.001)
    # classification_loss_fn = torch.nn.CrossEntropyLoss()
    # sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)

    # epochs = 700
    # for epoch in range(epochs):
    #     for train, label in data_train:
    #         pred, second_degree = sv_modules_wrapper(train.float())
    #         classification_loss = classification_loss_fn(pred, label)

    #         embeddings = []
    #         for i in range(sv_modules_wrapper.num_variables):
    #             embeddings.append(sv_modules_wrapper.single_variable_modules[i](train[:, :, i].unsqueeze(2).float()))

    #         prototype_similarity_penalty_term = 0
    #         encoded_space_coverage_penalty_term = 0
    #         prototype_diversity_penalty_term = 0
    #         for i in range(sv_modules_wrapper.num_variables):
    #             prototype_similarity_penalty_term += prototype_similarity_penalty(whole_data_tensor[:, :, i].unsqueeze(2).float(), sv_modules_wrapper.single_variable_modules[i])
    #             encoded_space_coverage_penalty_term += encoded_space_coverage_penalty(whole_data_tensor[:, :, i].unsqueeze(2).float(), sv_modules_wrapper.single_variable_modules[i])
    #             prototype_diversity_penalty_term += prototype_diversity_penalty(sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix)
            
    #         total_loss = (1.)*classification_loss +                  \
    #                     (1.)*prototype_similarity_penalty_term +    \
    #                     (5.)*encoded_space_coverage_penalty_term +  \
    #                     (5.)*prototype_diversity_penalty_term
            
    #         opt.zero_grad()
    #         total_loss.backward()
    #         opt.step()
    #     sched.step()

    #     if epoch % 50 == 0:
    #         with torch.no_grad():
    #             numerator = 0
    #             denominator = 0
    #             for test, label in data_test:
    #                 aggregate_features, reject = sv_modules_wrapper(test.float())
    #                 sof = torch.softmax(aggregate_features, 1)
    #                 prediction = torch.argmax(sof, 1)

    #                 numerator += torch.sum(prediction.eq(label).int())
    #                 denominator += test.shape[0]
    #             accuracy = float(numerator) / float(denominator)

    #             print("Epoch: " + str(epoch) + " Accuracy: " + str(accuracy) + " Loss: ", str(total_loss.item()))
    
    # with torch.no_grad():
    #     numerator = 0
    #     denominator = 0
    #     for test, label in data_test:
    #         aggregate_features, reject = sv_modules_wrapper(test.float())
    #         sof = torch.softmax(aggregate_features, 1)
    #         prediction = torch.argmax(sof, 1)

    #         numerator += torch.sum(prediction.eq(label).int())
    #         denominator += test.shape[0]
    #     accuracy = float(numerator) / float(denominator)

    #     print("Final Accuracy: " + str(accuracy))

    sv_modules_wrapper.load_state_dict(torch.load("models/epilepsy/sv_modules_wrapper.dat"))
    fig, axes = plt.subplots(2, 3)
    colors = plt.cm.rainbow(np.linspace(0,1,5))

    visualize_moment = torch.utils.data.DataLoader(test_ds, len(test_ds), True)
    for test_sample in visualize_moment:
        inp, out = test_sample[0].detach(), test_sample[1].detach()
        for i in range(3):
            embeddings = sv_modules_wrapper.single_variable_modules[i].encoder(inp[:,:,i].unsqueeze(2).float())
            embeddings = torch.concat([embeddings, sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix], dim=0)
            out = torch.concat([out, 4*torch.ones((sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix.shape[0],))], dim=0)
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
    # torch.save(sv_modules_wrapper.state_dict(), "models/epilepsy/sv_modules_wrapper.dat")
