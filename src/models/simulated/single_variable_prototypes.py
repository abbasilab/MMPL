import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from ...data.data import get_simulated_ds
from ..single_variables import EncodingModule, SingleVariableModulesWrapper, initialize_prototypes, prototype_diversity_penalty, prototype_similarity_penalty, encoded_space_coverage_penalty
from ...visualizations.umap_visualizer import UMAPLatent
from .single_variable_encoding import LSTMEncoder

if __name__ == "__main__":
    class_to_index={"Pattern 1":0, "Pattern 2":1, "Pattern 3":2,"Pattern 4":3}
    
    train_ds = torch.load("data/simulated/train_10.dat")
    test_ds = torch.load("data/simulated/test_10.dat")
    _, class_descriptor = get_simulated_ds(10)

    encoders = [LSTMEncoder(100, 30) for _  in range(4)]
    encoding_module = EncodingModule(torch.nn.ModuleList(encoders))
    encoding_module.load_state_dict(torch.load("models/simulated/enc.dat"))

    sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=4, num_classes=64, hidden=30, num_prototypes=4)
    for i in range(len(sv_modules_wrapper.single_variable_modules)):
            module = sv_modules_wrapper.single_variable_modules[i]
            module.encoder = encoding_module.module_list[i]

    initialize_prototypes(sv_modules_wrapper, train_ds)

    visualize_moment = torch.utils.data.DataLoader(test_ds, len(test_ds))
    for train_sample in visualize_moment:
            inp, out = train_sample[0].detach(), train_sample[1].detach()
            num_vars = inp.shape[-1]
            for i in range(4):
                embeddings = sv_modules_wrapper.single_variable_modules[i].encoder(inp[:,:,i].unsqueeze(2).float())
                embeddings = torch.concat([embeddings, sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix], dim=0)
                
                if i != 3:
                    act = []
                    for label in out:
                        act.append(class_descriptor[label][1][i])
                else:
                    act = torch.zeros(len(out))
                act = torch.FloatTensor(act)
                act = torch.concat([act, 4*torch.ones((sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix.shape[0],))], dim=0)
                visualizer = UMAPLatent()
                visualizer.visualize(embeddings, act, 4)
    plt.show()
    choice = input()
    if choice == "n":
        exit()


    # Disable gradients for the encoders
    for i in range(4):
        for param in sv_modules_wrapper.single_variable_modules[i].encoder.parameters():
            param.requires_grad = False

    data_train = torch.utils.data.DataLoader(train_ds, len(train_ds), True)
    data_test = torch.utils.data.DataLoader(test_ds, len(test_ds), True)
    whole_data_get = torch.utils.data.DataLoader(train_ds,len(train_ds),False)
    whole_data_iter = iter(whole_data_get)
    whole_data_tensor = next(whole_data_iter)[0]

    opt = torch.optim.Adam(filter(lambda x: x.requires_grad, sv_modules_wrapper.parameters()), lr=0.1)
    classification_loss_fn = torch.nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)

    epochs = 100
    for epoch in tqdm(range(epochs)):
        for train, label in data_train:
            pred, second_degree = sv_modules_wrapper(train.float())
            classification_loss = classification_loss_fn(pred, label)

            prototype_similarity_penalty_term = 0
            encoded_space_coverage_penalty_term = 0
            prototype_diversity_penalty_term = 0
            for i in range(sv_modules_wrapper.num_variables):
                prototype_similarity_penalty_term += prototype_similarity_penalty(whole_data_tensor[:, :, i].unsqueeze(2).float(), sv_modules_wrapper.single_variable_modules[i])
                encoded_space_coverage_penalty_term += encoded_space_coverage_penalty(whole_data_tensor[:, :, i].unsqueeze(2).float(), sv_modules_wrapper.single_variable_modules[i])
                prototype_diversity_penalty_term += prototype_diversity_penalty(sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix)
            
            total_loss = (1.)*classification_loss +                  \
                        (1.)*prototype_similarity_penalty_term +    \
                        (10.)*encoded_space_coverage_penalty_term +  \
                        (1.)*prototype_diversity_penalty_term
            
            opt.zero_grad()
            total_loss.backward()
            opt.step()
        sched.step()

        if epoch % 50 == 0:
            with torch.no_grad():
                numerator = 0
                denominator = 0
                for test, label in data_test:
                    aggregate_features, reject = sv_modules_wrapper(test.float())
                    sof = torch.softmax(aggregate_features, 1)
                    prediction = torch.argmax(sof, 1)

                    numerator += torch.sum(prediction.eq(label).int())
                    denominator += test.shape[0]
                accuracy = float(numerator) / float(denominator)

                print("Epoch: " + str(epoch) + " Accuracy: " + str(accuracy) + " Loss: ", str(total_loss.item()))
    
    with torch.no_grad():
        numerator = 0
        denominator = 0
        for test, label in data_test:
            aggregate_features, reject = sv_modules_wrapper(test.float())
            sof = torch.softmax(aggregate_features, 1)
            prediction = torch.argmax(sof, 1)

            numerator += torch.sum(prediction.eq(label).int())
            denominator += test.shape[0]
        accuracy = float(numerator) / float(denominator)

        print("Final Accuracy: " + str(accuracy))

    torch.save(sv_modules_wrapper.state_dict(), "models/simulated/sv_modules_wrapper.dat")

    sv_modules_wrapper.load_state_dict(torch.load("models/simulated/sv_modules_wrapper.dat"))

    visualize_moment = torch.utils.data.DataLoader(test_ds, len(test_ds))
    for train_sample in visualize_moment:
            inp, out = train_sample[0].detach(), train_sample[1].detach()
            num_vars = inp.shape[-1]
            for i in range(4):
                embeddings = sv_modules_wrapper.single_variable_modules[i].encoder(inp[:,:,i].unsqueeze(2).float())
                embeddings = torch.concat([embeddings, sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix], dim=0)
                
                if i != 3:
                    act = []
                    for label in out:
                        act.append(class_descriptor[label][1][i])
                else:
                    act = torch.zeros(len(out))
                act = torch.FloatTensor(act)
                act = torch.concat([act, 4*torch.ones((sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix.shape[0],))], dim=0)
                visualizer = UMAPLatent()
                visualizer.visualize(embeddings, act, 4)
    plt.show()



