import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

from ...data.data import get_simulated_ds
from ..single_variables import EncodingModule, SingleVariableModulesWrapper, initialize_prototypes, prototype_diversity_penalty, prototype_similarity_penalty, encoded_space_coverage_penalty
from ..multivariable import MultivariableModule, similarity_penalty1, similarity_penalty3, diversity_penalty
from ...visualizations.umap_visualizer import UMAPLatent
from .single_variable_encoding import LSTMEncoder

if __name__ == "__main__":
    class_to_index={"Pattern 1":0, "Pattern 2":1, "Pattern 3":2,"Pattern 4":3}
    
    train_ds, class_descriptor = get_simulated_ds(100)
    test_ds, _ = get_simulated_ds(100)
    print(len(train_ds))

    encoders = [LSTMEncoder(100, 30) for _  in range(4)]
    encoding_module = EncodingModule(torch.nn.ModuleList(encoders))
    encoding_module.load_state_dict(torch.load("models/simulated_binary/enc.dat"))

    sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=4, num_classes=2, hidden=30, num_prototypes=2)
    for i in range(len(sv_modules_wrapper.single_variable_modules)):
            module = sv_modules_wrapper.single_variable_modules[i]
            module.encoder = encoding_module.module_list[i]
    sv_modules_wrapper.load_state_dict(torch.load("models/simulated_binary/sv_modules_wrapper.dat"))

    model = MultivariableModule(single_variable_modules=sv_modules_wrapper.single_variable_modules, \
                                 num_variables=4, hidden=8, num_classes=8, num_prototypes=8)
    model.initialize_prototypes(train_ds)
    sns.heatmap(torch.relu(model.aggregate_prototype_layer.protos).detach().numpy())
    plt.show()
    choice = input()
    if choice == "n":
        exit()

    opt = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=0.01)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
    classification_loss = torch.nn.CrossEntropyLoss()

    data_train = torch.utils.data.DataLoader(train_ds, len(train_ds), True)
    data_test = torch.utils.data.DataLoader(test_ds, len(test_ds), True)

    epochs = 500
    for epoch in tqdm(range(epochs)):
        for train, label in data_train:
            pred, second_degree = model(train.float())

            class_loss = classification_loss(pred, label)
            total_loss = (1.)*class_loss + \
                         (1.)*similarity_penalty1(second_degree, model.aggregate_prototype_layer.protos) + \
                         (1.)*similarity_penalty3(second_degree, model.aggregate_prototype_layer.protos) + \
                         (10.)*diversity_penalty(model.aggregate_prototype_layer.protos)

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

    torch.save(model.state_dict(), "models/simulated_binary/multivariable_module.dat")

    model.load_state_dict(torch.load("models/simulated_binary/multivariable_module.dat"))
    sns.heatmap(torch.relu(model.aggregate_prototype_layer.protos).detach().numpy())
    plt.show()