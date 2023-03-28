import torch
import matplotlib.pyplot as plt
import seaborn as sns

from ...data.data import get_ds
from ..single_variables import SingleVariableModulesWrapper
from ..multivariable import MultivariableModule, similarity_penalty1, similarity_penalty3, diversity_penalty
from ...visualizations.umap_visualizer import UMAPLatent

if __name__ == "__main__":
    class_to_index={"standing":0, "running":1, "walking":2,"badminton":3}
    train_ds, test_ds = get_ds("data/basicmotions/BasicMotions_TRAIN.ts", class_to_index), get_ds("data/basicmotions/BasicMotions_TEST.ts", class_to_index)

    sv_modules_wrapper = SingleVariableModulesWrapper(6, 4, 10, 4)
    sv_modules_wrapper.load_state_dict(torch.load("models/basicmotions/sv_modules_wrapper.dat"))

    model = MultivariableModule(single_variable_modules=sv_modules_wrapper.single_variable_modules, \
                                 num_variables=6, hidden=24, num_classes=4, num_prototypes=4)
    model.initialize_prototypes(train_ds)
    
    opt = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=0.1)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
    classification_loss = torch.nn.CrossEntropyLoss()

    data_train = torch.utils.data.DataLoader(train_ds, 64, True)
    data_test = torch.utils.data.DataLoader(test_ds, 64, True)

    epochs = 2000
    for epoch in range(epochs):
        for train, label in data_train:
            pred, second_degree = model(train.float())

            class_loss = classification_loss(pred, label)
            total_loss = (1.)*class_loss + (1.)*similarity_penalty1(second_degree, model.aggregate_prototype_layer.protos) + \
                (1.)*similarity_penalty3(second_degree, model.aggregate_prototype_layer.protos) + \
                    (1.)*diversity_penalty(model.aggregate_prototype_layer.protos)

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

    torch.save(model.state_dict(), "models/basicmotions/multivariable_module.dat")

    plt.figure()
    sns.heatmap(torch.relu(model.aggregate_prototype_layer.protos).detach().numpy())
    plt.xlabel("Single Variable Prototypes")
    plt.ylabel("Multivariable Prototype Index")
    plt.show()


