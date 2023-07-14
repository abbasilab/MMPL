import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from ...data.data import get_ds, filter_classes
from ..single_variables import EncodingModule, SingleVariableModulesWrapper
from ..multivariable import MultivariableModule, similarity_penalty1, similarity_penalty3, diversity_penalty
from .single_variable_encoding import LSTMEncoder

if __name__ == "__main__":
    class_to_index = {
        "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
        "9":9, "10":10, "11":11, "12":12, "13":13, "14":14, "15":15,
        "16":16, "17":17, "18":18, "19":19, "20":20
    }
    train_ds, test_ds = get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TRAIN.ts", class_to_index), get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TEST.ts", class_to_index)
    filtered_train, filtered_test = filter_classes(train_ds, [2, 4, 12, 13]), filter_classes(test_ds, [2, 4, 12, 13])

    encoding_module = EncodingModule(torch.nn.ModuleList([LSTMEncoder(119, 10) for _ in range(3)]))
    encoding_module.load_state_dict(torch.load("models/charactertrajectories_filtered/enc.dat"))

    sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=3, num_classes=4, hidden=10, num_prototypes=4)
    for i in range(len(sv_modules_wrapper.single_variable_modules)):
        module = sv_modules_wrapper.single_variable_modules[i]
        module.encoder = encoding_module.module_list[i]
    sv_modules_wrapper.load_state_dict(torch.load("models/charactertrajectories_filtered/sv_modules_wrapper.dat"))

    model = MultivariableModule(single_variable_modules=sv_modules_wrapper.single_variable_modules, \
                                 num_variables=3, hidden=12, num_classes=4, num_prototypes=4)
    model.initialize_prototypes(filtered_train)
    sns.heatmap(torch.relu(model.aggregate_prototype_layer.protos).detach().numpy())
    plt.show()
    choice = input()
    if choice == "n":
        exit()

    opt = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=0.001)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
    classification_loss = torch.nn.CrossEntropyLoss()

    data_train = torch.utils.data.DataLoader(filtered_train, len(filtered_train), True)
    data_test = torch.utils.data.DataLoader(filtered_test, len(filtered_test), True)

    epochs = 700
    for epoch in tqdm(range(epochs)):
        for train, label in data_train:
            pred, second_degree = model(train.float())

            class_loss = classification_loss(pred, label)
            total_loss = (1.)*class_loss + \
                         (1.)*similarity_penalty1(second_degree, model.aggregate_prototype_layer.protos) + \
                         (10.)*similarity_penalty3(second_degree, model.aggregate_prototype_layer.protos) + \
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

    torch.save(model.state_dict(), "models/charactertrajectories_filtered/multivariable_module.dat")

    model.load_state_dict(torch.load("models/charactertrajectories_filtered/multivariable_module.dat"))
    sns.heatmap(torch.relu(model.aggregate_prototype_layer.protos).detach().numpy())
    plt.show()
    