import torch
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from ...data.data import get_ds, filter_classes
from ..single_variables import SingleVariableModulesWrapper
from ..multivariable import MultivariableModule, similarity_penalty1, similarity_penalty3, diversity_penalty
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
    sv_modules_wrapper.load_state_dict(torch.load("models/charactertrajectories/sv_modules_wrapper_bdpq.dat"))

    model = MultivariableModule(single_variable_modules=sv_modules_wrapper.single_variable_modules, \
                                 num_variables=3, hidden=12, num_classes=4, num_prototypes=4)
    model.load_state_dict(torch.load("models/charactertrajectories/multivariable_module_bdpq.dat"))
    # model.initialize_prototypes(filtered_train)
    # print(model.aggregate_prototype_layer.protos)

    visualize_moment = torch.utils.data.DataLoader(filtered_train, len(filtered_train))
    for train_sample  in visualize_moment:
            inp, out = train_sample[0].detach(), train_sample[1].detach()
            num_vars = inp.shape[-1]
            for i in range(3):
                embeddings = sv_modules_wrapper.single_variable_modules[i].encoder(inp[:,:,i].unsqueeze(2).float())
                embeddings = torch.concat([embeddings, sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix], dim=0)
                out = torch.concat([out, 4*torch.ones((sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix.shape[0],))], dim=0)
                visualizer = UMAPLatent()
                visualizer.visualize(embeddings, out, 4)


    protos = model.aggregate_prototype_layer.protos
    min_val = protos.min()
    max_val = protos.max()
    scaled_protos = (protos - min_val) / (max_val - min_val)
    plt.figure()
    sns.heatmap(torch.relu(scaled_protos).detach().numpy())
    plt.show()
    exit()
    

    opt = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=0.01)
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
    classification_loss = torch.nn.CrossEntropyLoss()

    # data_train = torch.utils.data.DataLoader(train_ds, len(train_ds), True)
    # data_test = torch.utils.data.DataLoader(test_ds, len(test_ds), True)
    data_train = torch.utils.data.DataLoader(filtered_train, len(filtered_train), True)
    data_test = torch.utils.data.DataLoader(filtered_test, len(filtered_test), True)

    epochs = 700
    for epoch in tqdm(range(epochs)):
        for train, label in data_train:
            pred, second_degree = model(train.float())

            class_loss = classification_loss(pred, label)
            total_loss = (1.)*class_loss + \
                         (1.)*similarity_penalty1(second_degree, model.aggregate_prototype_layer.protos) + \
                         (1.)*similarity_penalty3(second_degree, model.aggregate_prototype_layer.protos) + \
                         (5.)*diversity_penalty(model.aggregate_prototype_layer.protos)

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

    
    sns.heatmap(torch.relu(model.aggregate_prototype_layer.protos).detach().numpy())
    plt.show()
    torch.save(model.state_dict(), "models/charactertrajectories/multivariable_module_bdpq.dat")