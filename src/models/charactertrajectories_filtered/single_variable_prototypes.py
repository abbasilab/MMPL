import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

from ...visualizations.umap_visualizer import UMAPLatent
from ...data.data import get_ds, filter_classes
from ..single_variables import EncodingModule, SingleVariableModulesWrapper, initialize_prototypes, prototype_diversity_penalty, prototype_similarity_penalty, encoded_space_coverage_penalty
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

    # Initialize the single-variable prototypes
    initialize_prototypes(sv_modules_wrapper, filtered_train)

    visualize_moment = torch.utils.data.DataLoader(filtered_test, len(filtered_test))
    for train_sample  in visualize_moment:
            inp, out = train_sample[0].detach(), train_sample[1].detach()
            num_vars = inp.shape[-1]
            for i in range(3):
                embeddings = sv_modules_wrapper.single_variable_modules[i].encoder(inp[:,:,i].unsqueeze(2).float())
                embeddings = torch.concat([embeddings, sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix], dim=0)
                out = torch.concat([out, 4*torch.ones((sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix.shape[0],))], dim=0)
                visualizer = UMAPLatent()
                visualizer.visualize(embeddings, out, 4)
    plt.show()
    choice = input()
    if choice == "n":
        exit()

    # Disable gradients for the encoders
    for i in range(3):
        for param in sv_modules_wrapper.single_variable_modules[i].encoder.parameters():
            param.requires_grad = False

    data_train = torch.utils.data.DataLoader(filtered_train, len(filtered_train), True)
    data_test = torch.utils.data.DataLoader(filtered_test, len(filtered_test), True)
    whole_data_get = torch.utils.data.DataLoader(filtered_train,len(filtered_train),False)
    whole_data_iter = iter(whole_data_get)
    whole_data_tensor = next(whole_data_iter)[0]

    opt = torch.optim.Adam(filter(lambda x: x.requires_grad, sv_modules_wrapper.parameters()), lr=0.01)
    classification_loss_fn = torch.nn.CrossEntropyLoss()
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)

    epochs = 500
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

    torch.save(sv_modules_wrapper.state_dict(), "models/charactertrajectories_filtered/sv_modules_wrapper.dat")

    sv_modules_wrapper.load_state_dict(torch.load("models/charactertrajectories_filtered/sv_modules_wrapper.dat"))

    visualize_moment = torch.utils.data.DataLoader(filtered_test, len(filtered_test))
    for train_sample  in visualize_moment:
            inp, out = train_sample[0].detach(), train_sample[1].detach()
            num_vars = inp.shape[-1]
            for i in range(3):
                embeddings = sv_modules_wrapper.single_variable_modules[i].encoder(inp[:,:,i].unsqueeze(2).float())
                embeddings = torch.concat([embeddings, sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix], dim=0)
                out = torch.concat([out, 4*torch.ones((sv_modules_wrapper.single_variable_modules[i].protolayer.prototype_matrix.shape[0],))], dim=0)
                visualizer = UMAPLatent()
                visualizer.visualize(embeddings, out, 4)
    plt.show()
