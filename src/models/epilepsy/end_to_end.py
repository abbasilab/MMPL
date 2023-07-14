import torch

from ...data.data import get_ds
from ..single_variables import (
    SiameseContrastiveLoss, EncodingModule, SingleVariableModulesWrapper, 
    initialize_prototypes, prototype_diversity_penalty, prototype_similarity_penalty, encoded_space_coverage_penalty)
from ..multivariable import MultivariableModule, similarity_penalty1, similarity_penalty3, diversity_penalty

if __name__ == "__main__":
    accuracies = []
    for it in range(1):
        print("Iteration: " + str(it + 1))
        class_to_index={"epilepsy":0, "walking":1, "running":2,"sawing":3}
        train_ds, test_ds = get_ds("data/epilepsy/Epilepsy_TRAIN.ts", class_to_index), get_ds("data/epilepsy/Epilepsy_TEST.ts", class_to_index)
        sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=3, num_classes=4, hidden=40, num_prototypes=5)

        # Initialize an encoding module for each variable
        encoding_module = EncodingModule(torch.nn.ModuleList([sv_module.encoder for sv_module in sv_modules_wrapper.single_variable_modules]))

        data_load = torch.utils.data.DataLoader(train_ds, len(train_ds), True)
        opt = torch.optim.Adam(params=encoding_module.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
        loss = SiameseContrastiveLoss(m=1.0)

        epochs = 2000
        for epoch in range(epochs):
            for data_matrix, labels in data_load:
                output = encoding_module(data_matrix.float())

                total_loss = 0
                for i in range(encoding_module.num_variables):
                    total_loss += loss(output[i], labels)
                opt.zero_grad()
                total_loss.backward()
                opt.step()
            sched.step()
            print("Epoch: ", epoch, " Total Loss: ", float(total_loss))

        # Initialize the single-variable prototypes
        initialize_prototypes(sv_modules_wrapper, train_ds)

        # Disable gradients for the encoders
        for i in range(sv_modules_wrapper.num_variables):
            for param in sv_modules_wrapper.single_variable_modules[i].encoder.parameters():
                param.requires_grad = False

        

        data_train = torch.utils.data.DataLoader(train_ds, len(train_ds), True)
        data_test = torch.utils.data.DataLoader(test_ds, len(test_ds), True)
        whole_data_get = torch.utils.data.DataLoader(train_ds,len(train_ds),False)
        whole_data_iter = iter(whole_data_get)
        whole_data_tensor = next(whole_data_iter)[0]

        opt = torch.optim.Adam(filter(lambda x: x.requires_grad, sv_modules_wrapper.parameters()), lr=0.001)
        classification_loss_fn = torch.nn.CrossEntropyLoss()
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)

        epochs = 700
        for epoch in range(epochs):
            for train, label in data_train:
                pred, second_degree = sv_modules_wrapper(train.float())
                classification_loss = classification_loss_fn(pred, label)

                embeddings = []
                for i in range(sv_modules_wrapper.num_variables):
                    embeddings.append(sv_modules_wrapper.single_variable_modules[i](train[:, :, i].unsqueeze(2).float()))

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

                    # print("Epoch: " + str(epoch) + " Accuracy: " + str(accuracy) + " Loss: ", str(total_loss.item()))
        
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

            # print("Final Accuracy: " + str(accuracy))

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
                    # print("Epoch: ", epoch, "Accuracy: ", accuracy, "Loss: ", float(total_loss))
        
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
            accuracies.append(accuracy)
            
    print(torch.std_mean(torch.tensor(accuracies)))