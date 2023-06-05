import torch
from tqdm import tqdm
from ...data.data import get_ds, filter_classes
from ..single_variables import (
    SiameseContrastiveLoss, EncodingModule, SingleVariableModulesWrapper, 
    initialize_prototypes, prototype_diversity_penalty, prototype_similarity_penalty, encoded_space_coverage_penalty)
from ..multivariable import MultivariableModule, similarity_penalty1, similarity_penalty3, diversity_penalty
from .single_variable_encoding import LSTMEncoder

if __name__ == "__main__":
    accuracies = []
    for it in tqdm(range(30)):
        class_to_index = {
        "1":1, "2":2, "3":3, "4":4, "5":5, "6":6, "7":7, "8":8,
        "9":9, "10":10, "11":11, "12":12, "13":13, "14":14, "15":15,
        "16":16, "17":17, "18":18, "19":19, "20":20
        }
        
        train_ds, test_ds = get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TRAIN.ts", class_to_index), get_ds("data/charactertrajectories/CharacterTrajectoriesEq_TEST.ts", class_to_index)
        filtered_train, filtered_test = filter_classes(train_ds, [2, 4, 12, 13]), filter_classes(test_ds, [2, 4, 12, 13])

        encoders = [LSTMEncoder(119, 10) for _  in range(3)]
        encoding_module = EncodingModule(torch.nn.ModuleList(encoders))

        data_load = torch.utils.data.DataLoader(filtered_train, len(filtered_train), True)
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

        sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=3, num_classes=4, hidden=10, num_prototypes=4)
        for i in range(len(sv_modules_wrapper.single_variable_modules)):
            module = sv_modules_wrapper.single_variable_modules[i]
            module.encoder = encoding_module.module_list[i]

        # Initialize the single-variable prototypes
        initialize_prototypes(sv_modules_wrapper, filtered_train)

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
        for epoch in range(epochs):
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
        
        model = MultivariableModule(single_variable_modules=sv_modules_wrapper.single_variable_modules, \
                                 num_variables=3, hidden=12, num_classes=4, num_prototypes=4)
        model.initialize_prototypes(filtered_train)

        opt = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=0.01)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
        classification_loss = torch.nn.CrossEntropyLoss()

        data_train = torch.utils.data.DataLoader(filtered_train, len(filtered_train), True)
        data_test = torch.utils.data.DataLoader(filtered_test, len(filtered_test), True)

        epochs = 700
        for epoch in range(epochs):
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
            print("Final Accuracy: ", float(accuracy))
            accuracies.append(accuracy)