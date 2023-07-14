import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from ...data.data import get_simulated_ds
from ..single_variables import SiameseContrastiveLoss, EncodingModule, SingleVariableModulesWrapper, initialize_prototypes, prototype_diversity_penalty, prototype_similarity_penalty, encoded_space_coverage_penalty
from ..multivariable import MultivariableModule, similarity_penalty1, similarity_penalty3, diversity_penalty
from .single_variable_encoding import LSTMEncoder

if __name__ == "__main__":

    num_iter = 30
    accuracies = []
    for it in tqdm(range(num_iter)):

        class_to_index={"Pattern 1":0, "Pattern 2":1, "Pattern 3":2,"Pattern 4":3}
        
        train_ds, class_descriptor = get_simulated_ds(100)
        test_ds, _ = get_simulated_ds(100)
        print(len(train_ds))

        # Initialize an encoding module for each variable
        encoders = [LSTMEncoder(100, 30) for _  in range(4)]
        encoding_module = EncodingModule(torch.nn.ModuleList(encoders))

        data_load = torch.utils.data.DataLoader(train_ds, len(train_ds), False)
        opt = torch.optim.Adam(params=encoding_module.parameters(), lr=0.01)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
        loss = SiameseContrastiveLoss(m=0.1)
        batch_size = len(train_ds)

        epochs = 500
        for epoch in range(epochs):
            for data_matrix, labels in data_load:
                # indices = torch.randperm(len(data_matrix))[:batch_size]
                # output = encoding_module(data_matrix[indices].float())
                output = encoding_module(data_matrix.float())

                total_loss = 0
                for i in range(encoding_module.num_variables):
                    total_loss += loss(output[i], labels)
                opt.zero_grad()
                total_loss.backward()
                opt.step()
            sched.step()

        sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=4, num_classes=2, hidden=30, num_prototypes=2)
        for i in range(len(sv_modules_wrapper.single_variable_modules)):
                module = sv_modules_wrapper.single_variable_modules[i]
                module.encoder = encoding_module.module_list[i]

        initialize_prototypes(sv_modules_wrapper, train_ds)

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
                            (1.)*encoded_space_coverage_penalty_term +  \
                            (1.)*prototype_diversity_penalty_term
                
                opt.zero_grad()
                total_loss.backward()
                opt.step()
            sched.step()
        
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
                                    num_variables=4, hidden=8, num_classes=8, num_prototypes=8)
        model.initialize_prototypes(train_ds)

        opt = torch.optim.Adam(filter(lambda x: x.requires_grad, model.parameters()), lr=0.01)
        sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.999)
        classification_loss = torch.nn.CrossEntropyLoss()

        data_train = torch.utils.data.DataLoader(train_ds, len(train_ds), True)
        data_test = torch.utils.data.DataLoader(test_ds, len(test_ds), True)

        epochs = 500
        for epoch in range(epochs):
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
            accuracies.append(accuracy)

    print(torch.std_mean(torch.tensor(accuracies)))

