import torch

from ...data.data import get_ds
from ..single_variables import SingleVariableModulesWrapper, initialize_prototypes, prototype_diversity_penalty, prototype_similarity_penalty, encoded_space_coverage_penalty

if __name__ == "__main__":
    class_to_index={"standing":0, "running":1, "walking":2,"badminton":3}
    train_ds, test_ds = get_ds("data/basicmotions/BasicMotions_TRAIN.ts", class_to_index), get_ds("data/basicmotions/BasicMotions_TEST.ts", class_to_index)

    sv_modules_wrapper = SingleVariableModulesWrapper(6, 4, 10, 4)
    sv_modules_wrapper.load_state_dict(torch.load("models/basicmotions/enc.dat"))

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

    epochs = 2000
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
                        (0.1)*prototype_diversity_penalty_term
            
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

    torch.save(sv_modules_wrapper.state_dict(), "models/basicmotions/sv_modules_wrapper.dat")

