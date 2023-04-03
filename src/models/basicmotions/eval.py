import torch

from ...data.data import get_ds
from ..single_variables import SingleVariableModulesWrapper
from ..multivariable import MultivariableModule

if __name__ == "__main__":
    # Load in dataset
    class_to_index={"standing":0, "running":1, "walking":2,"badminton":3}
    train_ds, test_ds = get_ds("data/basicmotions/BasicMotions_TRAIN.ts", class_to_index), get_ds("data/basicmotions/BasicMotions_TEST.ts", class_to_index)

    # Load in models
    sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=6, num_classes=4, hidden=10, num_prototypes=4)
    sv_modules_wrapper.load_state_dict(torch.load("models/basicmotions/sv_modules_wrapper.dat"))
    model = MultivariableModule(sv_modules_wrapper.single_variable_modules, num_variables=6, hidden=24, num_classes=4, num_prototypes=4)
    model.load_state_dict(torch.load("models/basicmotions/multivariable_module.dat"))

    # Evaluate model
    data_test = torch.utils.data.DataLoader(test_ds, len(test_ds), True)
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

    print("Accuracy: " + str(accuracy))
