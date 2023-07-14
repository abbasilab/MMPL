import torch
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns

from ..models.simulated.single_variable_encoding import LSTMEncoder
from ..data.data import get_ds, get_simulated_ds, filter_classes
from ..models.single_variables import EncodingModule, SingleVariableModulesWrapper
from ..models.multivariable import MultivariableModule

def wrapped_model(inp):
    return model(inp)[0]

def integrated_gradients(model, input, target_class_index, baseline=None):
    if baseline == None:
        baseline = torch.zeros_like(input)

    ig = IntegratedGradients(model)
    attributions, approximation_error = ig.attribute(input, target=target_class_index,
                                                      return_convergence_delta=True)
    return attributions




if __name__ == "__main__":
    class_to_index={"Pattern 1":0, "Pattern 2":1, "Pattern 3":2,"Pattern 4":3}
    
    train_ds = torch.load("data/simulated/train_10.dat")
    test_ds = torch.load("data/simulated/test_10.dat")
    _, class_descriptor = get_simulated_ds(10)

    encoders = [LSTMEncoder(100, 30) for _  in range(4)]
    encoding_module = EncodingModule(torch.nn.ModuleList(encoders))
    encoding_module.load_state_dict(torch.load("models/simulated/enc.dat"))

    sv_modules_wrapper = SingleVariableModulesWrapper(num_variables=4, num_classes=64, hidden=30, num_prototypes=4)
    for i in range(len(sv_modules_wrapper.single_variable_modules)):
            module = sv_modules_wrapper.single_variable_modules[i]
            module.encoder = encoding_module.module_list[i]
    sv_modules_wrapper.load_state_dict(torch.load("models/simulated/sv_modules_wrapper.dat"))

    model = MultivariableModule(single_variable_modules=sv_modules_wrapper.single_variable_modules, \
                                 num_variables=4, hidden=16, num_classes=64, num_prototypes=64)
    model.load_state_dict(torch.load("models/simulated/multivariable_module.dat"))

    # class_to_index={"epilepsy":0, "walking":1, "running":2,"sawing":3}
    # index_to_class = {0:"Epilepsy", 1:"Walking", 2:"Running", 3:"Sawing"}
    # train_ds, test_ds = get_ds("data/epilepsy/Epilepsy_TRAIN.ts", class_to_index), get_ds("data/epilepsy/Epilepsy_TEST.ts", class_to_index)

    # sv_modules_wrapper = SingleVariableModulesWrapper(3, 4, 40, 4)
    # sv_modules_wrapper.load_state_dict(torch.load("models/epilepsy/sv_modules_wrapper.dat"))
    # model = MultivariableModule(sv_modules_wrapper.single_variable_modules, 3, 12, 4, 4)
    # model.load_state_dict(torch.load("models/epilepsy/multivariable_module.dat"))

    data_train = torch.utils.data.DataLoader(train_ds, len(train_ds), False)
    for train, label in data_train:
        train = train[:10]
        all_attributions = integrated_gradients(wrapped_model, train.float(), 0)

        single_variable_ts = train[0, :, 0]
        attribution = all_attributions[0, :, 0]
        print(single_variable_ts.size(), attribution.size())
        ax = plt.axes()
        ax.plot(single_variable_ts, c='k', lw=1.0)
        ax.set_ylim(-1., 2.)
        sns.heatmap(attribution.unsqueeze(0).detach().numpy(), cbar=False, ax=ax)
        ax.set_ylim(-1., 2.)
        plt.show()