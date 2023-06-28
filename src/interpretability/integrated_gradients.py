import torch
from captum.attr import IntegratedGradients
import matplotlib.pyplot as plt
import seaborn as sns

from ..data.data import get_ds
from ..models.single_variables import SingleVariableModulesWrapper
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
    class_to_index={"standing":0, "running":1, "walking":2,"badminton":3}
    index_to_class = {0:"Standing", 1:"Running", 2:"Walking", 3:"Badminton"}
    train_ds, test_ds = get_ds("data/basicmotions/BasicMotions_TRAIN.ts", class_to_index), get_ds("data/basicmotions/BasicMotions_TEST.ts", class_to_index)

    sv_modules_wrapper = SingleVariableModulesWrapper(6, 4, 10, 4)
    sv_modules_wrapper.load_state_dict(torch.load("models/basicmotions/sv_modules_wrapper.dat"))
    model = MultivariableModule(sv_modules_wrapper.single_variable_modules, 6, 24, 4, 4)
    model.load_state_dict(torch.load("models/basicmotions/multivariable_module.dat"))

    data_train = torch.utils.data.DataLoader(train_ds, len(train_ds), False)
    for train, label in data_train:
        all_attributions = integrated_gradients(wrapped_model, train.float(), 0)

        single_variable_ts = train[0, :, 5]
        attribution = all_attributions[0, :, 5]
        print(single_variable_ts.size(), attribution.size())
        ax = plt.axes()
        ax.plot(single_variable_ts, c='k', lw=1.0)
        ax.set_ylim(-1., 2.)
        sns.heatmap(attribution.unsqueeze(0).detach().numpy(), cbar=False, ax=ax)
        ax.set_ylim(-1., 2.)
        plt.show()