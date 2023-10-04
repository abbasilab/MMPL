import argparse

import torch

from src.data.data import get_ds
from src.models.encoding import Encoder
from src.models.single_variable_prototypes import SingleVariablePrototypesWrapper
from src.models.multivariable_prototypes import MultivariableModule
from src.train.multivariable_prototypes.trainer import MultivariableModuleTrainer
from src.utils.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    config = get_config_from_dataset(args.dataset)
    encoding_config = config['encoding']
    single_variable_prototypes_config = config['single_variable_prototypes']
    multivariable_config = config['multivariable_prototypes']

    encoders = load_encoders(config)

    wrapper = load_single_variable_prototypes_wrapper(config)

    multivariable_module = MultivariableModule(
        wrapper=wrapper,
        num_classes=config['num_classes'],
        num_variables=config['num_variables'],
        num_sv_prototypes=single_variable_prototypes_config['num_prototypes'],
        num_layers=multivariable_config['num_layers']
    ).to(device)

    trainer = MultivariableModuleTrainer(
        multivariable_prototypes=multivariable_module,
        train_ds=get_ds(get_train_path_from_dataset(args.dataset), config['class_to_index']),
        test_ds=get_ds(get_test_path_from_dataset(args.dataset), config['class_to_index']),
        classes=config['classes'],
        num_variables=config['num_variables'],
        num_prototypes=config['num_classes'],
        num_layers=multivariable_config['num_layers'],
        batch_size=multivariable_config['batch_size'],
        lr=multivariable_config['lr'],
        gamma=multivariable_config['gamma'],
        epochs=multivariable_config['epochs'],
        l1=multivariable_config['l1'],
        l2=multivariable_config['l2'],
        l3=multivariable_config['l3'],
        l4=multivariable_config['l4'],
        d_min=multivariable_config['d_min']
    ).to(device)

    # trainer.initialize_prototypes()
    trainer.train()

    trainer.evaluate()

    if args.save:
        trainer.save(multivariable_config['save_dir'])

    if args.view:
        trainer.plot_classification_loss()
        trainer.plot_diversity_penalties()
        trainer.plot_similarity_penalties()
        trainer.plot_coverage_penalties()
        trainer.plot_prototypes_heatmap()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g. <basicmotions>)")
    parser.add_argument("--view", action=argparse.BooleanOptionalAction, default=False, help="Whether to view loss curves/latent spaces")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the model or not")

    args = parser.parse_args()
    main(args)