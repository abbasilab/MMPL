import argparse

from src.comparisons.no_contrastive.model import SingleVariablePrototypesWrapper2, MultivariableModule
from src.comparisons.no_contrastive.univariate_trainer import Trainer
from src.comparisons.no_contrastive.multivariate_trainer import MultivariableModuleTrainer
from src.data.data import get_ds
from src.utils.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    config = get_comparison_config_from_dataset("no_contrastive", args.dataset)
    single_var_config = config['single_variable_prototypes']

    wrapper = SingleVariablePrototypesWrapper2(
        num_variables=config['num_variables'],
        num_classes=config['num_classes'],
        num_prototypes=single_var_config['num_prototypes'],
        hidden_dim=single_var_config['hidden_dim'],
        latent_dim=single_var_config['latent_dim'],
        num_layers=single_var_config['num_layers']
    ).to(device)

    trainer = Trainer(
        wrapper=wrapper,
        train_ds=get_ds(get_train_path_from_dataset(args.dataset), config['class_to_index']),
        test_ds=get_ds(get_test_path_from_dataset(args.dataset), config['class_to_index']),
        classes=config['classes'],
        num_variables=config['num_variables'],
        num_prototypes=single_var_config['num_prototypes'],
        num_layers=single_var_config['num_layers'],
        batch_size=single_var_config['batch_size'],
        lr=single_var_config['lr'],
        gamma=single_var_config['gamma'],
        epochs=single_var_config['epochs'],
        l1=single_var_config['l1'],
        l2=single_var_config['l2'],
        l3=single_var_config['l3'],
        l4=single_var_config['l4'],
        d_min=single_var_config['d_min']
    )

    trainer.load(single_var_config['save_dir'])
    wrapper = trainer.wrapper

    multi_config = config['multivariable_prototypes']

    model = MultivariableModule(
        wrapper=wrapper,
        num_classes=config['num_classes'],
        num_variables=config['num_variables'],
        num_sv_prototypes=single_var_config['num_prototypes'],
        num_layers=multi_config['num_layers']
    )
    mtrainer = MultivariableModuleTrainer(
        multivariable_prototypes=model,
        train_ds=get_ds(get_train_path_from_dataset(args.dataset), config['class_to_index']),
        test_ds=get_ds(get_test_path_from_dataset(args.dataset), config['class_to_index']),
        classes=config['classes'],
        num_variables=config['num_variables'],
        num_prototypes=config['num_classes'],
        num_layers=multi_config['num_layers'],
        batch_size=multi_config['batch_size'],
        lr=multi_config['lr'],
        gamma=multi_config['gamma'],
        epochs=multi_config['epochs'],
        l1=multi_config['l1'],
        l2=multi_config['l2'],
        l3=multi_config['l3'],
        l4=multi_config['l4'],
        d_min=multi_config['d_min']
    ).to(device)

    mtrainer.evaluate()
    mtrainer.evaluate(use_test=True)

    if args.view:
        mtrainer.plot_classification_loss()
        mtrainer.plot_diversity_penalties()
        mtrainer.plot_similarity_penalties()
        mtrainer.plot_coverage_penalties()
        mtrainer.plot_total_loss()
        mtrainer.plot_prototypes_heatmap()

    if args.save:
        mtrainer.save(multi_config['save_dir'])
        mtrainer.load(multi_config['save_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g. <basicmotions>)")
    parser.add_argument("--view", action=argparse.BooleanOptionalAction, default=False, help="Whether to view loss curves/latent spaces")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the model or not")
    parser.add_argument("--initialize", action=argparse.BooleanOptionalAction, default=False, help="Whether to use k-means++ initialization for prototypes")
    args = parser.parse_args()
    main(args)