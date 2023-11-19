import argparse

from src.comparisons.no_contrastive.model import SingleVariablePrototypesWrapper2
from src.comparisons.no_contrastive.univariate_trainer import Trainer
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
    ).to(device)

    trainer.train()
    trainer.evaluate()

    if args.view:
        trainer.plot_classification_loss()
        trainer.plot_diversity_penalties()
        trainer.plot_similarity_penalties()
        trainer.plot_coverage_penalties()
        trainer.plot_total_loss()
        trainer.visualize_single_variable_prototypes()

    if args.save:
        trainer.save(single_var_config['save_dir'])
        trainer.load(single_var_config['save_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g. <basicmotions>)")
    parser.add_argument("--view", action=argparse.BooleanOptionalAction, default=False, help="Whether to view loss curves/latent spaces")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the model or not")
    args = parser.parse_args()
    main(args)