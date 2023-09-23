import argparse

import torch

from src.data.data import get_ds
from src.models.encoding import Encoder
from src.models.single_variable_prototypes import SingleVariablePrototypesWrapper
from src.train.single_variable_prototypes.trainer import SingleVariablePrototypesTrainer
from src.utils.utils import get_config_from_dataset, get_train_path_from_dataset, get_test_path_from_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    config = get_config_from_dataset(args.dataset)
    encoding_config = config['encoding']
    single_variable_prototypes_config = config['single_variable_prototypes']

    encoders = []
    for i in range(config['num_variables']):
        encoder = Encoder(
            input_dim=encoding_config['input_dim'],
            hidden_dim=encoding_config['hidden_dim'],
            latent_dim=encoding_config['latent_dim']
        ).to(device)
        encoder.load_state_dict(torch.load(encoding_config['save_dir'] + "encoder" + str(i+1) + ".pth", map_location=device))
        encoders.append(encoder)

    wrapper = SingleVariablePrototypesWrapper(
        encoders=encoders,
        num_variables=config['num_variables'],
        num_classes=config['num_classes'],
        num_prototypes=single_variable_prototypes_config['num_prototypes'],
        latent_dim=encoding_config['latent_dim'],
        num_layers=single_variable_prototypes_config['num_layers'],
        dropout=single_variable_prototypes_config['dropout'],
    ).to(device)


    trainer = SingleVariablePrototypesTrainer(
        wrapper=wrapper,
        train_ds=get_ds(get_train_path_from_dataset(args.dataset), config['class_to_index']),
        test_ds=get_ds(get_test_path_from_dataset(args.dataset), config['class_to_index']),
        classes=config['classes'],
        num_variables=config['num_variables'],
        num_prototypes=single_variable_prototypes_config['num_prototypes'],
        num_layers=single_variable_prototypes_config['num_layers'],
        batch_size=single_variable_prototypes_config['batch_size'],
        lr=single_variable_prototypes_config['lr'],
        gamma=single_variable_prototypes_config['gamma'],
        epochs=single_variable_prototypes_config['epochs'],
        l1=single_variable_prototypes_config['l1'],
        l2=single_variable_prototypes_config['l2'],
        l3=single_variable_prototypes_config['l3'],
        l4=single_variable_prototypes_config['l4'],
        d_min=single_variable_prototypes_config['d_min']
    ).to(device)

    trainer.initialize_prototypes()
    trainer.train()

    trainer.evaluate()

    if args.save:
        trainer.save(single_variable_prototypes_config['save_dir'])

    if args.view:
        trainer.plot_classification_loss()
        trainer.plot_diversity_penalties()
        trainer.plot_similarity_penalties()
        trainer.plot_coverage_penalties()
        trainer.plot_all_latent_spaces_with_prototypes()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g. <basicmotions>)")
    parser.add_argument("--view", action=argparse.BooleanOptionalAction, default=False, help="Whether to view loss curves/latent spaces")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the model or not")

    args = parser.parse_args()
    main(args)