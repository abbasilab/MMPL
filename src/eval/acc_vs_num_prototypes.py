import argparse

import torch
from tqdm import tqdm

from src.data.data import get_ds
from src.models.encoding import Encoder
from src.train.encoding.trainer import EncoderTrainer
from src.models.single_variable_prototypes import SingleVariablePrototypesWrapper
from src.train.single_variable_prototypes.trainer import SingleVariablePrototypesTrainer
from src.models.multivariable_prototypes import MultivariableModule
from src.train.multivariable_prototypes.trainer import MultivariableModuleTrainer
from src.utils.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    config = get_config_from_dataset(args.dataset)
    encoding_config = config['encoding']
    single_variable_prototypes_config = config['single_variable_prototypes']
    multivariable_config = config['multivariable_prototypes']

    accuracies = []
    num_sv_prototypes = list(range(2, 11))
    for num_proto in num_sv_prototypes:
        encoders = []
        for i in range(config['num_variables']):
            encoders.append(Encoder(
                input_dim=encoding_config['input_dim'],
                hidden_dim=encoding_config['hidden_dim'],
                latent_dim=encoding_config['latent_dim']
            ).to(device))
        
        trainer = EncoderTrainer(
            encoders=encoders,
            train_ds=get_ds(get_train_path_from_dataset(args.dataset), config['class_to_index']),
            test_ds=get_ds(get_test_path_from_dataset(args.dataset, train=False), config['class_to_index']),
            classes=config['classes'],
            num_variables=config['num_variables'],
            batch_size=encoding_config['batch_size'],
            lr=encoding_config['lr'],
            gamma=encoding_config['gamma'],
            epochs=encoding_config['epochs'],
            m=encoding_config['m']
        ).to(device)

        trainer.train()

        wrapper = SingleVariablePrototypesWrapper(
            encoders=encoders,
            num_variables=config['num_variables'],
            num_classes=config['num_classes'],
            num_prototypes=num_proto,
            latent_dim=encoding_config['latent_dim'],
            num_layers=single_variable_prototypes_config['num_layers']
        ).to(device)


        trainer = SingleVariablePrototypesTrainer(
            wrapper=wrapper,
            train_ds=get_ds(get_train_path_from_dataset(args.dataset), config['class_to_index']),
            test_ds=get_ds(get_test_path_from_dataset(args.dataset, train=False), config['class_to_index']),
            classes=config['classes'],
            num_variables=config['num_variables'],
            num_prototypes=num_proto,
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

        multivariable_module = MultivariableModule(
            wrapper=wrapper,
            num_classes=config['num_classes'],
            num_variables=config['num_variables'],
            num_sv_prototypes=num_proto,
            num_layers=multivariable_config['num_layers']
        ).to(device)

        trainer = MultivariableModuleTrainer(
            multivariable_prototypes=multivariable_module,
            train_ds=get_ds(get_train_path_from_dataset(args.dataset), config['class_to_index']),
            test_ds=get_ds(get_test_path_from_dataset(args.dataset, train=False), config['class_to_index']),
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

        trainer.train()

        if args.dataset == "simulated_6400":
            accuracy = trainer.evaluate(use_test=True)
        else:
            accuracy = trainer.evaluate()
        accuracies.append(accuracy)

        print(f"Number of Single Variable Prototypes: {num_proto}\n Accuracy: {accuracy}")
    
    accuracies = torch.Tensor(accuracies)
    torch.save(accuracies, f"models/{args.dataset}/accuracies_vs_num_prototypes/accuracies.pth")
    torch.save(torch.Tensor(num_sv_prototypes, f"models/{args.dataset}/accuracies_vs_num_prototypes/num_protos.pth"))
    print(accuracies.mean(), accuracies.std())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g. <basicmotions>)")

    args = parser.parse_args()
    main(args)