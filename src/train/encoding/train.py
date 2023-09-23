import argparse

import torch

from src.data.data import get_ds
from src.models.encoding import Encoder
from src.train.encoding.trainer import EncoderTrainer
from src.utils.utils import get_config_from_dataset, get_train_path_from_dataset, get_test_path_from_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    config = get_config_from_dataset(args.dataset)
    encoding_config = config['encoding']

    encoders = []
    for i in range(config['num_variables']):
        encoders.append(Encoder(
            input_dim=encoding_config['input_dim'],
            hidden_dim=encoding_config['hidden_dim'],
            latent_dim=encoding_config['latent_dim']
        ))
    
    trainer = EncoderTrainer(
        encoders=encoders,
        train_ds=get_ds(get_train_path_from_dataset(args.dataset), config['class_to_index']),
        test_ds=get_ds(get_test_path_from_dataset(args.dataset), config['class_to_index']),
        classes=config['classes'],
        num_variables=config['num_variables'],
        batch_size=encoding_config['batch_size'],
        lr=encoding_config['lr'],
        gamma=encoding_config['gamma'],
        epochs=encoding_config['epochs'],
        m=encoding_config['m']
    ).to(device)

    trainer.train()

    if args.view:
        trainer.plot_contrastive_losses()
        trainer.plot_latent_spaces()

    if args.save:
        trainer.save(encoding_config['save_dir'])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g. <basicmotions>)")
    parser.add_argument("--view", action=argparse.BooleanOptionalAction, default=False, help="Whether to view loss curves/latent spaces")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the model or not")

    args = parser.parse_args()
    main(args)