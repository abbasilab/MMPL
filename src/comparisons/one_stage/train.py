import argparse

from src.comparisons.one_stage.model import AutoencoderPrototypeModel
from src.comparisons.one_stage.trainer import Trainer
from src.data.data import get_ds
from src.utils.utils import *

def main(args):
    config = get_comparison_config_from_dataset("one_stage", args.dataset)
    one_stage_config = config['one_stage']

    model = AutoencoderPrototypeModel(
        input_dim=one_stage_config['input_dim'],
        hidden_dim=one_stage_config['hidden_dim'],
        latent_dim=one_stage_config['latent_dim'],
        num_prototypes=one_stage_config['num_prototypes'],
        seq_len=one_stage_config['seq_len'],
        num_classes=one_stage_config['num_classes'],
        num_layers=one_stage_config['num_layers']
    )

    trainer = Trainer(
        autoencoder_prototype_model=model,
        train_ds=get_ds(get_train_path_from_dataset(args.dataset), config['class_to_index']),
        test_ds=get_ds(get_test_path_from_dataset(args.dataset), config['class_to_index']),
        classes=config['classes'],
        batch_size=one_stage_config['batch_size'],
        lr=one_stage_config['lr'],
        gamma=one_stage_config['gamma'],
        epochs=one_stage_config['epochs'],
        l1=one_stage_config['l1'],
        l2=one_stage_config['l2'],
        l3=one_stage_config['l3'],
        l4=one_stage_config['l4'],
        d_min=one_stage_config['d_min']
    )

    trainer.train()

    if args.view:
        trainer.plot_classification_loss()
        trainer.plot_diversity_penalties()
        trainer.plot_similarity_penalties()
        trainer.plot_coverage_penalties()
        trainer.eval()
        trainer.visualize_latent_space()
        trainer.visualize_prototypes()

    if args.save:
        trainer.save(one_stage_config['save_dir'])
        trainer.load(one_stage_config['save_dir'])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g. <basicmotions>)")
    parser.add_argument("--view", action=argparse.BooleanOptionalAction, default=False, help="Whether to view loss curves/latent spaces")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the model or not")
    parser.add_argument("--initialize", action=argparse.BooleanOptionalAction, default=False, help="Whether to use k-means++ initialization for prototypes")
    args = parser.parse_args()
    main(args)