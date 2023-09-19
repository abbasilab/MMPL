import argparse

import torch

from src.data.data import get_ds
from src.visualizations.datasets.basicmotions import basicmotions_visualize
from src.visualizations.datasets.epilepsy import epilepsy_visualize
from src.visualizations.datasets.charactertrajectories_filtered import charactertrajectories_filtered_visualize

def main(args):
    
    if args.dataset == "basicmotions":
        basicmotions_visualize(args.dataset, args.type, args.save)
    elif args.dataset == "epilepsy":
        epilepsy_visualize(args.dataset, args.type, args.save)
    elif args.dataset == "charactertrajectories_filtered":
        charactertrajectories_filtered_visualize(args.dataset, args.type, args.save)

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g. <basicmotions>)")
    parser.add_argument("--type", type=str, help="What do you want to visualize? single-var, multi-var")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the figure or not")

    args = parser.parse_args()
    main(args)