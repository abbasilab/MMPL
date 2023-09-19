import argparse

import torch

from src.data.data import get_ds
from src.visualizations.datasets.basicmotions import basicmotions_visualize

def main(args):
    
    if args.dataset == "basicmotions":
        basicmotions_visualize(args.dataset, args.type, args.save)
    elif args.dataset == "epilepsy":
        return
    elif args.dataset == "charactertrajectories_filtered":
        return

    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g. <basicmotions>)")
    parser.add_argument("--type", type=str, help="What do you want to visualize? single-var, multi-var")
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=False, help="Whether to save the figure or not")

    args = parser.parse_args()
    main(args)