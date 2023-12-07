import random

import h5py
from sktime.datasets import load_from_tsfile_to_dataframe
import torch

def main():
    """
    One time script just to do any preprocessing we need to do.
    Code here will probably not be neat :)
    """

    # Filter out CharacterTrajectories to just "b", "d", "p", and "q"
    # mapping = {":2": ":b", ":4": ":d", ":12": ":p", ":13": ":q"}
    # with open("data/charactertrajectories/processed/CharacterTrajectoriesEq_TRAIN.ts", 'r') as infile:
    #     lines = infile.readlines()

    # with open("data/charactertrajectories_filtered/processed/train.ts", 'w') as outfile:
    #     for i, line in enumerate(lines):
    #         if i < 7:
    #             outfile.write(line)
    #         else:
    #             stripped_line = line.strip()
    #             for old, new in mapping.items():
    #                 if stripped_line.endswith(old):
    #                     outfile.write(stripped_line[:-len(old)] + new + "\n")
    #                     break

    # with open("data/charactertrajectories/processed/CharacterTrajectoriesEq_TEST.ts", 'r') as infile:
    #     lines = infile.readlines()

    # with open("data/charactertrajectories_filtered/processed/test.ts", 'w') as outfile:
    #     for i, line in enumerate(lines):
    #         if i < 7:
    #             outfile.write(line)
    #         else:
    #             stripped_line = line.strip()
    #             for old, new in mapping.items():
    #                 if stripped_line.endswith(old):
    #                     outfile.write(stripped_line[:-len(old)] + new + "\n")
    #                     break

    # # Create validation sets, 80/20 split on train data
    # with open("data/charactertrajectories_filtered/processed/train.ts", 'r') as f:
    #     lines = f.readlines()

    # metadata = lines[:7]
    # data = lines[7:]

    # with open("data/charactertrajectories_filtered/processed/train.ts", 'w') as f:
    #     f.writelines(metadata)
    #     f.writelines(data)


    # mapping = {":2": ":b", ":4": ":d", ":12": ":p", ":13": ":q"}
    # with open("data/charactertrajectories/raw/CharacterTrajectories_TRAIN.ts", 'r') as infile:
    #     lines = infile.readlines()

    # with open("data/charactertrajectories_filtered/raw/train.ts", 'w') as outfile:
    #     for i, line in enumerate(lines):
    #         if i < 42:
    #             outfile.write(line)
    #         else:
    #             stripped_line = line.strip()
    #             for old, new in mapping.items():
    #                 if stripped_line.endswith(old):
    #                     outfile.write(stripped_line[:-len(old)] + new + "\n")
    #                     break

    # with open("data/charactertrajectories/raw/CharacterTrajectories_TEST.ts", 'r') as infile:
    #     lines = infile.readlines()

    # with open("data/charactertrajectories_filtered/raw/test.ts", 'w') as outfile:
    #     for i, line in enumerate(lines):
    #         if i < 42:
    #             outfile.write(line)
    #         else:
    #             stripped_line = line.strip()
    #             for old, new in mapping.items():
    #                 if stripped_line.endswith(old):
    #                     outfile.write(stripped_line[:-len(old)] + new + "\n")
    #                     break

    with open("data/charactertrajectories/processed/train.ts", 'r') as infile:
        lines = infile.readlines()
    
    with open("data/charactertrajectories/processed/train_new.ts", 'w') as outfile:
        for i, line in enumerate(lines):
            if i < 7:
                outfile.write(line)
            else:
                line = line.strip()
                data = "".join(line.split(":")[:-1])
                label = int(line.split(":")[-1])
                new_label = label - 1
                new_line = data + str(new_label)
                outfile.write(new_line)

    with open("data/charactertrajectories/processed/test.ts", 'r') as infile:
        lines = infile.readlines()
    
    with open("data/charactertrajectories/processed/test_new.ts", 'w') as outfile:
        for i, line in enumerate(lines):
            if i < 7:
                outfile.write(line)
            else:
                line = line.strip()
                data = "".join(line.split(":")[:-1])
                label = int(line.split(":")[-1])
                new_label = label - 1
                new_line = data + str(new_label)
                outfile.write(new_line)


if __name__ == "__main__":
    main()
