import h5py
from sktime.datasets import load_from_tsfile_to_dataframe
import torch

def main():
    """
    One time script just to do any preprocessing we need to do.
    Code here will probably not be neat :)
    """

    # Filter out CharacterTrajectories to just "b", "d", "p", and "q"
    mapping = {":2": ":b", ":4": ":d", ":12": ":p", ":13": ":q"}
    with open("data/charactertrajectories/processed/CharacterTrajectoriesEq_TRAIN.ts", 'r') as infile:
        lines = infile.readlines()

    with open("data/charactertrajectories_filtered/processed/train.ts", 'w') as outfile:
        for i, line in enumerate(lines):
            if i < 7:
                outfile.write(line)
            else:
                stripped_line = line.strip()
                for old, new in mapping.items():
                    if stripped_line.endswith(old):
                        outfile.write(stripped_line[:-len(old)] + new + "\n")
                        break

    with open("data/charactertrajectories/processed/CharacterTrajectoriesEq_TEST.ts", 'r') as infile:
        lines = infile.readlines()

    with open("data/charactertrajectories_filtered/processed/test.ts", 'w') as outfile:
        for i, line in enumerate(lines):
            if i < 7:
                outfile.write(line)
            else:
                stripped_line = line.strip()
                for old, new in mapping.items():
                    if stripped_line.endswith(old):
                        outfile.write(stripped_line[:-len(old)] + new + "\n")
                        break




if __name__ == "__main__":
    main()
