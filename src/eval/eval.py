import argparse

import torch

from src.data.data import get_ds
from src.utils.utils import get_config_from_dataset, load_multivariable_prototypes, get_test_path_from_dataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main(args):
    config = get_config_from_dataset(args.dataset)
    multivariable_module = load_multivariable_prototypes(config).to(device)
    multivariable_module.eval()

    ds = get_ds(get_test_path_from_dataset(args.dataset, train=False), config['class_to_index'])
    dl = torch.utils.data.DataLoader(ds, len(ds), False, pin_memory=True)
    with torch.no_grad():
        numerator = 0
        denominator = 0
        for data_matrix, labels in dl:
            data_matrix, labels = data_matrix.to(device), labels.to(device)
            output, _ = multivariable_module(data_matrix.float())
            sof = torch.softmax(output, 1)
            prediction = torch.argmax(sof, 1)

            numerator += torch.sum(prediction.eq(labels).int())
            denominator += data_matrix.shape[0]
        accuracy = float(numerator) / float(denominator)
        print("Accuracy: " + str(accuracy))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, help="Name of the dataset (e.g. <basicmotions>)")

    args = parser.parse_args()
    main(args)