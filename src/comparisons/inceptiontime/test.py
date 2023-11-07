import argparse

import matplotlib.pyplot as plt
import torch
from tsai.all import *
import sklearn.metrics as skm

from src.data.data import get_ds
from src.utils.utils import *

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class_to_index = {}
for i in range(64):
    class_to_index[str(i)] = i

def main(args):
    accuracies = []
    for i in range(args.resamples):
        train_ds=get_ds(get_train_path_from_dataset("simulated_640"), class_to_index)
        val_ds=get_ds(get_val_path_from_dataset("simulated_640"), class_to_index)
        test_ds=get_ds(get_test_path_from_dataset("simulated_640"), class_to_index)

        train_dataloader = torch.utils.data.DataLoader(train_ds, len(train_ds), shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_ds, len(val_ds), shuffle=True)
        test_dataloader = torch.utils.data.DataLoader(test_ds, len(test_ds), shuffle=False)

        for X_train, y_train in train_dataloader:
            for X_val, y_val in val_dataloader:
                X_train, y_train = X_train.to(device), y_train.to(device)
                X_val, y_val = X_val.to(device), y_val.to(device)
                X_train = X_train.permute(0, 2, 1)
                X_val = X_val.permute(0, 2, 1)
                X, y, splits = combine_split_data([X_train, X_val], [y_train, y_val])
                X, y = X.detach().numpy(), y.detach().numpy()

                tfms  = [None, [Categorize()]]
                dsets = TSDatasets(X, y, tfms=tfms, splits=splits, inplace=True)
                dls = TSDataLoaders.from_dsets(dsets.train, dsets.valid, bs=[64, 128], batch_tfms=[TSStandardize()], num_workers=0)
                model = InceptionTime(dls.vars, dls.c)
                learn = Learner(dls, model, metrics=accuracy)

                learn.fit_one_cycle(40, lr_max=1e-3)

                valid_dl = dls.valid
                for X_test, y_test in test_dataloader:
                    X_test = X_test.permute(0, 2, 1)
                    X_test, y_test = X_test.detach().numpy(), y_test.detach().numpy()
                    test_ds_tsai = valid_dl.dataset.add_test(X_test, y_test)
                    test_dl_tsai = valid_dl.new(test_ds_tsai)
                    test_probas, test_targets, test_preds = learn.get_preds(dl=test_dl_tsai, with_decoded=True, save_preds=None, save_targs=None)

                    test_accuracy = skm.accuracy_score(test_targets, test_preds)
                    print(test_accuracy)
                    accuracies.append(test_accuracy)

    accuracies = torch.Tensor(accuracies)
    print(accuracies.mean(), accuracies.std())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--resamples", type=int, help="Number of resamples")

    args = parser.parse_args()
    main(args)