import logging
import argparse

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split

from utils import load_config, get_abs_dirname


def main(config):

    filename_annotation = config["design"]["filename_annotation"]
    ann = pd.read_csv(filename_annotation)

    random_state = config["shared"]["seed"]
    kfold_splits = config["design"]["kfold_splits"]

    kfold = KFold(n_splits=kfold_splits, shuffle=True, random_state=random_state)

    val_size = config["design"]["val_size"]

    ann["fold_author"] = None

    for label in ann.label.unique():

        mask_label = ann.label == label
        authors = ann.loc[mask_label, "author"]

        if authors.isna().all():

            indices_train_test, indices_val = train_test_split(
                authors.index, test_size=val_size, random_state=random_state
            )
            ann.loc[indices_val, "fold_author"] = "val"

            msg = label + "\n"

            p = len(indices_val) / len(authors.index)
            msg = msg + f"no autors: {p:.2f} %"

            logging.info(msg)

            splits = kfold.split(indices_train_test)
            for i, (indices_train, indices_test) in enumerate(splits):
                indices_test_global = indices_train_test[indices_test]
                ann.loc[indices_test_global, "fold_author"] = i

        else:

            authors_counts = authors.value_counts(ascending=True)
            authors_cumsum = authors_counts.cumsum()
            authors_cumsum_normalized = authors_cumsum / authors_cumsum.iloc[-1]

            mask_val = authors_cumsum_normalized < val_size
            authors_val = authors_cumsum_normalized[mask_val].index
            authors_train_test = authors_cumsum_normalized[~mask_val].index

            ann.loc[mask_label & ann.author.isin(authors_val), "fold_author"] = "val"

            msg = label + "\n"

            n_authors_val = len(authors_val)
            n_authors = len(authors.unique())

            percent_authors_val = n_authors_val / n_authors * 100

            msg = msg + f"{n_authors_val} / {n_authors} = {percent_authors_val:.2f} %"
            logging.info(msg)

            best_split = None
            best_std = 100
            n_trials = 100

            for i_trial in range(n_trials):

                test_sizes = []

                shuffler = KFold(kfold_splits, random_state=i_trial, shuffle=True)

                for i, (indices_train, indices_test) in enumerate(
                    shuffler.split(authors_train_test)
                ):
                    autors_test = authors_train_test[indices_test]
                    mask_test = mask_label & ann.author.isin(autors_test)
                    ann.loc[mask_test, "fold_author"] = i
                    test_sizes.append(sum(mask_test))

                std = np.std(test_sizes)
                if std < best_std:
                    msg = f"new best: {std} {test_sizes} {sum(test_sizes)}"
                    logging.info(msg)
                    best_std = std
                    best_split = ann.loc[mask_label, "fold_author"].copy()

            ann.loc[mask_label, "fold_author"] = best_split
            assert ~ann.loc[mask_label, "fold_author"].isna().any()

    filename_save = config["design"]["filename_save"]
    ann.to_csv(filename_save, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str)
    parser.add_argument("--logging-level", type=str, default="WARNING")
    args = parser.parse_args()

    filename_config = args.config

    numeric_level = getattr(logging, args.logging_level.upper(), None)
    logging.basicConfig(level=numeric_level)

    config = load_config(filename_config)

    project_root = get_abs_dirname(filename_config)
    config["shared"]["project_root"] = project_root

    np.random.seed(config["shared"]["seed"])

    main(config)
