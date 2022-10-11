from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision.io import read_image

from tqdm import tqdm


class PaintingDatasetCombo(Dataset):
    def __init__(
        self,
        annotation: pd.DataFrame,
        folder_images: str,
        transform_train=None,
        transform_preprocess=None,
    ):
        self.ann = annotation.copy()
        self.folder_images = Path(folder_images)
        self.transform_train = transform_train

        self.images = []
        self.Y = []
        self.labels = []

        for _, row in tqdm(self.ann.iterrows(), total=len(self.ann)):
            label = row["label"]
            label_code = row["label_code"]
            assert isinstance(label_code, int)

            filename_image = self.folder_images / label / row.filename
            image = read_image(str(filename_image))

            if transform_preprocess:
                image = transform_preprocess(image)

            self.images.append(image)
            self.Y.append(label_code)
            self.labels.append(label)

        self.N = len(self.Y)
        self.indices = np.arange(self.N)
        self.labels = np.array(self.labels)
        self.codes = np.array(self.Y)
        self.Y = one_hot(torch.from_numpy(self.codes)).float()

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        # https://www.kaggle.com/code/hirotaka0122/triplet-loss-with-pytorch?scriptVersionId=26699660&cellId=6

        anchor = self.images[idx]
        anchor_code = self.codes[idx]
        y_a = self.Y[idx]

        mask_positive = (self.codes == anchor_code) & (self.indices != idx)
        mask_negative = (self.codes != anchor_code) & (self.indices != idx)

        indices_positive = self.indices[mask_positive]
        idx_positive = np.random.choice(indices_positive)
        positive = self.images[idx_positive]
        # positive_code = self.codes[idx_positive]
        y_p = self.Y[idx_positive]

        indices_negative = self.indices[mask_negative]
        idx_negative = np.random.choice(indices_negative)
        negative = self.images[idx_negative]
        # negative_code = self.codes[idx_negative]
        y_n = self.Y[idx_negative]

        t = self.transform_train

        xs = []
        ys = [y_a, y_p, y_n]

        for x in anchor, positive, negative:
            x_transformed = t(x) if t else x
            xs.append(x_transformed)

        return xs, ys
