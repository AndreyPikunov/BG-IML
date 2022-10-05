from pathlib import Path

import pandas as pd

import torch
from torch.nn.functional import one_hot
from torch.utils.data import Dataset
from torchvision.io import read_image

from tqdm import tqdm


class PaintingDataset(Dataset):
    # загрузим все картинки в память, потому что их мало (~60 Mb)

    def __init__(
        self,
        annotation: pd.DataFrame,
        folder_images: str,
        transform=None,
    ):
        self.ann = annotation.copy()
        self.folder_images = Path(folder_images)
        self.images = []
        self.Y = []
        self.transform = transform

        for _, row in tqdm(self.ann.iterrows(), total=len(self.ann)):
            label = row["label"]
            label_code = row["label_code"]
            filename_image = self.folder_images / label / row.filename
            image = read_image(str(filename_image))
            self.images.append(image)
            self.Y.append(label_code)

        self.Y = one_hot(torch.tensor(self.Y))
        self.N = len(self.Y)

    def __len__(self):
        return self.N

    def __getitem__(self, idx):
        x = self.images[idx]
        x_transformed = self.transform(x) if self.transform else x
        y = self.Y[idx]
        return x_transformed, y
