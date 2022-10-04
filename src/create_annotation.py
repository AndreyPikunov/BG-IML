import logging
import argparse
from pathlib import Path

import pandas as pd
from torchvision.io import read_image
from tqdm.auto import tqdm

from utils import load_config, get_abs_dirname


def main(config):

    folder_images = config["annotation"]["folder_images"]
    folder_images = Path(folder_images)

    rows_annotation = []

    for folder_style in tqdm(folder_images.iterdir(), desc="styles"):

        style = folder_style.stem.lower()

        for filename_image in folder_style.iterdir():

            image_stem = filename_image.stem.lower()
            author, *title = image_stem.split("_", maxsplit=1)

            if not title:
                title = None
            else:
                title = title[0]

            if style in ("cartoon", "photo"):
                # no author for these styles
                author = None
                title = image_stem

            filename_image_posix = filename_image.as_posix()
            img = read_image(filename_image_posix)

            _, H, W = img.shape

            row = {
                "label": style,
                "author": author,
                "title": title,
                "filename": filename_image.name.lower(),
                "width": W,
                "height": H,
            }
            rows_annotation.append(row)

    annotation = pd.DataFrame(rows_annotation)

    filename_save = config["annotation"]["filename_annotation"]
    annotation.to_csv(filename_save, index=False)


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

    main(config)
