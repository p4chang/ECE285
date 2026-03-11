import os
import glob
import argparse
from multiprocessing import Pool
from os import path as osp
import sys
import numpy as np
from PIL import Image
from tqdm import tqdm
sys.path.append(os.path.abspath("./SRDiff_main"))
from utils.hparams import set_hparams, hparams
from utils.indexed_datasets import IndexedDatasetBuilder

def process_image(args):
    i, hr_path, lr_path, crop_size, thresh_size, sr_scale = args

    img_name = osp.splitext(osp.basename(hr_path))[0]

    img_hr = np.asarray(Image.open(hr_path).convert("RGB"))
    img_lr = np.asarray(Image.open(lr_path).convert("RGB"))

    h, w, _ = img_hr.shape
    h = h - h % sr_scale
    w = w - w % sr_scale

    img_hr = img_hr[:h, :w]
    img_lr = img_lr[:h // sr_scale, :w // sr_scale]

    results = []

    x = 0
    while x < h - thresh_size:
        y = 0
        while y < w - thresh_size:
            x_lr = x // sr_scale
            y_lr = y // sr_scale

            cropped_hr = img_hr[x:x + crop_size, y:y + crop_size]
            cropped_lr = img_lr[x_lr:x_lr + crop_size // sr_scale, y_lr:y_lr + crop_size // sr_scale]

            results.append({
                "item_name": img_name,
                "loc": [x // crop_size, y // crop_size],
                "loc_bdr": [(h + crop_size - 1) // crop_size, (w + crop_size - 1) // crop_size],
                "path": hr_path,
                "img": cropped_hr,
                "img_lr": cropped_lr,
            })

            y += crop_size
        x += crop_size

    return i, results


def build_dataset(hr_paths, lr_paths, binary_dir, prefix):
    os.makedirs(binary_dir, exist_ok=True)
    builder = IndexedDatasetBuilder(f"{binary_dir}/{prefix}")

    crop_size = hparams["crop_size"]
    thresh_size = hparams["thresh_size"]
    sr_scale = hparams["sr_scale"]

    worker_args = [
        (i, hr_paths[i], lr_paths[i], crop_size, thresh_size, sr_scale)
        for i in range(len(hr_paths))
    ]

    with Pool(processes=8) as pool:
        for idx, items in tqdm(pool.imap_unordered(process_image, worker_args), total=len(worker_args)):
            for item in items:
                builder.add_item(item)

    builder.finalize()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    set_hparams(args.config)

    # Get HR/LR paths separately
    hr_train = sorted(glob.glob(os.path.join(hparams["raw_data_dir"]["hr_path"], "Train", "*.*")))
    lr_train = sorted(glob.glob(os.path.join(hparams["raw_data_dir"]["lr_path"], "Train", "*.*")))

    hr_test = sorted(glob.glob(os.path.join(hparams["raw_data_dir"]["hr_path"], "Test", "*.*")))
    lr_test = sorted(glob.glob(os.path.join(hparams["raw_data_dir"]["lr_path"], "Test", "*.*")))

    hr_val = sorted(glob.glob(os.path.join(hparams["raw_data_dir"]["hr_path"], "Validate", "*.*")))
    lr_val = sorted(glob.glob(os.path.join(hparams["raw_data_dir"]["lr_path"], "Validate", "*.*")))

    # Ensure counts match
    assert len(hr_train) == len(lr_train), "Train HR/LR mismatch"
    assert len(hr_test) == len(lr_test), "Test HR/LR mismatch"
    assert len(hr_val) == len(lr_val), "Test HR/LR mismatch"

    # Build datasets
    print("Building train dataset...")
    build_dataset(hr_train, lr_train, hparams["binary_data_dir"], "train")

    print("Building test dataset...")
    build_dataset(hr_test, lr_test, hparams["binary_data_dir"], "test")

    print("Building valid dataset...")
    build_dataset(hr_val, lr_val, hparams["binary_data_dir"], "valid")
    
    print("Done.")