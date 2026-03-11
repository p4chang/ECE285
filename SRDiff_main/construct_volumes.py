import os
import re
import torch
import numpy as np
from PIL import Image
from collections import defaultdict
from tqdm import tqdm
import nibabel as nib

# -----------------------------
# CONFIG
# -----------------------------
IMG_SIZE = 512
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.uint8  # or torch.float32


# -----------------------------
# FILENAME PARSER
# -----------------------------
def parse_filename(filename):
    match = re.search(r"volume-(\d+)_slice(\d+)", filename)
    if not match:
        raise ValueError(f"Invalid filename: {filename}")
    return int(match.group(1)), int(match.group(2))


# -----------------------------
# LOAD IMAGE to TORCH
# -----------------------------
def load_image(path):
    img = Image.open(path).convert("L")
    arr = np.array(img, dtype=np.uint8)
    return torch.from_numpy(arr)  # stays on CPU initially


# -----------------------------
# ORGANIZE FILES
# -----------------------------
def collect_files(folder):
    data = defaultdict(list)

    for fname in os.listdir(folder):
        if not fname.endswith(".png"):
            continue

        vol_id, slice_idx = parse_filename(fname)
        data[vol_id].append((slice_idx, os.path.join(folder, fname)))

    for vol_id in data:
        data[vol_id].sort(key=lambda x: x[0])

    return data


# -----------------------------
# BUILD STACK (CPU to GPU)
# -----------------------------
def build_stack(file_list, axis):
    stack = torch.zeros((IMG_SIZE, IMG_SIZE, IMG_SIZE), dtype=DTYPE)

    for idx, path in file_list:
        img = load_image(path)

        if axis == 'x':
            stack[idx, :, :] = img
        elif axis == 'y':
            stack[:, idx, :] = img
        elif axis == 'z':
            stack[:, :, idx] = img

    return stack.to(DEVICE)  # move entire stack to GPU


def fuse_max(X, Y, Z):
    return torch.maximum(torch.maximum(X, Y), Z)


def fuse_closest_average(X, Y, Z):
    Xf = X.float()
    Yf = Y.float()
    Zf = Z.float()

    dXY = torch.abs(Xf - Yf)
    dXZ = torch.abs(Xf - Zf)
    dYZ = torch.abs(Yf - Zf)

    # Stack differences → shape (3, 512, 512, 512)
    diffs = torch.stack([dXY, dXZ, dYZ], dim=0)

    # Find index of minimum difference
    min_idx = torch.argmin(diffs, dim=0)

    # Compute candidate averages
    avg_XY = (Xf + Yf) / 2
    avg_XZ = (Xf + Zf) / 2
    avg_YZ = (Yf + Zf) / 2

    avgs = torch.stack([avg_XY, avg_XZ, avg_YZ], dim=0)

    # Select correct average per voxel
    V = torch.gather(avgs, 0, min_idx.unsqueeze(0)).squeeze(0)

    return V


# -----------------------------
# RECONSTRUCTION (GPU)
# -----------------------------
def reconstruct_volume(x_files, y_files, z_files, method="median"):
    X = build_stack(x_files, 'x')
    Y = build_stack(y_files, 'y')
    Z = build_stack(z_files, 'z')

    if method == "mean":
        V = (X.float() + Y.float() + Z.float()) / 3.0

    elif method == "median":
        stacked = torch.stack([X, Y, Z], dim=0)
        V = torch.median(stacked, dim=0).values
        
    elif method == "max":
        V = fuse_max(X, Y, Z)

    elif method == "closest_avg":
        V = fuse_closest_average(X, Y, Z)

    else:
        raise ValueError("Unknown method")

    return V.to(torch.uint8).cpu()  # bring back to CPU for saving
def volume(x_files):
    X = build_stack(x_files, 'x')
    return X.to(torch.uint8).cpu()  # bring back to CPU for saving


# -----------------------------
# MAIN
# -----------------------------
def process_all_volumes(x_dir, y_dir, z_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    X_data = collect_files(x_dir)
    Y_data = collect_files(y_dir)
    Z_data = collect_files(z_dir)

    volumes = set(X_data) & set(Y_data) & set(Z_data)

    print(f"Using device: {DEVICE}")
    print(f"Found {len(volumes)} volumes")

    for vol_id in tqdm(sorted(volumes)):
        x_files = X_data[vol_id]
        y_files = Y_data[vol_id]
        z_files = Z_data[vol_id]

        if not (len(x_files) == IMG_SIZE):
            continue

        V = reconstruct_volume(x_files, y_files, z_files,"closest_avg")
        img = nib.load(f"./Data/PKG - CT-ORG/CT-ORG/OrganSegmentations/volume-{vol_id}.nii.gz")
        affine = img.affine
        # affine = np.eye(4)
        # affine = np.array([
        #     [1, 0, 0, -256],
        #     [0, 1, 0, -256],
        #     [0, 0, 1, -256],
        #     [0, 0, 0, 1]
        # ]) #if need to center
        if V.is_cuda:
            numpy_data = V.cpu().numpy()
        else:
            numpy_data = V.numpy()
        nifti_img = nib.Nifti1Image(numpy_data, affine)
        nib.save(nifti_img, os.path.join(output_dir, f"volume_{vol_id}.nii.gz"))
def process_volumes(x_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    X_data = collect_files(x_dir)
    volumes = set(X_data)

    print(f"Using device: {DEVICE}")
    print(f"Found {len(volumes)} volumes")

    for vol_id in tqdm(sorted(volumes)):
        x_files = X_data[vol_id]

        if not (len(x_files) == IMG_SIZE):
            continue

        V = volume(x_files)
        img = nib.load(f"./Data/PKG - CT-ORG/CT-ORG/OrganSegmentations/volume-{vol_id}.nii.gz")
        affine = img.affine
        if V.is_cuda:
            numpy_data = V.cpu().numpy()
        else:
            numpy_data = V.numpy()
        nifti_img = nib.Nifti1Image(numpy_data, affine)
        nib.save(nifti_img, os.path.join(output_dir, f"volume_{vol_id}.nii.gz"))

# -----------------------------
# ENTRY
# -----------------------------
if __name__ == "__main__":
    # x_dir = "./checkpoints/rrdb_ctx/results_100000_/SR"
    # y_dir = "./checkpoints/rrdb_cty/results_100000_/SR"
    z_dir = "./checkpoints/rrdb_ctz/results_100000_/SR"
    output_dir = "./checkpoints/combo/z"
    process_volumes(z_dir,output_dir)
    # process_all_volumes(x_dir, y_dir, z_dir, output_dir)
