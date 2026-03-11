###        DATA formation###
import os
import nibabel as nib
import torch
import torch.nn.functional as F
from tqdm import tqdm
from PIL import Image
import numpy as np

# --- Settings ---
nii_folder = "./Data/PKG - CT-ORG/CT-ORG/OrganSegmentations"
output_hr_folder = "./Data/PKG_Slices/HR"
output_lr_folder = "./Data/PKG_Slices/LR"  # optional if you want LR images
target_shape = (512, 512, 512)  # (H, W, D)
scale = 8  # SR scale factor for LR images

os.makedirs(output_hr_folder+'Z', exist_ok=True)
os.makedirs(output_lr_folder+'Z', exist_ok=True)
os.makedirs(output_hr_folder+'X', exist_ok=True)
os.makedirs(output_lr_folder+'X', exist_ok=True)
os.makedirs(output_hr_folder+'Y', exist_ok=True)
os.makedirs(output_lr_folder+'Y', exist_ok=True)

# --- Loop over NIfTI files ---
nii_files = [f for f in os.listdir(nii_folder) if f.startswith('volume-') and f.endswith('.nii.gz')]

for nii_file in tqdm(nii_files, desc="Processing NIfTI files"):
    nii_path = os.path.join(nii_folder, nii_file)
    volume = nib.load(nii_path).get_fdata()  # shape: H x W x D
    
    # Normalize to 0-1
    volume = volume - volume.min()
    if volume.max() > 0:
        volume = volume / volume.max()
    
    # Convert to tensor: C x D x H x W for trilinear interpolation
    volume_tensor = torch.tensor(volume, dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # 1 x 1 x H x W x D
    volume_upsampled = F.interpolate(
        volume_tensor,
        size=target_shape,
        mode='trilinear',
        align_corners=False
    ).squeeze(0).squeeze(0)  # shape: H x W x D

    # Convert back to uint8 0-255 for saving
    volume_upsampled = (volume_upsampled.numpy() * 255).astype(np.uint8)
    
    # Slice along z-axis
    for z in range(0, 512):
        # Resize to target HR size
        slice_img = volume_upsampled[:, :, z]
        slice_hr = Image.fromarray(slice_img)
        slice_hr_filename = f"{nii_file.replace('.nii.gz','')}_slice{z:03d}_HR.png"
        slice_hr.save(os.path.join(output_hr_folder+'Z', slice_hr_filename))

        # Optional: save LR slice for SR training
        slice_lr = slice_hr.resize((target_shape[0]//scale, target_shape[1]//scale), Image.BICUBIC)
        slice_lr_filename = f"{nii_file.replace('.nii.gz','')}_slice{z:03d}_LR.png"
        slice_lr.save(os.path.join(output_lr_folder+'Z', slice_lr_filename))
        
        slice_img = volume_upsampled[z, :, :]
        slice_hr = Image.fromarray(slice_img)
        slice_hr_filename = f"{nii_file.replace('.nii.gz','')}_slice{z:03d}_HR.png"
        slice_hr.save(os.path.join(output_hr_folder+'X', slice_hr_filename))

        # Optional: save LR slice for SR training
        slice_lr = slice_hr.resize((target_shape[0]//scale, target_shape[1]//scale), Image.BICUBIC)
        slice_lr_filename = f"{nii_file.replace('.nii.gz','')}_slice{z:03d}_LR.png"
        slice_lr.save(os.path.join(output_lr_folder+'X', slice_lr_filename))
        
        slice_img = volume_upsampled[:, z, :]
        slice_hr = Image.fromarray(slice_img)
        slice_hr_filename = f"{nii_file.replace('.nii.gz','')}_slice{z:03d}_HR.png"
        slice_hr.save(os.path.join(output_hr_folder+'Y', slice_hr_filename))

        # Optional: save LR slice for SR training
        slice_lr = slice_hr.resize((target_shape[0]//scale, target_shape[1]//scale), Image.BICUBIC)
        slice_lr_filename = f"{nii_file.replace('.nii.gz','')}_slice{z:03d}_LR.png"
        slice_lr.save(os.path.join(output_lr_folder+'Y', slice_lr_filename))
        
        
        
###         this is for moving the files to the corresponding folder###        
# import os
# import shutil

# volume_start = 119                # Start of volume range
# volume_end = 139                  # End of volume range
# volume_start = 99                # Start of volume range
# volume_end = 118                  # End of volume range
# === Configuration ===
# for folder in ['HRZ','HRX','HRY','LRZ','LRX','LRY']:
#     source_dir = "./Data/PKG_Slices/Train/"+ folder             # Current directory to search
#     destination_dir = "./Data/PKG_Slices/Validate/"+folder # Destination folder

#     # Ensure destination folder exists
#     os.makedirs(destination_dir, exist_ok=True)
#     print("start:",folder)
#     # Walk through all files in source directory (non-recursive)
#     for filename in os.listdir(source_dir):
#         filepath = os.path.join(source_dir, filename)
#         # Skip if it's not a file
#         if not os.path.isfile(filepath):
#             continue

#         try:
#             # Open file and read the first line
#             with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
#                 first_line = f.readline().strip()

#             # Check if first line matches volume range
#             for vol in range(volume_start, volume_end + 1):
#                 if f"volume-{vol}_" in filename:
#                     # Move the file
#                     shutil.move(filepath, os.path.join(destination_dir, filename))
#                     # print(f"Moved: {filename}")
#                     break  # No need to check remaining volumes
#         except Exception as e:
#             print(f"Error processing {filename}: {e}")
# print('end')
