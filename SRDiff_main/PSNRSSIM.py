import os
import torch
import nibabel as nib
import numpy as np
import torch.nn.functional as F
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

closest_avg = "./checkpoints/combo/closest_avg"
the_max = "./checkpoints/combo/max"
the_mean = "./checkpoints/combo/mean"
the_median = "./checkpoints/combo/median"
original = "./Data/PKG - CT-ORG/CT-ORG/OrganSegmentations"
device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)

psnr_metric = PeakSignalNoiseRatio().to(device)
ssim_metric = StructuralSimilarityIndexMeasure().to(device)
overall_metrics = {
    "closest_avg": {"mse": [], "psnr": [], "ssim": []},
    "max": {"mse": [], "psnr": [], "ssim": []},
    "mean": {"mse": [], "psnr": [], "ssim": []},
    "median": {"mse": [], "psnr": [], "ssim": []}
}
def load_nifti_tensor(path):
    nii = nib.load(path)
    data = nii.get_fdata()
    t = torch.from_numpy(data).float()
    t = t.unsqueeze(0).unsqueeze(0)
    return t.to(device)

files = sorted([f for f in os.listdir(the_mean) if f.endswith(".nii.gz")])

for fname in files:

    mean_path = os.path.join(the_mean, fname)
    orig_path = os.path.join(original, fname.replace("_", "-"))

    avg_path = os.path.join(closest_avg, fname)
    max_path = os.path.join(the_max, fname)
    median_path = os.path.join(the_median, fname)

    if not os.path.exists(orig_path):
        print("Missing original:", fname)
        continue

    gt = load_nifti_tensor(orig_path)

    preds = {
        "closest_avg": avg_path,
        "max": max_path,
        "mean": mean_path,
        "median": median_path
    }

    for name, path in preds.items():

        if not os.path.exists(path):
            print(f"Missing {name}: {fname}")
            continue

        pred = load_nifti_tensor(path)
        # print("pred max:", torch.max(pred))
        # print("gt max:", torch.max(gt))
        # print("pred min:", torch.min(pred))
        # print("gt min:",torch.min(gt))
        # print("pred:", pred.shape)
        # print("gt:", gt.shape)
        pred = F.interpolate(pred, size=gt.shape[2:], mode='trilinear', align_corners=False)
        pred= (pred-pred.min()) / (pred.max()-pred.min())
        gt = (gt-gt.min()) / (gt.max()-gt.min())
        
        with torch.no_grad():
            mse = F.mse_loss(pred, gt)
            psnr = psnr_metric(pred, gt)
            ssim = ssim_metric(pred, gt)
        overall_metrics[name]["mse"].append(mse.item())
        overall_metrics[name]["psnr"].append(psnr.item())
        overall_metrics[name]["ssim"].append(ssim.item())

        print(
            f"{fname} | {name} | "
            f"MSE: {mse.item():.6f} | "
            f"PSNR: {psnr.item():.4f} | "
            f"SSIM: {ssim.item():.4f}"
        )

    psnr_metric.reset()
    ssim_metric.reset()
    
print("\nOverall Metrics Across All Files:")
for name, metrics in overall_metrics.items():
    mse_avg = sum(metrics["mse"]) / len(metrics["mse"]) if metrics["mse"] else float('nan')
    psnr_avg = sum(metrics["psnr"]) / len(metrics["psnr"]) if metrics["psnr"] else float('nan')
    ssim_avg = sum(metrics["ssim"]) / len(metrics["ssim"]) if metrics["ssim"] else float('nan')
    print(f"{name} | MSE: {mse_avg:.6f} | PSNR: {psnr_avg:.4f} | SSIM: {ssim_avg:.4f}")