import os
import cv2
import numpy as np
from tqdm import tqdm

base_dir = r"c:\Users\megha\OneDrive\Desktop\OutSlayer\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train"
mask_dir = os.path.join(base_dir, "Segmentation")

mask_files = os.listdir(mask_dir)
all_unique_vals = set()

# Check first 500 masks to be reasonably sure
for f in tqdm(mask_files[:500]):
    mask_path = os.path.join(mask_dir, f)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is not None:
        unique_vals = np.unique(mask)
        all_unique_vals.update(unique_vals)

print(f"All unique mask values across sample: {sorted(list(all_unique_vals))}")
