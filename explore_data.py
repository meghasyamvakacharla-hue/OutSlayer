import os
import cv2
import numpy as np

base_dir = r"c:\Users\megha\OneDrive\Desktop\OutSlayer\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train"
mask_dir = os.path.join(base_dir, "Segmentation")

mask_files = os.listdir(mask_dir)
if len(mask_files) > 0:
    mask_path = os.path.join(mask_dir, mask_files[0])
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Could not read mask with grayscale, try BGR: {mask_path}")
        mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
        if mask is not None:
            reshaped = mask.reshape(-1, mask.shape[-1])
            unique_colors = np.unique(reshaped, axis=0)
            print(f"Mask is color. Unique colors: \n{unique_colors}")
        else:
            print("Failed to read mask.")
    else:
        unique_vals = np.unique(mask)
        print(f"Grayscale mask unique values: {unique_vals}")
        print(f"Mask shape: {mask.shape}")
else:
    print("No masks found.")

