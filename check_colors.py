import os
import cv2
import numpy as np
from tqdm import tqdm

d_img = r'c:\Users\megha\OneDrive\Desktop\OutSlayer\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val\Color_Images'
d_mask = r'c:\Users\megha\OneDrive\Desktop\OutSlayer\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\val\Segmentation'

color_sums = np.zeros((6, 3))
counts = np.zeros(6)
cls_map = {0: 0, 1: 1, 2: 2, 3: 3, 27: 4, 39: 5}

files = os.listdir(d_mask)[:10]

for f in tqdm(files):
    mask = cv2.imread(os.path.join(d_mask, f), 0)
    img = cv2.imread(os.path.join(d_img, f))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # Ensure correct color channel order (RGB)
    
    for v, cls in cls_map.items():
        pixels = img[mask == v]
        counts[cls] += len(pixels)
        if len(pixels) > 0:
            color_sums[cls] += np.sum(pixels, axis=0)

averages = color_sums / np.maximum(counts[:, None], 1)
for i, avg in enumerate(averages):
    print(f"Class {i}: Average RGB [R: {int(avg[0])}, G: {int(avg[1])}, B: {int(avg[2])}] | Samples: {counts[i]}")
