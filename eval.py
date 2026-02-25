import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import OffroadDataset
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
import numpy as np
import os
import matplotlib.pyplot as plt

num_classes = 6
batch_size = 4

base_dir = r"c:\Users\megha\OneDrive\Desktop\OutSlayer\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset"
val_img_dir = os.path.join(base_dir, "val", "Color_Images")
val_mask_dir = os.path.join(base_dir, "val", "Segmentation")

val_dataset = OffroadDataset(val_img_dir, val_mask_dir)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=num_classes)
model.load_state_dict(torch.load("best_model_20_epochs.pth", map_location=device, weights_only=True))
model = model.to(device)
model.eval()

def calculate_iou(preds, labels, num_classes):
    intersections = np.zeros(num_classes)
    unions = np.zeros(num_classes)
    # preds: (B, H, W), labels: (B, H, W)
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        intersections[cls] = float(intersection)
        unions[cls] = float(union)
    return intersections, unions

total_intersections = np.zeros(num_classes)
total_unions = np.zeros(num_classes)
with torch.no_grad():
    for images, masks in val_loader:
        images = images.to(device)
        masks = masks.to(device)
        outputs = model(images)['out']
        # outputs shape: (B, C, H, W)
        _, preds = torch.max(outputs, 1)
        
        intersection, union = calculate_iou(preds, masks, num_classes)
        total_intersections += intersection
        total_unions += union

ious = []
for cls in range(num_classes):
    if total_unions[cls] == 0:
        ious.append(float('nan'))
    else:
        ious.append(total_intersections[cls] / total_unions[cls])

ious = np.array(ious)
print(f"IoU per class: {ious}")
print(f"Mean IoU: {np.nanmean(ious)}")

