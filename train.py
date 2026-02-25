import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import OffroadDataset
from torchvision.models.segmentation import lraspp_mobilenet_v3_large
import time
import numpy as np

# Hyperparameters
batch_size = 8  # Reduced to avoid OOM
learning_rate = 1e-4 # Better lr
num_epochs = 10  # Explicitly train for 10 more epochs
num_classes = 6

# Paths
base_dir = r"c:\Users\megha\OneDrive\Desktop\OutSlayer\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset"
train_img_dir = os.path.join(base_dir, "train", "Color_Images")
train_mask_dir = os.path.join(base_dir, "train", "Segmentation")
val_img_dir = os.path.join(base_dir, "val", "Color_Images")
val_mask_dir = os.path.join(base_dir, "val", "Segmentation")

# Data Loaders
# num_workers > 0 causes hanging on Windows, setting to 0. 
train_dataset = OffroadDataset(train_img_dir, train_mask_dir, is_train=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)

val_dataset = OffroadDataset(val_img_dir, val_mask_dir, is_train=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

# Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# We'll use a lightweight LRASPP MobileNetV3 to train quickly on CPU/GPU
model = lraspp_mobilenet_v3_large(weights=None, weights_backbone=None, num_classes=num_classes)

if os.path.exists("best_model_continuous.pth"):
    print("Loading existing best_model_continuous.pth...")
    model.load_state_dict(torch.load("best_model_continuous.pth", map_location=torch.device('cpu')))
elif os.path.exists("best_model_early_stopped.pth"):
    print("Loading existing best_model_early_stopped.pth to continue 10 more epochs...")
    model.load_state_dict(torch.load("best_model_early_stopped.pth", map_location=torch.device('cpu')))

model = model.to(device)

def calculate_iou(preds, labels, num_classes):
    intersections = np.zeros(num_classes)
    unions = np.zeros(num_classes)
    for cls in range(num_classes):
        pred_inds = preds == cls
        target_inds = labels == cls
        intersection = (pred_inds[target_inds]).long().sum().item()
        union = pred_inds.long().sum().item() + target_inds.long().sum().item() - intersection
        intersections[cls] = float(intersection)
        unions[cls] = float(union)
    return intersections, unions

# Computed inverse frequency class weights to heavily penalize missing minority classes (e.g., Class 3)
class_weights = torch.tensor([1.76, 0.83, 2.29, 13.90, 0.68, 0.44], dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Cosine Annealing Scheduler to dynamically lower LR as training nears the 50-epoch end
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)

if __name__ == '__main__':
    # Training Loop
    print(f"Starting specific 10-epoch training on {len(train_dataset)} images...")
    best_miou = 0.5169 # starting off the previous best score
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        start_time = time.time()
        
        for i, (images, masks) in enumerate(train_loader):
            images = images.to(device)
            masks = masks.to(device)
            
            optimizer.zero_grad()
            
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            if (i+1) % 50 == 0:
                elapsed = time.time() - start_time
                print(f"Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(train_loader)}], Loss: {loss.item():.4f}, Time: {elapsed:.2f}s")
                start_time = time.time()
                
        print(f"Epoch [{epoch+1}/{num_epochs}] completed. Average Train Loss: {running_loss/len(train_loader):.4f}")
        
        # Validation loop
        model.eval()
        val_loss = 0.0
        total_intersections = np.zeros(num_classes)
        total_unions = np.zeros(num_classes)
        
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)['out']
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                
                _, preds = torch.max(outputs, 1)
                bs_inter, bs_union = calculate_iou(preds, masks, num_classes)
                total_intersections += bs_inter
                total_unions += bs_union
                
        # Calculate final IoU
        ious = []
        for cls in range(num_classes):
            if total_unions[cls] == 0:
                ious.append(float('nan'))
            else:
                ious.append(total_intersections[cls] / total_unions[cls])
        
        ious = np.array(ious)
        miou = np.nanmean(ious)
                
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
        print(f"Validation IoU per class: {ious}")
        print(f"Validation mIoU: {miou:.4f}")
        
        if miou > best_miou:
            print(f"New best mIoU ({best_miou:.4f} -> {miou:.4f}). Saving model...")
            best_miou = miou
            # Save the ultimate best model
            torch.save(model.state_dict(), "best_model_plus_10.pth")
            
        scheduler.step()
        print("-" * 30)
                
    print("Training finished.")
    print(f"Best validation mIoU: {best_miou:.4f}")
