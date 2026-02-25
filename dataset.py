import os
from PIL import Image
import numpy as np
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

class OffroadDataset(Dataset):
    def __init__(self, images_dir, masks_dir, is_train=False):
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.is_train = is_train
        self.images = sorted(os.listdir(images_dir))
        
        # Mapping mask pixel values to class indices 0-5
        self.class_mapping = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            27: 4,
            39: 5
        }
        
    def __len__(self):
        return len(self.images)
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx]) # assuming mask names match image names
        
        image = Image.open(img_path).convert("RGB")
        import cv2
        mask_cv = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = Image.fromarray(mask_cv)
        
        image = TF.resize(image, (256, 256))
        mask = TF.resize(mask, (256, 256), interpolation=Image.NEAREST)
        
        image = TF.to_tensor(image)
        mask = torch.tensor(np.array(mask), dtype=torch.long)
        
        # Apply class mapping
        mapped_mask = torch.zeros_like(mask)
        for original_val, mapped_idx in self.class_mapping.items():
            mapped_mask[mask == original_val] = mapped_idx
            
        # Data augmentation (only for training)
        if self.is_train:
            import torchvision.transforms as T
            
            # Color Jitter (only affects image, not mask)
            if torch.rand(1).item() > 0.5:
                jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)
                image = jitter(image)
                
            # Random Affine (Rotation and Scale) affects both identically
            if torch.rand(1).item() > 0.5:
                angle = float(torch.empty(1).uniform_(-15.0, 15.0).item())
                scale = float(torch.empty(1).uniform_(0.8, 1.2).item())
                
                # Apply identical transformation to both
                image = TF.affine(image, angle=angle, translate=(0,0), scale=scale, shear=0, interpolation=TF.InterpolationMode.BILINEAR)
                
                # Expand mask to 3D for affine, then squeeze
                mapped_mask = mapped_mask.unsqueeze(0).float()
                mapped_mask = TF.affine(mapped_mask, angle=angle, translate=(0,0), scale=scale, shear=0, interpolation=TF.InterpolationMode.NEAREST)
                mapped_mask = mapped_mask.squeeze(0).long()
                
            # Random Horizontal Flip
            if torch.rand(1).item() > 0.5:
                image = TF.hflip(image)
                mapped_mask = TF.hflip(mapped_mask)
            
        return image, mapped_mask

if __name__ == "__main__":
    train_img_dir = r"c:\Users\megha\OneDrive\Desktop\OutSlayer\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Color_Images"
    train_mask_dir = r"c:\Users\megha\OneDrive\Desktop\OutSlayer\Offroad_Segmentation_Training_Dataset\Offroad_Segmentation_Training_Dataset\train\Segmentation"
    
    dataset = OffroadDataset(train_img_dir, train_mask_dir, is_train=True)
    img, mask = dataset[0]
    print(f"Image shape: {img.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Unique mask values: {torch.unique(mask)}")
