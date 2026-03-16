import matplotlib.pyplot as plt
import numpy as np
import torch
import cv2
from model import UNET
from train import val_transform, device, dataset
from utils import load_checkpoint

# The official FoodSeg103 Color Palette (104 colors)
FOODSEG_PALETTE = np.array([
    [255, 255, 255], [40, 100, 150], [80, 150, 200], [120, 200, 10], [160, 10, 60], 
    [200, 60, 110], [0, 110, 160], [40, 160, 210], [80, 210, 20], [120, 20, 70], 
    [160, 70, 120], [200, 120, 170], [0, 170, 220], [40, 220, 30], [80, 30, 80], 
    [120, 80, 130], [160, 130, 180], [200, 180, 230], [0, 230, 40], [40, 40, 90], 
    [80, 90, 140], [120, 140, 190], [160, 190, 0], [200, 0, 50], [0, 50, 100], 
    [40, 100, 150], [80, 150, 200], [120, 200, 10], [160, 10, 60], [200, 60, 110], 
    [0, 110, 160], [40, 160, 210], [80, 210, 20], [120, 20, 70], [160, 70, 120], 
    [200, 120, 170], [0, 170, 220], [40, 220, 30], [80, 30, 80], [120, 80, 130], 
    [160, 130, 180], [200, 180, 230], [0, 230, 40], [40, 40, 90], [80, 90, 140], 
    [120, 140, 190], [160, 190, 0], [200, 0, 50], [0, 50, 100], [40, 100, 150], 
    [80, 150, 200], [120, 200, 10], [160, 10, 60], [200, 60, 110], [0, 110, 160], 
    [40, 160, 210], [80, 210, 20], [120, 20, 70], [160, 70, 120], [200, 120, 170], 
    [0, 170, 220], [40, 220, 30], [80, 30, 80], [120, 80, 130], [160, 130, 180], 
    [200, 180, 230], [0, 230, 40], [40, 40, 90], [80, 90, 140], [120, 140, 190], 
    [160, 190, 0], [200, 0, 50], [0, 50, 100], [40, 100, 150], [80, 150, 200], 
    [120, 200, 10], [160, 10, 60], [200, 60, 110], [0, 110, 160], [40, 160, 210], 
    [80, 210, 20], [120, 20, 70], [160, 70, 120], [200, 120, 170], [0, 170, 220], 
    [40, 220, 30], [80, 30, 80], [120, 80, 130], [160, 130, 180], [200, 180, 230], 
    [0, 230, 40], [40, 40, 90], [80, 90, 140], [120, 140, 190], [160, 190, 0], 
    [200, 0, 50], [0, 50, 100], [40, 100, 150], [80, 150, 200], [120, 200, 10], 
    [160, 10, 60], [200, 60, 110], [0, 110, 160], [40, 160, 210]
], dtype=np.uint8)

def decode_segmentation_mask(mask, palette):
    """Maps a 2D array of class integers to an RGB image."""
    rgb_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    for class_id in range(len(palette)):
        rgb_image[mask == class_id] = palette[class_id]
    return rgb_image

def visualize_prediction(model, dataset_split, image_index, device="cuda"):
    model.eval()
    
    # 1. Grab raw image and mask from the dataset
    item = dataset_split[image_index]
    raw_image = np.array(item['image'].convert("RGB"))
    raw_mask = np.array(item['label'])
    
    # 2. Apply validation transforms (to match how the model was trained)
    augmentations = val_transform(image=raw_image, mask=raw_mask)
    input_tensor = augmentations['image'].unsqueeze(0).to(device) # Add batch dimension
    true_mask = augmentations['mask'].numpy()
    
    # 3. Model Inference
    with torch.no_grad():
        with torch.amp.autocast(device_type='cuda'):
            logits = model(input_tensor)
            # Find highest probability class for each pixel
            pred_mask = torch.argmax(logits, dim=1).squeeze(0).cpu().numpy()
            
    # 4. Convert Masks to RGB
    true_mask_rgb = decode_segmentation_mask(true_mask, FOODSEG_PALETTE)
    pred_mask_rgb = decode_segmentation_mask(pred_mask, FOODSEG_PALETTE)
    
    # Format original image for plotting (un-normalize it if necessary)
    input_img_vis = input_tensor.squeeze(0).cpu().numpy().transpose(1, 2, 0)
    # Simple clip to valid range for visualization if you used Normalize(mean=0, std=1)
    input_img_vis = np.clip(input_img_vis, 0, 1) 
    
    # 5. Plotting
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    axes[0].imshow(input_img_vis)
    axes[0].set_title("Original Image (Resized)")
    axes[0].axis("off")
    
    axes[1].imshow(true_mask_rgb)
    axes[1].set_title("Ground Truth Mask")
    axes[1].axis("off")
    
    axes[2].imshow(pred_mask_rgb)
    axes[2].set_title("U-Net Prediction")
    axes[2].axis("off")
    
    plt.tight_layout()
    plt.show()

# --- LOAD YOUR MODEL AND RUN ---

# REPLACE THIS with your actual saved model file name!
checkpoint_path = r"C:\Users\boys5\OneDrive\Desktop\Research\codebase\from_scratch\ImageSeg\FoodSegwithUnet\saved_models\DiceCELossmodel_MIoU_256x256_epoch5_loss2.82_mIoU3.00.pth.tar" 
# checkpoint_path = 'my_checkpoint.pth.tar'
# Initialize a fresh model and load the weights
inference_model = UNET(in_channels=3, out_channels=104).to(device)
load_checkpoint(torch.load(checkpoint_path, map_location=device), inference_model)

# Grab a random index from the validation set (e.g., image #15)
image_idx_to_test = 15

# Visualize it!
print(f"Visualizing validation image #{image_idx_to_test}...")
visualize_prediction(inference_model, dataset['validation'], image_idx_to_test, device)