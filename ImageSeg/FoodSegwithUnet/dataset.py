import numpy as np
from torch.utils.data import Dataset

class FoodSegDataset(Dataset):
    def __init__(self, hf_dataset_split, transform=None):
        """
        Args:
            hf_dataset_split: A specific split from the Hugging Face dataset 
                              (e.g., dataset['train'] or dataset['validation']).
            transform: Albumentations transforms to apply.
        """
        self.hf_dataset = hf_dataset_split
        self.transform = transform

    def __len__(self):
        # Hugging Face datasets know their own length
        return len(self.hf_dataset)

    def __getitem__(self, index):
        # 1. Fetch the dictionary for this specific index from the Arrow file
        item = self.hf_dataset[index]
        
        # 2. Extract the image and mask. 
        # Hugging Face usually decodes these into PIL Images automatically.
        image = np.array(item['image'].convert("RGB"))
        
        # For multi-class, masks are 2D arrays of class integers (0 to 103).
        # We do NOT convert this to RGB or scale it to 0-1.
        mask = np.array(item['label']) 
        
        # 3. Apply Albumentations if you have them
        if self.transform is not None:
            augmentations = self.transform(image=image, mask=mask)
            image = augmentations['image']
            mask = augmentations['mask']
            
            # Albumentations sometimes changes the mask type, ensure it stays integer 
            # for CrossEntropyLoss later.
            mask = mask.long() if hasattr(mask, 'long') else mask 
            
        return image, mask