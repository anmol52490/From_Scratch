import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
# from A.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset

from model import UNET

from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)



lr = 1e-4
device = 'cuda' if torch.cuda .is_available() else 'cpu'
batch = 2
epochs = 1
workers = 2
img_height = 256
img_width = 256
pin_mem = True
load_model = False
# train_img_dir =
# train_mask_dir =
# val_img_dir =
# val_mask_dir =


dataset = load_dataset("EduardoPacheco/FoodSeg103", cache_dir="./data/")

train_transform = A.Compose([
      A.Resize(height = img_height, width = img_width),
      A.Rotate(limit = 35, p = 1.0),
      A.HorizontalFlip(p = 0.5),
      A.VerticalFlip(p = 0.1),
      A.Normalize(
          mean = [0.0, 0.0, 0.0],
          std = [1.0, 1.0, 1.0],
          max_pixel_value = 255.0,
      ),
      ToTensorV2(),
  ])

val_transform = A.Compose([
    A.Resize(height = img_height, width = img_width),
    A.Normalize(
        mean = [0.0, 0.0, 0.0],
        std = [1.0, 1.0, 1.0],
        max_pixel_value = 255.0,
    ),
    ToTensorV2(),
])


def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(device=device)
        # CRITICAL: CrossEntropyLoss expects target shape (Batch, H, W) as integers
        targets = targets.long().to(device=device)

        # Forward
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():
  

  model = UNET(in_channels = 3, out_channels = 104).to(device)
  loss_fn = nn.CrossEntropyLoss()
  optimizer = optim.Adam(model.parameters(), lr = lr)

  
  

  train_loader, val_loader = get_loaders(
    #   train_img_dir,
    #   train_mask_dir,
    #   val_img_dir,
    #   val_mask_dir,
      dataset,
      batch,
      train_transform,
      val_transform,
  )

  if load_model:
        load_checkpoint(torch.load("my_checkpoint.pth.tar"), model)

#   check_accuracy(val_loader, model, device=device)
#   scaler = torch.cuda.amp.GradScaler()


  for epoch in range(epochs):
    train_fn(train_loader, model, optimizer, loss_fn)

    #save model

    checkpoint = {
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_checkpoint(checkpoint)


    #check_accuracy
    check_accuracy(val_loader, model, device = device)

    #print some examples
    # save_predictions_as_imgs(val_loader, model, folder = 'saved_images/', device = device)

if __name__ == '__main__':
  main()