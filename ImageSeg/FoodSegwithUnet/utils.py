from ast import Num
import torchvision
import torch
from torch.utils.data import DataLoader
from dataset import FoodSegDataset

def save_checkpoint(state, filename='my_checkpoint.pth.tar'):
  print('=> Saving checkpoint')
  torch.save(state, filename)

def load_checkpoint(checkpoint, model):
  print('=> Loading checkpoint')
  model.load_state_dict(checkpoint['state_dict'])


def check_accuracy(loader, model, device = 'cuda'):
  num_correct = 0
  num_pixels = 0
  dice_score = 0
  model.eval()

  with torch.no_grad():
    for x, y in loader:
      x = x.to(device)
      y = y.to(device).unsqueeze(1)
      preds = torch.sigmoid(model(x))
      preds = (preds>0.5).float()
      num_correct += (preds == y).sum()
      num_pixels += torch.numel(preds)
      dice_score += (2*(preds*y).sum())/((preds +y).sum() + 1e-8)

  print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
  print(f"Dice score: {dice_score/len(loader)}")
  model.train()




def get_loaders(
                dataset,
                batch_size,
                train_transform,
                val_transform,
                num_workers=4,
                pin_memory=True):

  train_ds = FoodSegDataset(
    hf_dataset_split=dataset['train'], 
    transform=train_transform
)

  train_loader = DataLoader(train_ds,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            pin_memory=pin_memory,
                            shuffle=True,
                            )

  val_ds = FoodSegDataset(
    hf_dataset_split=dataset['validation'], 
    transform=val_transform
)

  val_loader = DataLoader(val_ds,
                          batch_size=batch_size,
                          num_workers=num_workers,
                          pin_memory=pin_memory,
                          shuffle=False,
                          )

  return train_loader, val_loader



def check_accuracy(loader, model, device='cpu'):
    num_correct = 0
    num_pixels = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device) # No unsqueeze needed for multi-class

            preds = model(x) # Output shape: (Batch, 104, H, W)
            
            # Find the class index with the highest probability for each pixel
            preds = torch.argmax(preds, dim=1) 
            
            num_correct += (preds == y).sum()
            num_pixels += torch.numel(preds)

    print(f"Got {num_correct}/{num_pixels} with acc {num_correct/num_pixels*100:.2f}")
    model.train()


def save_predictions_as_imgs(
    loader, model, folder="saved_images/", device="cuda"
):
  model.eval()
  for idex, (x, y) in enumerate(loader):
    x = x.to(device)
    with torch.no_grad():
      preds = torch.sigmoid(model(x))
      preds = (preds > 0.5).float()

    torchvision.utils.save_image(
        preds, f"{folder}/pred_{idex}.png"
    )
    torchvision.utils.save_image(y.unsqueeze(1), f"{folder}{idex}"
    )