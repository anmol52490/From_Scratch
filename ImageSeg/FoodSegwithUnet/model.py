import torch.nn as nn
import torch
import torch.nn as nn
import torchvision.transforms.functional as T


class DoubleConv(nn.Module):
  def __init__(self, in_channels, out_channels):
    super(DoubleConv, self).__init__()
    self.conv = nn.Sequential(
    nn.Conv2d(in_channels, out_channels, 3, 1, 1, bias = False),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace = True),
    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias = False),
    nn.BatchNorm2d(out_channels),
    nn.ReLU(inplace=True),
    )


  def forward(self, x):
    # print(x.shape)
    x = self.conv(x)
    # print(x.shape)
    return x
  


class UNET(nn.Module):
  def __init__(self, in_channels = 3, out_channels = 104, features = [64, 128, 256, 512]):
    super().__init__()
    self.ups = nn.ModuleList()
    self.downs = nn.ModuleList()
    self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)

    #down part of UNET
    for feature in features:
      self.downs.append(DoubleConv(in_channels, feature))
      in_channels = feature


    #Up part
    for feature in reversed(features):
      self.ups.append(
          nn.ConvTranspose2d(
              feature*2,
              feature,
              2,
              2,
          )
      )

      self.ups.append(DoubleConv(feature*2, feature))


    self.bottleneck = DoubleConv(features[-1], features[-1]*2)

    self.final_conv = nn.Conv2d(features[0], out_channels, 1,1)

  def forward(self, x):
    skip_connections = []
    # print("\noriginal image\n")
    # print(x.shape)

    # print("\nnow into down\n")
    for down in self.downs:
      x = down(x)
      skip_connections.append(x)
      x = self.pool(x)
    #   print(x.shape)

    x = self.bottleneck(x)
    # print('\nin bottleneck\n')
    # print(x.shape)
    # print("\nup\n")
    skip_connections = skip_connections[::-1]

    for idx in range(0, len(self.ups), 2):
    #   print(x.shape)
      x = self.ups[idx](x)
      skip_connection = skip_connections[idx//2]

      if x.shape != skip_connection.shape:
        x = T.resize(x, size = skip_connection.shape[2:])
      concat_skip = torch.cat((skip_connection, x), dim = 1)
      x = self.ups[idx+1](concat_skip)

    # print('\n output \n')
    x = self.final_conv(x)
    # print(x.shape)

    return x