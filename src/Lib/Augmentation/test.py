# Imports
import torch
import torchvision
import matplotlib.pyplot as plt

import numpy as np
from torchvision.utils import make_grid

# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_transforms.py

def show_dataset(dataset, n=6):
    imgs = torch.stack([dataset[i][0] for _ in range(n)
                       for i in range(len(dataset))])
    grid = make_grid(imgs).numpy()
    plt.imshow(np.transpose(grid, (1, 2, 0)), interpolation='nearest')
    plt.axis('off')

# Load Data
tsf = torchvision.transforms.Compose(
    [
       torchvision.transforms.Resize((640, 480)),  # Resizes (32,32) to (36,36)
    ]
)

dataset = torchvision.datasets.ImageFolder(root='/Users/rparak/Desktop/Dataset_Type_0_Obj_ID_0/', transform=tsf)

#show_dataset(dataset)