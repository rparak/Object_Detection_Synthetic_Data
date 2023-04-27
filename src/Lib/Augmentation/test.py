# Imports
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

# https://pytorch.org/vision/0.11/auto_examples/plot_transforms.html
# https://pseudo-lab.github.io/Tutorial-Book-en/chapters/en/object-detection/Ch3-preprocessing.html
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_transforms.py

RAND_X = 1544 * 0.99
RAND_Y = 2064 * 0.99

only_totensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

img = Image.open('/Users/rparak/Documents/GitHub/Blender_Synthetic_Data/images/Image_00001.png')

for i in range(1):
    # Load Data
    tsf = torchvision.transforms.Compose(
        [
        #torchvision.transforms.RandomRotation([8,8]),
        torchvision.transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.1, 0.5)),
        torchvision.transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.01, 1.0)),
        torchvision.transforms.RandomAffine(degrees = 0, translate = (0.05, 0.05)),
        torchvision.transforms.RandomCrop((RAND_X, RAND_Y)),
        torchvision.transforms.ToTensor()
        ]
    )

    img_1 = only_totensor(img)
    img_2 = tsf(img)

    img_1 = img_1.permute(1,2,0)
    img_2 = img_2.permute(1,2,0)

    fig, (ax_1, ax_2) = plt.subplots(1,2)
    ax_1.imshow(img_1)
    ax_2.imshow(img_2)
    plt.show()

