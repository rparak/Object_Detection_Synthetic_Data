# Imports
import torchvision
import matplotlib.pyplot as plt
from PIL import Image
import cv2
import numpy as np

# https://pytorch.org/vision/0.11/auto_examples/plot_transforms.html
# https://pseudo-lab.github.io/Tutorial-Book-en/chapters/en/object-detection/Ch3-preprocessing.html
# https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/Basics/pytorch_transforms.py

RAND_X = 1544 * 0.99
RAND_Y = 2064 * 0.99

img = cv2.imread('/Users/rparak/Documents/GitHub/Blender_Synthetic_Data/images/Image_00001.png')

only_totensor = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

#img = Image.open('/Users/rparak/Documents/GitHub/Blender_Synthetic_Data/images/Image_00001.png')

# You may need to convert the color.
img_raw = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = Image.fromarray(img_raw)

# Load Data
tsf = torchvision.transforms.Compose(
    [
    torchvision.transforms.ColorJitter(brightness=(0.5, 1.5), contrast=(0.5, 1.5), saturation=(0.1, 0.5)),
    torchvision.transforms.GaussianBlur(kernel_size=(5, 5), sigma=(0.01, 1.0)),
    torchvision.transforms.RandomAffine(degrees = 0, translate = (0.025, 0.025)),
    torchvision.transforms.RandomResizedCrop(size = (1544, 2064), scale = (0.98, 1.0)),
    torchvision.transforms.ToTensor()
    ]
)

img_1 = only_totensor(img)
img_2 = tsf(img)

img_1 = img_1.permute(1,2,0)
img_2 = img_2.permute(1,2,0) 


img_n = cv2.cvtColor(np.array(img_2), cv2.COLOR_RGB2BGR)


cv2.imwrite('/Users/rparak/Documents/GitHub/Blender_Synthetic_Data/images/Image_00002.png', 255*img_n)
# Displays the image in the window.
#cv2.imshow('Synthetic Data Generated by Blender', img_n)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

