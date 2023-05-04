# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# OS (Operating system interfaces)
import os
# SciencePlots (Matplotlib styles for scientific plotting) [pip3 install SciencePlots]
import scienceplots
# Time (Time access and conversions)
import time
# PyTorch (Tensors and Dynamic neural networks) [pip3 install torch]
import torch
# Torchvision (Image and video datasets and models) [pip3 install torchvision]
import torchvision.ops.boxes
# Custom Library:
#   ../Lib/Utilities/General
import Lib.Utilities.General

def main():
    pass

if __name__ == '__main__':
    sys.exit(main())