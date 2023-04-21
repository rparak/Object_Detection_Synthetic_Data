# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../../' + 'src' not in sys.path:
    sys.path.append('../../../' + 'src')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OS (Operating system interfaces)
import os
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Custom Library:
#   ../Lib/Utilities/Image_Processing
import Lib.Utilities.Image_Processing

"""
Description:
    Initialization of constants.
"""
# The ID of the object to be scanned.
#   ID{0} = 'T_Joint'
#   ID{1} = 'Metal_Blank'
CONST_SCANNED_OBJ_ID = 0
# The identification number of the iteration to save the image. It starts with the number 1.
#   1 = 'Image_001', 2 = 'Image_002', etc.
CONST_INIT_INDEX = 1

def main():
    """
    Description:
        Image processing of the ...
    """

    # Locate the path to the Desktop folder.
    desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')

    # The specified path to the folder from which the image will be loaded (*_in) and where the image will be saved (*_out).
    file_path_in  = f'{desktop_path}/Data/Raw/Image_{(CONST_INIT_INDEX):05}.png'
    file_path_out = f'{desktop_path}/Data/Processed/Image_{(CONST_INIT_INDEX):05}.png'

    # Loads the image to the specified file.
    #image_in = cv2.imread(file_path_in)
    # Saves the image to the specified file.
    #cv2.imwrite(file_path_out, image_out)

if __name__ == '__main__':
    sys.exit(main())