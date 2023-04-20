# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../' + 'src')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OS (Operating system interfaces)
import os

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
        Get the raw image from ...
    """

    # Locate the path to the Desktop folder.
    desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')


    full_path = f'{desktop_path}/Data/Photoneo/Raw/Images/ID_{CONST_SCANNED_OBJ_ID}/Image_{(CONST_INIT_INDEX + 1):05}'

if __name__ == '__main__':
    sys.exit(main())