# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
# OS (Operating system interfaces)
import os
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2

"""
Description:
    Initialization of constants.
"""
# The ID of the object to be processed.
CONST_OBJECT_ID = 0
# Iteration of the scanning process.
CONST_SCAN_ITERATION = 1
# Type of stored data:
#   Note:
#       - 'Raw': Raw data obtained from the sensor.
#       - 'Processed': Processed data from the previous type.
CONST_TYPE_STORED_DATA = 'Raw'

def main():
    """
    Description:
        A simple script to display data from a real camera. In our case it is the camera from the PhoXi 3D Scanner M.
    """

    # Locate the path to the Desktop folder.
    desktop_path = os.path.join(os.path.join(os.path.expanduser('~')), 'Desktop')

    # The specified path of the file.
    file_path = f'{desktop_path}/Data/Photoneo/{CONST_TYPE_STORED_DATA}'    

    # Load a raw image from a file.
    raw_image  = cv2.imread(f'{file_path}/Images/ID_{CONST_OBJECT_ID}/Image_{CONST_SCAN_ITERATION:05}.png')

    # Displays the image in the window.
    cv2.imshow('Camera data processed by the PhoXi 3D Scanner M', raw_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main())