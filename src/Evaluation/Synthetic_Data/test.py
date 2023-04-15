# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Custom Library:
#   ../Lib/Utilities/Image_Processing
import Lib.Utilities.Image_Processing
#   ../Lib/Utilities/File_IO
import Lib.Utilities.File_IO as File_IO

"""
Description:
    Initialization of constants.
"""
CONST_OBJECT_ID = 1
CONST_OBJECT_NAME = ['T_Joint', 'Metal_Blank']
CONST_SCAN_ITERATION = 1

def main():
    path = '../../../Data/Train'    

    raw_image  = cv2.imread(f'{path}/Images/ID_{CONST_OBJECT_ID}/Image_{CONST_SCAN_ITERATION:07}.png')
    label_data = File_IO.Load(f'{path}/Labels/ID_{CONST_OBJECT_ID}/Label_{CONST_SCAN_ITERATION:07}', 'txt', ' ')[0]
    
    Bounding_Box_Properties = {'Name': f'{CONST_OBJECT_NAME[int(label_data[0])]}', 'Accuracy': '99.99', 
                               'Data': {'x_c': label_data[1], 'y_c': label_data[2], 'width': label_data[3], 'height': label_data[4]}}
    eval_image = Lib.Utilities.Image_Processing.Draw_Bounding_Box(raw_image, Bounding_Box_Properties, 'YOLO', (0, 255, 0), True, True)

    # ...
    cv2.imshow('Test: Synthetic Data', eval_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main())