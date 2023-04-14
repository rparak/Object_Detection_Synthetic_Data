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
CONST_OBJECT_NAME = 'Test_Name'
CONST_OBJECT_ID = 1

def main():
    raw_image  = cv2.imread(f'../../../Data/Train/Image/{CONST_OBJECT_NAME}_{CONST_OBJECT_ID:07}.png')
    label_data = File_IO.Load(f'../../../Data/Train/Label/{CONST_OBJECT_NAME}_{CONST_OBJECT_ID:07}', 'txt', ' ')[0]
    
    Bounding_Box_Properties = {'Name': f'Obj_Name_Id_{int(label_data[0])}', 'Accuracy': '100', 
                               'Data': {'x_c': label_data[1], 'y_c': label_data[2], 'width': label_data[3], 'height': label_data[4]}}
    eval_image = Lib.Utilities.Image_Processing.Draw_Bounding_Box(raw_image, Bounding_Box_Properties, 'YOLO', (0, 255, 0), True, True)

    # ...
    cv2.imshow('Test OpenCV', eval_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main())