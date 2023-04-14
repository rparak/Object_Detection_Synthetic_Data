# System (Default)
import sys
sys.path.append('..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Custom Library:
#   ../Lib/Utilities/Image_Processing
import Lib.Utilities.Image_Processing

def main():
    img_in = cv2.imread('Test_Image_10.png')

    BBox_Properties = {'Name': 'Obj_Name_Id_0', 'Accuracy': '100', 'Data': {'x_min': 950, 'y_min': 614, 'x_max': 1132, 'y_max': 752}}
    img_out = Lib.Utilities.Image_Processing.Draw_Bounding_Box(img_in, BBox_Properties, 'PASCAL_VOC', (0, 255, 0), True, True)

    cv2.imshow('Test OpenCV', img_out)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main())