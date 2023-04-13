# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Custom Library:
#   ../Lib/Utilities/General
import Lib.Utilities.General

"""
XML = {'x_min': 950, 'y_min': 614, 'x_max': 1132, 'y_max': 752}
YOLO = {'x_c': 0.504347, 'y_c': 0.442419, 'width': 0.087728, 'height': 0.089276}

res = Lib.Utilities.General.Convert_Annotation('YOLO', 'PASCAL_VOC', YOLO, {'x': width, 'y': height})
print(res)
"""
def Draw_Bounding_Box(image, bounding_box_properties = {'Name': 'Obj_Name_Id_0', 'Accuracy': '100', 'Data': None}, format = 'YOLO/Pascal_VOC', 
                      Color = (0, 255, 0), fill_box = False, visibility_info = False):
    
    if format == 'YOLO':
        data = Lib.Utilities.General.Convert_Annotation(format, 'PASCAL_VOC', bounding_box_properties['Data']) 
    data = bounding_box_properties['Data']

    image_copy = image.copy()


    # return image_out