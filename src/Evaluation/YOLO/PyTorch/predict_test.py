# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../../..')
# OS (Operating system interfaces)
import os
# Time (Time access and conversions)
import time
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Ultralytics (Real-time object detection and image segmentation 
# model) [pip install ultralytics]
from ultralytics import YOLO
# Custom Library:
#   ../Lib/Utilities/Image_Processing
import Lib.Utilities.Image_Processing


def main():
    """
    Description:
        ..
    """

    image_data = cv2.imread('Image_00001.png')

    # Load a pre-trained custom YOLO model.
    model = YOLO(f'best.pt')

    # Predict (test) the model on a test dataset.
    result = model.predict(source='Image_00001.png', imgsz=[480, 640], conf=0.25)

    if result[0].boxes.shape[0] >= 1:
        #print(result[0].boxes.cls.cpu().numpy())
        #print(result[0].boxes.conf.cpu().numpy())

        # A bounding box (in yolo format) with an associated confidence value. 
        bounding_box = result[0].boxes.xywhn.cpu().numpy()
        confidence   = result[0].boxes.conf.cpu().numpy()

        #print(f'[INFO] The model found {bounding_box.shape[0]} object in the input image.')
        for i, (bounding_box_i, confidence_i) in enumerate(zip(bounding_box, confidence)):
            #print(f'[INFO]  - Bounding Box (xywhn): {bounding_box_i}')
            #print(f'[INFO]  - Confidence: {str(np.round(confidence_i, 2))}')
        
            # Create a bounding box from the label data.
            Bounding_Box_Properties = {'Name': f'Image_{i}', 'Precision': f'{str(np.round(confidence_i, 2))}', 
                                        'Data': {'x_c': bounding_box_i[0], 'y_c': bounding_box_i[1], 'width': bounding_box_i[2], 'height': bounding_box_i[3]}}
            
            # Draw the bounding box of the object with additional dependencies (name, precision, etc.) in 
            # the raw image.
            image_data = Lib.Utilities.Image_Processing.Draw_Bounding_Box(image_data, Bounding_Box_Properties, 'YOLO', (0, 255, 0), True, True)

    # Displays the image in the window.
    cv2.imshow('Data', image_data)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()