# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# OS (Operating system interfaces)
import os
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Ultralytics (Real-time object detection and image segmentation 
# model) [pip install ultralytics]
from ultralytics import YOLO
# Custom Library:
#   ../Lib/Utilities/Image_Processing
import Lib.Utilities.Image_Processing
#   ../Lib/Utilities/General
import Lib.Utilities.General

"""
Description:
    Initialization of constants.
"""
# Available objects.
#   ID{0} = 'T_Joint'
#   ID{1} = 'Metal_Blank'
CONST_OBJECT_NAME = ['T_Joint', 'Metal_Blank']
# The color of the bounding box of the object.
CONST_OBJECT_BB_COLOR = [(255, 165, 0), (0, 165, 255)]
# The identification number of the dataset type.
CONST_DATASET_TYPE = 0
# Name of the dataset.
CONST_DATASET_NAME = f'Dataset_Type_{CONST_DATASET_TYPE}'
# The number of data to be tested for a single object.
CONST_NUM_OF_TEST_DATA = 25
# Initial iteration of the scanning process.
CONST_SCAN_ITERATION = 0
# Format of the trained model.
#   Standard YOLO *.pt format: 'PyTorch'
#   ONNX *.onnx format: 'ONNX'
CONST_MODEL_FORMAT = 'PyTorch'
# Determine whether or not to save the images.
CONST_SAVE_IMAGES = True

def main():
    """
    Description:
        A program to save image results based on object predictions from the test partition of a dataset.

        Note:
            The prediction is performed using both *.onnx and *.pt formats.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Object_Detection_Synthetic_Data')[0] + 'Object_Detection_Synthetic_Data'

    # The path to the trained model.
    if CONST_MODEL_FORMAT == 'PyTorch':
        file_path = f'{project_folder}/YOLO/Model/Type_{CONST_DATASET_TYPE}/yolov8n_custom.pt'
    elif CONST_MODEL_FORMAT == 'ONNX':
        file_path = f'{project_folder}/YOLO/Model/Type_{CONST_DATASET_TYPE}/yolov8n_dynamic_True_custom.onnx'

    # Load a pre-trained custom YOLO model.
    model = YOLO(file_path)

    # Tests the data up to the desired maximum number of iterations, which is given by the constant {CONST_NUM_OF_TEST_DATA}.
    for n_i in range(CONST_NUM_OF_TEST_DATA):
        # File path of the processed image.
        image_file_path = f'{project_folder}/Data/{CONST_DATASET_NAME}/images/test/Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png'

        # Loads images from the specified file.
        image_data = cv2.imread(image_file_path)

        # Predict (test) the model on a test dataset.
        results = model.predict(source=image_file_path, imgsz=[480, 640], conf=0.5, iou=0.7)

        # Display information.
        print(f'[INFO] Iteration: {n_i}')
        print(f'[INFO] Image: Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png')

        if CONST_SAVE_IMAGES == True:
            # If the model has found an object in the current processed image, express the results (class, bounding box, confidence).
            if results[0].boxes.shape[0] >= 1:
                # Express the data from the prediction:
                #   ID name of the class, Bounding box in the YOLO format and Confidence.
                class_id = results[0].boxes.cls.cpu().numpy(); b_box = results[0].boxes.xywhn.cpu().numpy()
                conf = results[0].boxes.conf.cpu().numpy()

                for i, (class_id_i, b_box_i, conf_i) in enumerate(zip(class_id, b_box, conf)):
                    # Create a bounding box from the label data.
                    Bounding_Box_Properties = {'Name': f'{CONST_OBJECT_NAME[int(class_id_i)]}_{i}', 'Precision': f'{str(conf_i)[0:5]}', 
                                                'Data': {'x_c': b_box_i[0], 'y_c': b_box_i[1], 'width': b_box_i[2], 'height': b_box_i[3]}}
                    
                    # Draw the bounding box of the object with additional dependencies (name, precision, etc.) in 
                    # the raw image.
                    image_data = Lib.Utilities.Image_Processing.Draw_Bounding_Box(image_data, Bounding_Box_Properties, 'YOLO', CONST_OBJECT_BB_COLOR[int(class_id_i)], 
                                                                                  True, True)
            else:
                print('[INFO] The model did not find object in the input image.')

            # Saves the images to the specified file.
            cv2.imwrite(f'{project_folder}/Data/Results/{CONST_MODEL_FORMAT}/Type_{CONST_DATASET_TYPE}/images/' +
                        f'Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png', image_data)
            print(f'[INFO] The data in iteration {int(n_i + 1)} was successfully saved to the folder {project_folder}/Data/Results/{CONST_MODEL_FORMAT}/Type_{CONST_DATASET_TYPE}/images/.')

if __name__ == '__main__':
    sys.exit(main())