# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../../..')
# OS (Operating system interfaces)
import os
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Time (Time access and conversions)
import time
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Custom Library:
#   ../Lib/Utilities/Image_Processing
import Lib.Utilities.Image_Processing

"""
Description:
    Initialization of constants.
"""
# The ID of the object to be tested.
CONST_OBJECT_ID = 0
# Available objects.
#   ID{0} = 'T_Joint'
#   ID{1} = 'Metal_Blank'
CONST_OBJECT_NAME = ['T_Joint', 'Metal_Blank']
# The identification number of the dataset type.
CONST_DATASET_TYPE = 0
# Name of the dataset.
CONST_DATASET_NAME = f'Dataset_Type_{CONST_DATASET_TYPE}_Obj_ID_{CONST_OBJECT_ID}'
# Number of data to be tested.
CONST_NUM_OF_TEST_DATA = 15
# Iteration of the testing process.
CONST_SCAN_ITERATION = 30
# The type of image folder to be processed.
#   'DATASET': Images for the dataset.
#   'ADDITIONAL': Images for additional tests.
CONST_IMAGE_FOLDER_TYPE = 'DATASET'
# Determine whether or not to save the images.
CONST_SAVE_IMAGES = True

def main():
    """
    Description:
        ...
    """
        
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Blender_Synthetic_Data')[0] + 'Blender_Synthetic_Data'

    # Load a pre-trained YOLO model in the *.onnx format.
    model = cv2.dnn.readNet(f'{project_folder}/YOLO/Model/Type_{CONST_DATASET_TYPE}_Obj_ID_{CONST_OBJECT_ID}/yolov8s_custom.onnx')

    for n_i in range(CONST_NUM_OF_TEST_DATA):
        # Loads images from the specified file.
        if CONST_IMAGE_FOLDER_TYPE == 'DATASET':
            image_data = cv2.imread(f'{project_folder}/Data/{CONST_DATASET_NAME}/images/test/Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png')
        elif CONST_IMAGE_FOLDER_TYPE == 'ADDITIONAL':
            image_data = cv2.imread(f'{project_folder}/Additional/ID_{CONST_OBJECT_ID}/processed/Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png')

        # Start the timer.
        t_0 = time.time()
        
        # Object detection using the trained YOLO model.
        (class_id, bounding_box, confidence) = Lib.Utilities.Image_Processing.YOLO_Object_Detection(image_data, model, 640, 0.5)

        # Display information.
        print(f'[INFO]  - Image: Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png')
        print(f'[INFO] Time: {(time.time() - t_0):0.05f} in seconds.')
        
        if CONST_SAVE_IMAGES == True:
            # Start the timer.
            t_0 = time.time()

            if class_id != None:
                for i, (class_id_i, bounding_box_i, confidence_i) in enumerate(zip(class_id, bounding_box, confidence)):
                    # Create a bounding box from the label data.
                    Bounding_Box_Properties = {'Name': f'{CONST_OBJECT_NAME[CONST_OBJECT_ID]}_{i}', 'Precision': f'{str(np.round(confidence_i, 2))}', 
                                               'Data': bounding_box_i}
                    
                    # Draw the bounding box of the object with additional dependencies (name, precision, etc.) in 
                    # the raw image.
                    image_data = Lib.Utilities.Image_Processing.Draw_Bounding_Box(image_data, Bounding_Box_Properties, 'PASCAL_VOC', (0, 255, 0), True, True)
                
            # Loads images from the specified file.
            if CONST_IMAGE_FOLDER_TYPE == 'DATASET':
                # Save the image to a file.
                cv2.imwrite(f'{project_folder}/Data/Results/ONNX/{CONST_DATASET_NAME}/images/Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png', image_data)
                print(f'[INFO] The data in iteration {int(n_i)} was successfully saved to the folder {project_folder}/Data/Results/ONNX/{CONST_DATASET_NAME}/images/.')
            elif CONST_IMAGE_FOLDER_TYPE == 'ADDITIONAL':
                # Save the image to a file.
                cv2.imwrite(f'{project_folder}/Additional/ID_{CONST_OBJECT_ID}/results/ONNX/Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png', image_data)
                print(f'[INFO] The data in iteration {int(n_i)} was successfully saved to the folder {project_folder}/Additional/ID_{CONST_OBJECT_ID}/results/ONNX/.')

            # Display information.
            print(f'[INFO]  - Image: Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png')
            print(f'[INFO] Time: {(time.time() - t_0):0.05f} in seconds.')

if __name__ == '__main__':
    sys.exit(main())