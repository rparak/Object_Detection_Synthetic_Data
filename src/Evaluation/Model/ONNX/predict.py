# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../../..')
# Numpy (Array computing) [pip3 install numpy]
import numpy as np
# Time (Time access and conversions)
import time
# OS (Operating system interfaces)
import os
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
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
CONST_NUM_OF_TEST_DATA = 30
# Initial iteration of the scanning process.
CONST_SCAN_ITERATION = 0
# The boundaries of the object area {A}.
#   The boundaries of each object were generated using this script:
#       ../Blender/gen_object_boundaries.py
#   imported into a Blender file:
#       ../Blender/Gen_Synthetic_Data.blend
CONST_BOUNDARIES_OBJECT_A = [[0.0056, 0.0112], [0.0045, 0.0129]]
# The type of image folder to be processed.
#   'DATASET': Images for the dataset.
#   'ADDITIONAL': Images for additional tests.
CONST_IMAGE_FOLDER_TYPE = 'DATASET'
# Determine whether or not to save the images.
CONST_SAVE_IMAGES = False

def main():
    """
    Description:
        A program to save image results based on object predictions from the test partition of a dataset. The prediction is performed using 
        the *.onnx format.

        The possibility to use additional images, which can be found in the folder here:
            ../Additional/processed/images/Image_*.png

        Note:
            The program also observes the prediction speed.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Blender_Synthetic_Data')[0] + 'Blender_Synthetic_Data'

    # Load a pre-trained YOLO model in the *.onnx format.
    model = cv2.dnn.readNet(f'{project_folder}/YOLO/Model/Type_{CONST_DATASET_TYPE}/yolov8n_custom.onnx')

    # Tests the data up to the desired maximum number of iterations, which is given by the constant {CONST_NUM_OF_TEST_DATA}.
    prediction_speed = []
    for n_i in range(CONST_NUM_OF_TEST_DATA):
        # File path of the processed image.
        if CONST_IMAGE_FOLDER_TYPE == 'DATASET':
            image_file_path = f'{project_folder}/Data/{CONST_DATASET_NAME}/images/test/Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png'
        elif CONST_IMAGE_FOLDER_TYPE == 'ADDITIONAL':
            image_file_path = f'{project_folder}/Additional/processed/images/Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png'

        # Loads images from the specified file.
        image_data = cv2.imread(image_file_path)

        # Start the timer.
        t_0 = time.time()

        # Object detection using the trained YOLO model. 
        #   Express the data from the prediction:
        #       ID name of the class, Bounding box in the YOLO format and Confidence.
        (class_id, b_box, conf) = Lib.Utilities.Image_Processing.YOLO_ONNX_Format_Object_Detection(image_data, model, [640, 480], 0.50)

        # Save the prediction speed data.
        prediction_speed.append((time.time() - t_0))

        # Display information.
        print(f'[INFO] Iteration: {n_i}')
        print(f'[INFO] Image: Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png')
        #print(result[0].speed)
        # macos ~ 49ms

        if CONST_SAVE_IMAGES == True:
            # If the model has found an object in the current processed image, express the results (class, bounding box, confidence).
            if class_id != None:
                # Check if the area of each predicted object is within the boundaries.
                for i, (class_id_i, b_box_i, conf_i) in enumerate(zip(class_id, b_box, conf)):
                    # Convert bounding box data from PASCAL_VOC format to YOLO format to get the area of the rectangle..
                    b_box_yolo_tmp = Lib.Utilities.General.Convert_Boundig_Box_Data('PASCAL_VOC', 'YOLO', b_box_i, 
                                                                                    {'x': image_data.shape[1], 'y': image_data.shape[0]})

                    # Calculates the area of the rectangle from the bounding box.
                    A_tmp = b_box_yolo_tmp['width'] * b_box_yolo_tmp['height']

                    if CONST_BOUNDARIES_OBJECT_A[int(class_id_i)][0] < A_tmp < CONST_BOUNDARIES_OBJECT_A[int(class_id_i)][1]:
                        # Create a bounding box from the label data.
                        Bounding_Box_Properties = {'Name': f'{CONST_OBJECT_NAME[int(class_id_i)]}_{i}', 'Precision': f'{str(conf_i)[0:5]}', 
                                                   'Data': b_box_i}
                        
                        # Draw the bounding box of the object with additional dependencies (name, precision, etc.) in 
                        # the raw image.
                        image_data = Lib.Utilities.Image_Processing.Draw_Bounding_Box(image_data, Bounding_Box_Properties, 'PASCAL_VOC', CONST_OBJECT_BB_COLOR[int(class_id_i)], 
                                                                                      True, True)
            else:
                print('[INFO] The model did not find object in the input image.')

            # Saves the images to the specified file.
            if CONST_IMAGE_FOLDER_TYPE == 'DATASET':
                cv2.imwrite(f'{project_folder}/Data/Results/ONNX/Type_{CONST_DATASET_TYPE}/images/' +
                            f'Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png', image_data)
                print(f'[INFO] The data in iteration {int(n_i + 1)} was successfully saved to the folder {project_folder}/Data/Results/ONNX/Type_{CONST_DATASET_TYPE}/images/.')
            elif CONST_IMAGE_FOLDER_TYPE == 'ADDITIONAL':
                cv2.imwrite(f'{project_folder}/Additional/results/ONNX/Type_{CONST_DATASET_TYPE}/images/' +
                            f'Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png', image_data)
                print(f'[INFO] The data in iteration {int(n_i + 1)} was successfully saved to the folder {project_folder}/Additional/results/ONNX/Type_{CONST_DATASET_TYPE}/images/.')

    # Display the results of the average prediction time.
    print(f'[INFO] Average Prediction Speed: {np.mean(prediction_speed) * 1000:0.05f} in milliseconds.')

if __name__ == '__main__':
    sys.exit(main())