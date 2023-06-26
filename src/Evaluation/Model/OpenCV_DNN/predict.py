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
CONST_DATASET_TYPE = 5
# Name of the dataset.
CONST_DATASET_NAME = f'Dataset_Type_{CONST_DATASET_TYPE}'
# The number of data to be tested for a single object.
CONST_NUM_OF_TEST_DATA = 25
# Initial iteration of the scanning process.
CONST_SCAN_ITERATION = 0
# Determine whether or not to save the images.
CONST_SAVE_IMAGES = True

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
    project_folder = os.getcwd().split('Object_Detection_Synthetic_Data')[0] + 'Object_Detection_Synthetic_Data'

    # Load a pre-trained YOLO model in the *.onnx format.
    model = cv2.dnn.readNet(f'{project_folder}/YOLO/Model/Type_{CONST_DATASET_TYPE}/yolov8n_dynamic_False_custom.onnx')

    # Tests the data up to the desired maximum number of iterations, which is given by the constant {CONST_NUM_OF_TEST_DATA}.
    prediction_speed = []
    for n_i in range(CONST_NUM_OF_TEST_DATA):
        # File path of the processed image.
        image_file_path = f'{project_folder}/Data/{CONST_DATASET_NAME}/images/test/Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png'

        # Loads images from the specified file.
        image_data = cv2.imread(image_file_path)

        # Start the timer.
        t_0 = time.time()

        # Object detection using the trained YOLO model. 
        #   Express the data from the prediction:
        #       ID name of the class, Bounding box in the YOLO format and Confidence.
        (class_id, b_box, conf) = Lib.Utilities.Image_Processing.YOLO_ONNX_Format_Object_Detection(image_data, model, [640, 480], 0.5, 0.7)

        # Save the prediction speed data.
        prediction_speed.append((time.time() - t_0))

        # Display information.
        print(f'[INFO] Iteration: {n_i}')
        print(f'[INFO] Image: Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png')

        if CONST_SAVE_IMAGES == True:
            # If the model has found an object in the current processed image, express the results (class, bounding box, confidence).
            if class_id != None:
                for i, (class_id_i, b_box_i, conf_i) in enumerate(zip(class_id, b_box, conf)):
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
            cv2.imwrite(f'{project_folder}/Data/Results/OpenCV_DNN/Type_{CONST_DATASET_TYPE}/images/' +
                        f'Image_{(CONST_SCAN_ITERATION + (n_i + 1)):05}.png', image_data)
            print(f'[INFO] The data in iteration {int(n_i + 1)} was successfully saved to the folder {project_folder}/Data/Results/OpenCV_DNN/Type_{CONST_DATASET_TYPE}/images/.')

    # Display the results of the average prediction time.
    print(f'[INFO] Average Prediction Speed: {np.mean(prediction_speed) * 1000:0.05f} in milliseconds.')

if __name__ == '__main__':
    sys.exit(main())