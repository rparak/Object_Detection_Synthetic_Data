# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
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
# The ID of the object to be processed.
CONST_OBJECT_ID = 0
# Available objects.
#   ID{0} = 'T_Joint'
#   ID{1} = 'Metal_Blank'
CONST_OBJECT_NAME = ['T_Joint', 'Metal_Blank']
# Iteration of the scanning process.
CONST_SCAN_ITERATION = 1

def main():
    """
    Description:
        A simple script to evaluate synthetic data (image with corresponding label) generated from Blender.
    """

    # The specified path of the file.
    file_path = '../../../Data/Train'    

    # Load a raw image from a file.
    raw_image  = cv2.imread(f'{file_path}/Images/ID_{CONST_OBJECT_ID}/Image_{CONST_SCAN_ITERATION:05}.png')
    # Load a label (annotation) from a file.
    label_data = File_IO.Load(f'{file_path}/Labels/ID_{CONST_OBJECT_ID}/Label_{CONST_SCAN_ITERATION:05}', 'txt', ' ')[0]
    
    # Create a bounding box from the label data.
    Bounding_Box_Properties = {'Name': f'{CONST_OBJECT_NAME[int(label_data[0])]}', 'Accuracy': '99.99', 
                               'Data': {'x_c': label_data[1], 'y_c': label_data[2], 'width': label_data[3], 'height': label_data[4]}}
    
    # Draw the bounding box of the object with additional dependencies (name, precision, etc.) in 
    # the raw image.
    eval_image = Lib.Utilities.Image_Processing.Draw_Bounding_Box(raw_image, Bounding_Box_Properties, 'YOLO', (0, 255, 0), True, True)

    # Displays the image in the window.
    cv2.imshow('Test: Synthetic Data', eval_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    sys.exit(main())