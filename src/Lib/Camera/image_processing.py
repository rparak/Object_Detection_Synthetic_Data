# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../../' + 'src' not in sys.path:
    sys.path.append('../../../' + 'src')
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
# The ID of the object to be processed.
#   ID{0} = 'T_Joint'
#   ID{1} = 'Metal_Blank'
CONST_OBJECT_ID = 0
# Number of data to be processed.
# Note:
#   It must be equal to the number of data 
#   in the raw images folder.
CONST_NUM_OF_DATA = 10

def main():
    """
    Description:
        A simple script to adjust the contrast {alpha} and brightness {beta} of a raw image.

        Note:
            The raw images were collected using the script here:
                ./Data_Collection/{Camera_Name}/main.py
    """

    # Processes the data up to the desired maximum number of iterations, which is given by the constant {CONST_NUM_OF_DATA}.
    i = 0
    while CONST_NUM_OF_DATA > i:
        # Start the timer.
        t_0 = time.time()

        # The specified path to the folder from which the image will be loaded (*_in) and where the image will be saved (*_out).
        file_path_in  = f'../../../Data/Camera/Raw/Images/ID_{CONST_OBJECT_ID}/Image_{(i + 1):05}.png'
        file_path_out = f'../../../Data/Camera/Processed/Images/ID_{CONST_OBJECT_ID}/Image_{(i + 1):05}.png'

        # Loads the image to the specified file.
        image_in = cv2.imread(file_path_in)
        
        # Function to adjust the contrast and brightness parameters of the input image 
        # by clipping the histogram.
        (alpha_custom, beta_custom) = Lib.Utilities.Image_Processing.Get_Alpha_Beta_Parameters(image_in, 1.0)
        
        # Adjust the contrast and brightness of the image using the alpha and beta parameters.
        #   Equation:
        #       g(i, j) = alpha * f(i, j) + beta
        image_out = cv2.convertScaleAbs(image_in, alpha=alpha_custom, beta=beta_custom)

        # Saves the image to the specified file.
        cv2.imwrite(file_path_out, image_out)
        i += 1
        
        # Display information.
        print(f'[INFO] The image with index {(i + 1)} was successfully saved to the folder.')
        print(f'[INFO]  - Path: {file_path_out}')
        print(f'[INFO] Time: {(time.time() - t_0):0.05f} in seconds.')

    print('[INFO] The data processing was completed successfully.')

if __name__ == '__main__':
    sys.exit(main())