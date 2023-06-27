# System (Default)
import sys
#   Add access if it is not in the system path.
if '../../' + 'src' not in sys.path:
    sys.path.append('../../' + 'src')
# OS (Operating system interfaces)
import os
# Time (Time access and conversions)
import time
# OpenCV (Computer Vision) [pip3 install opencv-python]
import cv2
# Harvester (Image acquisition for GenICam-based system) [pip3 install harvesters]
from harvesters.core import Harvester

"""
Description:
    Initialization of constants.
"""
# The ID of the object to be scanned.
#   ID{0} = 'T_Joint'
#   ID{1} = 'Metal_Blank'
#   ID{-1} = ALL
CONST_OBJECT_ID = 0
# The identification number of the iteration to save the image. It starts with the number 1.
#   1 = 'Image_001', 2 = 'Image_002', etc.
CONST_INIT_INDEX = 1
# Camera resolution in pixels (x, y).
# Note: 
#   Low{'x': 1032px, 'y': 772px}, High{'x': 2064px, 'y': 1544px}
CONST_IMAGE_RESOLUTION = [2064, 1544]
# The identification number of the scanner: PhotoneoTL_DEV_{ID}
#   Note:
#       It can be found in the device details when the PhoXi 
#       Control software is opened.
CONST_DEVICE_ID = 'PhotoneoTL_DEV_' + '2019-06-011-LC3'

def main():
    """
    Description:
        A program to get a raw image (2D) from a Photoneo sensor. In our case, it is a Photoneo 
        PhoXi 3D scanner M.

        Note:
            This program can also be used for another type of Photoneo scanner.

        Additional notes can be found in the README.md file.
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Object_Detection_Synthetic_Data')[0] + 'Object_Detection_Synthetic_Data'

    # The specified path to the folder where the image will be saved.
    if CONST_OBJECT_ID == -1:
        file_path = f'{project_folder}/Data/Camera/raw/images/Image_{(CONST_INIT_INDEX):05}.png'
    else:
        file_path = f'{project_folder}/Data/Camera/raw/images/Object_ID_{CONST_OBJECT_ID}_{(CONST_INIT_INDEX):05}.png'

    # Set the path to the destination CTI file.
    cti_file_path = os.getenv('PHOXI_CONTROL_PATH') + "/API/bin/photoneo.cti"

    # Start the timer.
    t_0 = time.time()

    with Harvester() as H_Cls:
        # Adds a CTI file as one of GenTL Producers to work with.
        H_Cls.add_file(cti_file_path, True, True)
        # Updates the list that contains available devices.
        H_Cls.update()

        # Creates an image acquisition device that is mapped to the specified 
        # remote device. In our case Photoneo Scanner.
        with H_Cls.create({'id_': CONST_DEVICE_ID}) as Image_Acquirer:
            # Get a map of the element nodes that belong to the owner object.
            PhoXi_Scanner = Image_Acquirer.remote_device.node_map

            if PhoXi_Scanner.IsConnected.value == True:
                print('[INFO] The Photoneo scanner is successfully connected to the computer.')
                print(f'[INFO]  - Device ID: {CONST_DEVICE_ID}')

            # Set the main parameters of the scanner to collect data from the camera only.
            PhoXi_Scanner.PhotoneoTriggerMode.value = "Software"
            # 'Res_1032_772': Low, 'Res_2064_1544': High
            PhoXi_Scanner.Resolution.value = 'Res_2064_1544'
            # Camera-only mode.
            PhoXi_Scanner.CameraOnlyMode.value = True
            # The structure of the output to be saved.
            PhoXi_Scanner.SendTexture.value = True
            PhoXi_Scanner.SendPointCloud.value = False
            PhoXi_Scanner.SendNormalMap.value  = False
            PhoXi_Scanner.SendDepthMap.value   = False
            PhoXi_Scanner.SendConfidenceMap.value = False

            # Start the image acquisition process.
            Image_Acquirer.start()

            # Trigger the frame to acquire the data from the scanner.
            PhoXi_Scanner.TriggerFrame.execute()

            # Read the available data from the object. Component 0 in our case indicates 
            # the texture of the image.
            image_texture = Image_Acquirer.fetch().payload.components[0]

            # Check that the loaded data is in the correct format. If true, process them and save them in a folder.
            if image_texture.width == CONST_IMAGE_RESOLUTION[0] and image_texture.height == CONST_IMAGE_RESOLUTION[1]:
                # Change the shape of the collected data from 1D to 2D.
                image = image_texture.data.reshape(CONST_IMAGE_RESOLUTION[1], CONST_IMAGE_RESOLUTION[0], 1).copy()

                # Normalization of a 16-bit grayscale image.
                image_normalized = cv2.normalize(image, None, 0, 65535, cv2.NORM_MINMAX)

                # Convert a 16-bit image to 8-bit.
                image_8bit = (image_normalized/256).astype('uint8')

                # Saves the image to the specified file.
                cv2.imwrite(file_path, image_8bit)

                # Display information.
                print(f'[INFO] The image with index {CONST_INIT_INDEX} was successfully saved to the folder.')
                print(f'[INFO]  - Path: {file_path}')
                print(f'[INFO] Time: {(time.time() - t_0):0.05f} in seconds.')

if __name__ == '__main__':
    sys.exit(main())