# Ultralytics (Real-time object detection and image segmentation 
# model) [pip install ultralytics]
from ultralytics import YOLO
# OS (Operating system interfaces)
import os

"""
Description:
    Initialization of constants.
"""
# The identification number of the dataset type.
CONST_DATASET_TYPE = 5
# Select the desired size of YOLOv* to build the model.
#   Note:
#     Detection Model.
#   Nano: 'yolov8n', Small: 'yolov8s', Medium: 'yolov8m', Large: 'yolov8l', XLarge: 'yolov8x'}
CONST_YOLO_SIZE = 'yolov8n'

def main():
    """
    Description:
        Exporting the YOLOv8 model to a format that can be used for deployment. In this case, the model is converted 
        to (*.onnx) format.

        For more information see: 
            https://docs.ultralytics.com/modes/export/

        More information about the onnx format can be found at: 
            https://onnx.ai
    """
    
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Blender_Synthetic_Data')[0] + 'Blender_Synthetic_Data'

    # Load a pre-trained custom YOLO model.
    model = YOLO(f'{project_folder}/YOLO/Model/Type_{CONST_DATASET_TYPE}/{CONST_YOLO_SIZE}_custom.pt')

    # Export the model to *.onnx format.
    #model.export(format='onnx', imgsz=[480, 640], dynamic=True, opset=12)
    model.export(format='onnx', imgsz=640, dynamic=False, opset=12)

if __name__ == '__main__':
    main()