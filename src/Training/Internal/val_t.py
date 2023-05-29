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
CONST_DATASET_TYPE = 0
# An indication of whether the backbone layers of the model should be frozen.
CONST_FREEZE_BACKBONE = True
# Select the desired size of YOLOv* to build the model.
#   Note:
#     Detection Model.
#   Nano: 'yolov8n', Small: 'yolov8s', Medium: 'yolov8m', Large: 'yolov8l', XLarge: 'yolov8x'}
CONST_YOLO_SIZE = 'yolov8n'
# Format of the trained model.
#   Standard YOLO *.pt format: 'pt'
#   ONNX *.onnx format: 'onnx
CONST_MODEL_FORMAT = 'pt'
# ONNX/TensorRT: Dynamic axes.
#   Note:
#       It is only used if the onnx model format is enabled.
CONST_DYNAMIC = True

def main():
    """
    Description:
        Validation of the YOLOv8 model after training. In this case, the model is evaluated on a test dataset to measure 
        its accuracy and generalization performance.

        For more information see: 
            https://docs.ultralytics.com/modes/val/

        Note:
            A model in onnx format can also be used for validation.

            Make sure the model was copied using the program:
                $ copy_model.py
            and converted to *.onnx format using the program:
                $ convert_model.py
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Blender_Synthetic_Data')[0] + 'Blender_Synthetic_Data'

    for i in range(5):
        # Load a pre-trained custom YOLO model in the desired format.
        model = YOLO(f'/home/romanp/Documents/GitHub/Blender_Synthetic_Data/YOLO/Results/Type_{CONST_DATASET_TYPE}/train_fb_True_{i}/weights/best.pt')

        # Evaluate the performance of the model on the validation dataset.
        model.val(data=f'{project_folder}/YOLO/Configuration/Type_{i}/config.yaml', batch=32, imgsz=640, conf=0.001, iou=0.6, rect=True, 
                  split='test')
    
if __name__ == '__main__':
    main()