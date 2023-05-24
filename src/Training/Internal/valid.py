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

def main():
    """
    Description:
        Validation of the YOLOv8 model after training. In this case, the model is evaluated on a validation dataset to measure 
        its accuracy and generalization performance.

        For more information see: 
            https://docs.ultralytics.com/modes/val/
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Blender_Synthetic_Data')[0] + 'Blender_Synthetic_Data'

    # Load a pre-trained custom YOLO model.
    model = YOLO(f'{project_folder}/YOLO/Results/Type_{CONST_DATASET_TYPE}/train_fb_{CONST_FREEZE_BACKBONE}/weights/best.pt')

    # Evaluate the performance of the model on the validation dataset.
    model.val(data=f'{project_folder}/YOLO/Configuration/Type_{CONST_DATASET_TYPE}/config.yaml', batch=32, imgsz=640, rect=True,
              name=f'{project_folder}/YOLO/Results/Type_{CONST_DATASET_TYPE}/valid_fb_{CONST_FREEZE_BACKBONE}')

if __name__ == '__main__':
    main()