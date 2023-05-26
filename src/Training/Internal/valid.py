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
# An indication of whether the backbone layers of the model should be frozen.
CONST_FREEZE_BACKBONE = True
# Format of the trained model.
#   Standard YOLO *.pt format: 'pt'
#   ONNX *.onnx format: 'onnx
CONST_MODEL_FORMAT = 'onnx'

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

    # The path to the trained model.
    file_path = f'{project_folder}/YOLO/Model/Type_{CONST_DATASET_TYPE}/yolov8n_custom.{CONST_MODEL_FORMAT}'

    if os.path.isfile(file_path):
        # Load a pre-trained custom YOLO model in the desired format.
        model = YOLO(file_path)

        # Evaluate the performance of the model on the validation dataset.
        """
        model.val(data=f'{project_folder}/YOLO/Configuration/Type_{CONST_DATASET_TYPE}/config.yaml', batch=32, imgsz=640, rect=True, save_json=True,
                  split='test', name=f'{project_folder}/YOLO/Results/Type_{CONST_DATASET_TYPE}/valid_fb_{CONST_FREEZE_BACKBONE}')
        """

        result = model.val(data=f'{project_folder}/YOLO/Configuration/Type_{CONST_DATASET_TYPE}/config.yaml', batch=32, imgsz=640, iou=0.6, rect=True, save_txt=True, save_conf=True, 
                           save_json=True, dnn=True, split='test', name=f'{project_folder}/YOLO/Results/Type_{CONST_DATASET_TYPE}/valid_fb_{CONST_FREEZE_BACKBONE}')
        
        #print(result.box)
    else:
        print('[INFO] The file does not exist.')

if __name__ == '__main__':
    main()