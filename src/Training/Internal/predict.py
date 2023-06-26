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
        Prediction (testing) using the trained YOLOv8 model on new images. In this case, the model is loaded from a checkpoint 
        file and the user can provide images to perform inference. The model predicts the classes and locations of objects 
        in the input images.

        For more information see: 
            https://docs.ultralytics.com/modes/predict/

        Note:
            A model in onnx format can also be used for prediction.

            Make sure the model was copied using the program:
                $ copy_model.py
            and converted to *.onnx format using the program:
                $ convert_model.py
    """
    
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Object_Detection_Synthetic_Data')[0] + 'Object_Detection_Synthetic_Data'

    # The path to the trained model.
    if CONST_MODEL_FORMAT == 'pt':
        file_path = f'{project_folder}/YOLO/Model/Type_{CONST_DATASET_TYPE}/{CONST_YOLO_SIZE}_custom.{CONST_MODEL_FORMAT}'
    elif CONST_MODEL_FORMAT == 'onnx':
        file_path = f'{project_folder}/YOLO/Model/Type_{CONST_DATASET_TYPE}/{CONST_YOLO_SIZE}_dynamic_{CONST_DYNAMIC}_custom.{CONST_MODEL_FORMAT}'

    if os.path.isfile(file_path):
        # Load a pre-trained custom YOLO model in the desired format.
        model = YOLO(file_path)

        # Predict (test) the model on a test dataset.
        model.predict(source=f'{project_folder}/Data/Dataset_Type_{CONST_DATASET_TYPE}/images/test', save=True, save_txt=True, save_conf=True, 
                      imgsz=[480, 640], conf=0.5, iou=0.7, name=f'{project_folder}/YOLO/Results/Type_{CONST_DATASET_TYPE}/predict_fb_{CONST_FREEZE_BACKBONE}')
    else:
        print('[INFO] The file does not exist.')

if __name__ == '__main__':
    main()