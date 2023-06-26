# Ultralytics (Real-time object detection and image segmentation 
# model) [pip install ultralytics]
from ultralytics import YOLO
# OS (Operating system interfaces)
import os

"""
Description:
    Initialization of constants.
"""
# Number of datasets.
CONST_NUM_OF_DATASETS = 6
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
    project_folder = os.getcwd().split('Object_Detection_Synthetic_Data')[0] + 'Object_Detection_Synthetic_Data'

    for i in range(CONST_NUM_OF_DATASETS):
        # The path to the trained model.
        file_path = f'{project_folder}/YOLO/Model/Type_{i}/'

        # Export onnx model with/without dynamic axes.
        for _, dynamic_const in enumerate([False, True]):
            # Remove the ONNX model, if it already exists.
            if os.path.isfile(f'{file_path}/{CONST_YOLO_SIZE}_dynamic_{dynamic_const}_custom.onnx'):
                print(f'[INFO] Removing the ONNX model.')
                os.remove(f'{file_path}/{CONST_YOLO_SIZE}_custom.onnx')

            # Load a pre-trained custom YOLO model.
            model = YOLO(f'{project_folder}/YOLO/Model/Type_{i}/{CONST_YOLO_SIZE}_custom.pt')

            # Export the model to *.onnx format.
            model.export(format='onnx', imgsz=[480, 640], dynamic=dynamic_const, opset=12)

            # Rename the file.
            os.rename(f'{file_path}/{CONST_YOLO_SIZE}_custom.onnx', f'{file_path}/{CONST_YOLO_SIZE}_dynamic_{dynamic_const}_custom.onnx')
            print(f'[INFO] The file {CONST_YOLO_SIZE}_custom.onnx was successfully renamed to {CONST_YOLO_SIZE}_dynamic_{dynamic_const}_custom.onnx.')

if __name__ == '__main__':
    main()