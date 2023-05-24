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
        Prediction (testing) using the trained YOLOv8 model on new images. In this case, the model is loaded from a checkpoint 
        file and the user can provide images to perform inference. The model predicts the classes and locations of objects 
        in the input images.

        For more information see: 
            https://docs.ultralytics.com/modes/predict/
    """
    
    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Blender_Synthetic_Data')[0] + 'Blender_Synthetic_Data'

    # Load a pre-trained custom YOLO model.
    model = YOLO(f'{project_folder}/YOLO/Results/Type_{CONST_DATASET_TYPE}/train_fb_{CONST_FREEZE_BACKBONE}/weights/best.pt')

    # Predict (test) the model on a test dataset.
    model.predict(source=f'{project_folder}/Data/Dataset_Type_{CONST_DATASET_TYPE}/images/test', save=True, save_txt=True, save_conf=True, 
                 imgsz=[480, 640], conf=0.5, name=f'{project_folder}/YOLO/Results/Type_{CONST_DATASET_TYPE}/predict_fb_{CONST_FREEZE_BACKBONE}')

if __name__ == '__main__':
    main()