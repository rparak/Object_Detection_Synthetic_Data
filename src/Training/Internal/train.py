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
# Name of the dataset.
CONST_DATASET_NAME = f'Dataset_Type_{CONST_DATASET_TYPE}'
# Select the desired size of YOLOv* to build the model.
#   Note:
#     Detection Model.
#   Nano: 'yolov8n', Small: 'yolov8s', Medium: 'yolov8m', Large: 'yolov8l', XLarge: 'yolov8x'}
CONST_YOLO_SIZE = 'yolov8n'
# An indication of whether the backbone layers of the model should be frozen.
CONST_FREEZE_BACKBONE = True

def Freeze_Backbone(trainer):
  """
  Description:
    Function to freeze the backbone layers of the model.

    Reference:
      https://docs.ultralytics.com/yolov5/tutorials/transfer_learning_with_frozen_layers/

    Note:
      The backbone of the model consists of layers 0-9 (10 layers).
  """

  # Number of layers to be frozen.
  num_frozen_layers = 10

  # Express the model.
  model = trainer.model

  print('[INFO] Freeze the backbone layers of the model:')
  freeze = [f'model.{x}.' for x in range(num_frozen_layers)]
  for k, v in model.named_parameters(): 
      v.requires_grad = True
      if any(x in k for x in freeze):
          print(f'[INFO] Freezing: {k}') 
          v.requires_grad = False

def main():
    """
    Description:
        Training the YOLOv8 model on a custom dataset. In this case, the model is trained using the specified dataset 
        and hyperparameters. 

        For more information see: 
            https://docs.ultralytics.com/modes/train/#arguments

        Warning:
            The config.yaml file needs to be changed to allow access to the path (internal/google colab) to the dataset 
            to be used for training.
                ../YOLO/Configuration/Type_{dataset_type}/config.yaml
    """

    # Locate the path to the project folder.
    project_folder = os.getcwd().split('Object_Detection_Synthetic_Data')[0] + 'Object_Detection_Synthetic_Data'
  
    # Remove the YOLO model, if it already exists.
    if os.path.isfile(f'{CONST_YOLO_SIZE}.pt'):
        print(f'[INFO] Removing the YOLO model.')
        os.remove(f'{CONST_YOLO_SIZE}.pt')

    # Load a pre-trained YOLO model.
    model = YOLO(f'{CONST_YOLO_SIZE}.pt')

    if CONST_FREEZE_BACKBONE == True:
        # Triggered when the training starts.
        model.add_callback('on_train_start', Freeze_Backbone)

    # Training the model on a custom dataset with additional dependencies (number of epochs, image size, etc.)
    model.train(data=f'{project_folder}/YOLO/Configuration/Type_{CONST_DATASET_TYPE}/config.yaml', batch=-1, imgsz=640, epochs=300, patience=0,
                rect=True, name=f'{project_folder}/YOLO/Results/Type_{CONST_DATASET_TYPE}/train_fb_{CONST_FREEZE_BACKBONE}')

if __name__ == '__main__':
    main()