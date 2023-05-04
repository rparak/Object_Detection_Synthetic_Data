# System (Default)
import sys
#   Add access if it is not in the system path.
if '../' + 'src' not in sys.path:
    sys.path.append('../..')
# ...
import torch
import torchvision.ops.boxes
# Custom Library:
#   ../Lib/Utilities/General
import Lib.Utilities.General
 

# need to convert the yolo bbox to pascal_voc bbox!
res = Lib.Utilities.General.Convert_Boundig_Box_Data('YOLO', 'PASCAL_VOC', {'x_c': 0.507994, 'y_c': 0.473122, 'width': 0.082849, 'height': 0.083549}, {'x': 2064, 'y': 1544})
print(res)

# Bounding box coordinates.
ground_truth_bbox = torch.tensor([[962, 666, 1133, 795]], dtype=torch.float)
prediction_bbox = torch.tensor([[964, 667, 1150, 790]], dtype=torch.float)
 
# Get iou.
iou = torchvision.ops.boxes.box_iou(ground_truth_bbox, prediction_bbox)
print('IOU : ', iou.numpy()[0][0])