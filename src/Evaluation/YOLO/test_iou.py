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
 
# 0.5752174979344399, 0.5547668865577835, 0.0760028951391436, 0.10135385826562227

# need to convert the yolo bbox to pascal_voc bbox!
res = Lib.Utilities.General.Convert_Boundig_Box_Data('YOLO', 'PASCAL_VOC', {'x_c': 0.5752174979344399, 'y_c': 0.5547668865577835, 'width': 0.0760028951391436, 'height': 0.10135385826562227}, {'x': 2064, 'y': 1544})

# Bounding box coordinates.
ground_truth_bbox = torch.tensor([[1108, 778, 1265, 934]], dtype=torch.float)
prediction_bbox = torch.tensor([[1108, 778, 1265, 934]], dtype=torch.float)
 
# Get iou.
iou = torchvision.ops.boxes.box_iou(ground_truth_bbox, prediction_bbox)
print('IOU : ', iou.numpy()[0][0])