# Analysis of Object Detection using Different Types of Datasets

Create readme file during the june. 

Test all of the algorithms. Etc., etc.

# Train Results

**Dataset 0**
```bash
[INFO] Evaluation Criteria: YOLOv8
[INFO] The name of the dataset: Type_0
[INFO] The best results were found in the 290 iteration.
[INFO]  Generalized Intersection over Union (GIoU):
[INFO]  [train = 0.25577, valid = 0.38983]
[INFO]  Objectness:
[INFO]  [train = 0.30117, valid = 0.34116]
[INFO]  Classification:
[INFO]  [train = 0.77362, valid = 0.76177]
[INFO]  Pr + Rec:
[INFO]  [precision = 0.98519, recall = 1.0]
[INFO]  Mean Average Precision (mAP):
[INFO]  [mAP@0.5 = 0.995, mAP@0.5:0.95 = 0.96651]
```
**Dataset 1**
```bash
[INFO] Evaluation Criteria: YOLOv8
[INFO] The name of the dataset: Type_1
[INFO] The best results were found in the 251 iteration.
[INFO]  Generalized Intersection over Union (GIoU):
[INFO]  [train = 0.18225, valid = 0.33785]
[INFO]  Objectness:
[INFO]  [train = 0.15247, valid = 0.2144]
[INFO]  Classification:
[INFO]  [train = 0.75575, valid = 0.77062]
[INFO]  Pr + Rec:
[INFO]  [precision = 0.99522, recall = 1.0]
[INFO]  Mean Average Precision (mAP):
[INFO]  [mAP@0.5 = 0.995, mAP@0.5:0.95 = 0.95957]
```
**Dataset 2**
```bash
[INFO] Evaluation Criteria: YOLOv8
[INFO] The name of the dataset: Type_2
[INFO] The best results were found in the 271 iteration.
[INFO]  Generalized Intersection over Union (GIoU):
[INFO]  [train = 0.32909, valid = 0.42055]
[INFO]  Objectness:
[INFO]  [train = 0.37482, valid = 0.31906]
[INFO]  Classification:
[INFO]  [train = 0.76431, valid = 0.80979]
[INFO]  Pr + Rec:
[INFO]  [precision = 0.97964, recall = 1.0]
[INFO]  Mean Average Precision (mAP):
[INFO]  [mAP@0.5 = 0.995, mAP@0.5:0.95 = 0.96579]
```
**Dataset 3**
```bash
[INFO] Evaluation Criteria: YOLOv8
[INFO] The name of the dataset: Type_3
[INFO] The best results were found in the 208 iteration.
[INFO]  Generalized Intersection over Union (GIoU):
[INFO]  [train = 0.22537, valid = 0.33045]
[INFO]  Objectness:
[INFO]  [train = 0.17904, valid = 0.20192]
[INFO]  Classification:
[INFO]  [train = 0.78709, valid = 0.79231]
[INFO]  Pr + Rec:
[INFO]  [precision = 0.99912, recall = 1.0]
[INFO]  Mean Average Precision (mAP):
[INFO]  [mAP@0.5 = 0.995, mAP@0.5:0.95 = 0.96948]
```
**Dataset 4**
```bash
[INFO] Evaluation Criteria: YOLOv8
[INFO] The name of the dataset: Type_4
[INFO] The best results were found in the 287 iteration.
[INFO]  Generalized Intersection over Union (GIoU):
[INFO]  [train = 0.20392, valid = 0.1957]
[INFO]  Objectness:
[INFO]  [train = 0.15947, valid = 0.16746]
[INFO]  Classification:
[INFO]  [train = 0.77582, valid = 0.74361]
[INFO]  Pr + Rec:
[INFO]  [precision = 0.99906, recall = 1.0]
[INFO]  Mean Average Precision (mAP):
[INFO]  [mAP@0.5 = 0.995, mAP@0.5:0.95 = 0.99326]
```
**Dataset 5**
```bash
[INFO] Evaluation Criteria: YOLOv8
[INFO] The name of the dataset: Type_5
[INFO] The best results were found in the 291 iteration.
[INFO]  Generalized Intersection over Union (GIoU):
[INFO]  [train = 0.19735, valid = 0.20787]
[INFO]  Objectness:
[INFO]  [train = 0.16188, valid = 0.17122]
[INFO]  Classification:
[INFO]  [train = 0.76623, valid = 0.74044]
[INFO]  Pr + Rec:
[INFO]  [precision = 0.99909, recall = 1.0]
[INFO]  Mean Average Precision (mAP):
[INFO]  [mAP@0.5 = 0.995, mAP@0.5:0.95 = 0.99296]
```

# Validation Results (PyTorch)

```bash
Ultralytics YOLOv8.0.107 🚀 Python-3.11.2 torch-2.1.0.dev20230416 CPU
Model summary (fused): 168 layers, 3006038 parameters, 0 gradients
val: Scanning /Users/rparak/Documents/GitHub/Object_Detection_Synthetic_Data/Data/Dataset_Type_0/labels/test.cache... 25 images, 0 backgrounds, 0 corrupt: 100%|██████████| 25/25 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:04<00:00,  4.82s/it]
                   all         25        146      0.974      0.991      0.995      0.881
               T_Joint         25        109      0.998      0.982      0.995       0.92
           Metal_Blank         25         37      0.949          1      0.995      0.841
Speed: 0.9ms preprocess, 137.0ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to /opt/homebrew/runs/detect/val120
Ultralytics YOLOv8.0.107 🚀 Python-3.11.2 torch-2.1.0.dev20230416 CPU
Model summary (fused): 168 layers, 3006038 parameters, 0 gradients

val: Scanning /Users/rparak/Documents/GitHub/Object_Detection_Synthetic_Data/Data/Dataset_Type_1/labels/test.cache... 25 images, 0 backgrounds, 0 corrupt: 100%|██████████| 25/25 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:04<00:00,  4.74s/it]
                   all         25        146      0.996          1      0.995      0.904
               T_Joint         25        109          1          1      0.995       0.93
           Metal_Blank         25         37      0.991          1      0.995      0.878
Speed: 0.9ms preprocess, 134.5ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to /opt/homebrew/runs/detect/val121
Ultralytics YOLOv8.0.107 🚀 Python-3.11.2 torch-2.1.0.dev20230416 CPU
Model summary (fused): 168 layers, 3006038 parameters, 0 gradients

val: Scanning /Users/rparak/Documents/GitHub/Object_Detection_Synthetic_Data/Data/Dataset_Type_2/labels/test.cache... 25 images, 0 backgrounds, 0 corrupt: 100%|██████████| 25/25 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:04<00:00,  4.81s/it]
                   all         25        146       0.95      0.976      0.992      0.833
               T_Joint         25        109       0.99      0.951      0.992      0.786
           Metal_Blank         25         37      0.909          1      0.992      0.879
Speed: 0.6ms preprocess, 136.1ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to /opt/homebrew/runs/detect/val122
Ultralytics YOLOv8.0.107 🚀 Python-3.11.2 torch-2.1.0.dev20230416 CPU
Model summary (fused): 168 layers, 3006038 parameters, 0 gradients

val: Scanning /Users/rparak/Documents/GitHub/Object_Detection_Synthetic_Data/Data/Dataset_Type_3/labels/test.cache... 25 images, 0 backgrounds, 0 corrupt: 100%|██████████| 25/25 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:04<00:00,  4.78s/it]
                   all         25        146      0.976       0.99      0.995      0.858
               T_Joint         25        109          1       0.98      0.995      0.815
           Metal_Blank         25         37      0.952          1      0.995      0.901
Speed: 0.6ms preprocess, 134.8ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to /opt/homebrew/runs/detect/val123
Ultralytics YOLOv8.0.107 🚀 Python-3.11.2 torch-2.1.0.dev20230416 CPU
Model summary (fused): 168 layers, 3006038 parameters, 0 gradients

val: Scanning /Users/rparak/Documents/GitHub/Object_Detection_Synthetic_Data/Data/Dataset_Type_4/labels/test.cache... 25 images, 0 backgrounds, 0 corrupt: 100%|██████████| 25/25 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:04<00:00,  4.72s/it]
                   all         25        146      0.983      0.996      0.995       0.87
               T_Joint         25        109          1      0.992      0.995      0.842
           Metal_Blank         25         37      0.966          1      0.995      0.897
Speed: 0.6ms preprocess, 133.2ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to /opt/homebrew/runs/detect/val124
Ultralytics YOLOv8.0.107 🚀 Python-3.11.2 torch-2.1.0.dev20230416 CPU
Model summary (fused): 168 layers, 3006038 parameters, 0 gradients

val: Scanning /Users/rparak/Documents/GitHub/Object_Detection_Synthetic_Data/Data/Dataset_Type_5/labels/test.cache... 25 images, 0 backgrounds, 0 corrupt: 100%|██████████| 25/25 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 1/1 [00:04<00:00,  4.70s/it]
                   all         25        146      0.998          1      0.995      0.916
               T_Joint         25        109      0.999          1      0.995      0.939
           Metal_Blank         25         37      0.997          1      0.995      0.893
Speed: 0.6ms preprocess, 132.5ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to /opt/homebrew/runs/detect/val125
```

# Validation Results (ONNX)

```bash
WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify', or 'pose'.
Ultralytics YOLOv8.0.107 🚀 Python-3.11.2 torch-2.1.0.dev20230416 CPU
Loading /Users/rparak/Documents/GitHub/Object_Detection_Synthetic_Data/YOLO/Model/Type_0/yolov8n_dynamic_True_custom.onnx for ONNX Runtime inference...
Forcing batch=1 square inference (1,3,640,640) for non-PyTorch models
val: Scanning /Users/rparak/Documents/GitHub/Object_Detection_Synthetic_Data/Data/Dataset_Type_0/labels/test.cache... 25 images, 0 backgrounds, 0 corrupt: 100%|██████████| 25/25 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 25/25 [00:02<00:00, 11.94it/s]
                   all         25        146      0.987      0.968      0.995      0.884
               T_Joint         25        109          1      0.936      0.995      0.918
           Metal_Blank         25         37      0.974          1      0.995      0.849
Speed: 0.7ms preprocess, 27.5ms inference, 0.0ms loss, 0.3ms postprocess per image
Results saved to /opt/homebrew/runs/detect/val126

WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify', or 'pose'.
Ultralytics YOLOv8.0.107 🚀 Python-3.11.2 torch-2.1.0.dev20230416 CPU
Loading /Users/rparak/Documents/GitHub/Object_Detection_Synthetic_Data/YOLO/Model/Type_1/yolov8n_dynamic_True_custom.onnx for ONNX Runtime inference...
Forcing batch=1 square inference (1,3,640,640) for non-PyTorch models
val: Scanning /Users/rparak/Documents/GitHub/Object_Detection_Synthetic_Data/Data/Dataset_Type_1/labels/test.cache... 25 images, 0 backgrounds, 0 corrupt: 100%|██████████| 25/25 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 25/25 [00:02<00:00, 12.00it/s]
                   all         25        146      0.995          1      0.995      0.905
               T_Joint         25        109          1          1      0.995      0.928
           Metal_Blank         25         37      0.991          1      0.995      0.881
Speed: 0.6ms preprocess, 27.7ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to /opt/homebrew/runs/detect/val127

WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify', or 'pose'.
Ultralytics YOLOv8.0.107 🚀 Python-3.11.2 torch-2.1.0.dev20230416 CPU
Loading /Users/rparak/Documents/GitHub/Object_Detection_Synthetic_Data/YOLO/Model/Type_2/yolov8n_dynamic_True_custom.onnx for ONNX Runtime inference...
Forcing batch=1 square inference (1,3,640,640) for non-PyTorch models
val: Scanning /Users/rparak/Documents/GitHub/Object_Detection_Synthetic_Data/Data/Dataset_Type_2/labels/test.cache... 25 images, 0 backgrounds, 0 corrupt: 100%|██████████| 25/25 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 25/25 [00:02<00:00, 11.89it/s]
                   all         25        146      0.969      0.975      0.994      0.827
               T_Joint         25        109      0.991      0.959      0.994      0.781
           Metal_Blank         25         37      0.948      0.991      0.993      0.873
Speed: 0.5ms preprocess, 28.2ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to /opt/homebrew/runs/detect/val128

WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify', or 'pose'.
Ultralytics YOLOv8.0.107 🚀 Python-3.11.2 torch-2.1.0.dev20230416 CPU
Loading /Users/rparak/Documents/GitHub/Object_Detection_Synthetic_Data/YOLO/Model/Type_3/yolov8n_dynamic_True_custom.onnx for ONNX Runtime inference...
Forcing batch=1 square inference (1,3,640,640) for non-PyTorch models
val: Scanning /Users/rparak/Documents/GitHub/Object_Detection_Synthetic_Data/Data/Dataset_Type_3/labels/test.cache... 25 images, 0 backgrounds, 0 corrupt: 100%|██████████| 25/25 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 25/25 [00:02<00:00, 11.87it/s]
                   all         25        146      0.981      0.983      0.995      0.858
               T_Joint         25        109          1      0.967      0.995      0.815
           Metal_Blank         25         37      0.963          1      0.995      0.901
Speed: 0.5ms preprocess, 27.7ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to /opt/homebrew/runs/detect/val129

WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify', or 'pose'.
Ultralytics YOLOv8.0.107 🚀 Python-3.11.2 torch-2.1.0.dev20230416 CPU
Loading /Users/rparak/Documents/GitHub/Object_Detection_Synthetic_Data/YOLO/Model/Type_4/yolov8n_dynamic_True_custom.onnx for ONNX Runtime inference...
Forcing batch=1 square inference (1,3,640,640) for non-PyTorch models
val: Scanning /Users/rparak/Documents/GitHub/Object_Detection_Synthetic_Data/Data/Dataset_Type_4/labels/test.cache... 25 images, 0 backgrounds, 0 corrupt: 100%|██████████| 25/25 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 25/25 [00:02<00:00, 11.90it/s]
                   all         25        146      0.993          1      0.995      0.859
               T_Joint         25        109      0.999          1      0.995      0.829
           Metal_Blank         25         37      0.987          1      0.995      0.889
Speed: 0.5ms preprocess, 27.7ms inference, 0.0ms loss, 0.2ms postprocess per image
Results saved to /opt/homebrew/runs/detect/val130

WARNING ⚠️ Unable to automatically guess model task, assuming 'task=detect'. Explicitly define task for your model, i.e. 'task=detect', 'segment', 'classify', or 'pose'.
Ultralytics YOLOv8.0.107 🚀 Python-3.11.2 torch-2.1.0.dev20230416 CPU
Loading /Users/rparak/Documents/GitHub/Object_Detection_Synthetic_Data/YOLO/Model/Type_5/yolov8n_dynamic_True_custom.onnx for ONNX Runtime inference...
Forcing batch=1 square inference (1,3,640,640) for non-PyTorch models
val: Scanning /Users/rparak/Documents/GitHub/Object_Detection_Synthetic_Data/Data/Dataset_Type_5/labels/test.cache... 25 images, 0 backgrounds, 0 corrupt: 100%|██████████| 25/25 [00:00<?, ?it/s]
                 Class     Images  Instances      Box(P          R      mAP50  mAP50-95): 100%|██████████| 25/25 [00:02<00:00, 11.85it/s]
                   all         25        146      0.998          1      0.995      0.919
               T_Joint         25        109      0.999          1      0.995      0.939
           Metal_Blank         25         37      0.996          1      0.995      0.899
Speed: 0.5ms preprocess, 27.8ms inference, 0.0ms loss, 0.3ms postprocess per image
Results saved to /opt/homebrew/runs/detect/val131
```
