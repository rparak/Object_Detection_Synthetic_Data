# Analysis of Training Object Detection Models using Different Types of Datasets

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

**Dataset 0**
```bash
[INFO] Evaluation Criteria: YOLOv8
[INFO] The name of the dataset: Dataset_Type_0
[INFO] Precision = 0.9738244338575095
[INFO] Recall = 0.9908256880733946
[INFO] mAP@0.5 = 0.994954128440367
[INFO] mAP@0.5:0.95 = 0.8805750003890992
```
**Dataset 1**
```bash
[INFO] Evaluation Criteria: YOLOv8
[INFO] The name of the dataset: Dataset_Type_1
[INFO] Precision = 0.9957268748983636
[INFO] Recall = 1.0
[INFO] mAP@0.5 = 0.995
[INFO] mAP@0.5:0.95 = 0.9042182376810197
```
**Dataset 2**
```bash
[INFO] Evaluation Criteria: YOLOv8
[INFO] The name of the dataset: Dataset_Type_2
[INFO] Precision = 0.9499364977915459
[INFO] Recall = 0.9756915212357327
[INFO] mAP@0.5 = 0.9920815850815851
[INFO] mAP@0.5:0.95 = 0.8325844233669368
```
**Dataset 3**
```bash
[INFO] Evaluation Criteria: YOLOv8
[INFO] The name of the dataset: Dataset_Type_3
[INFO] Precision = 0.9761978900918422
[INFO] Recall = 0.9899073555800009
[INFO] mAP@0.5 = 0.995
[INFO] mAP@0.5:0.95 = 0.8578403958725997
```
**Dataset 4**
```bash
[INFO] The name of the dataset: Dataset_Type_4
[INFO] Precision = 0.9828934023254254
[INFO] Recall = 0.9959432170318465
[INFO] mAP@0.5 = 0.995
[INFO] mAP@0.5:0.95 = 0.8695596659413892
```
**Dataset 5**
```bash
[INFO] Evaluation Criteria: YOLOv8
[INFO] The name of the dataset: Dataset_Type_5
[INFO] Precision = 0.9979897262164512
[INFO] Recall = 1.0
[INFO] mAP@0.5 = 0.995
[INFO] mAP@0.5:0.95 = 0.9163382434425511
```



# Validation Results (ONNX)

**Dataset 0**
```bash
[INFO] Evaluation Criteria: YOLOv8
[INFO] The name of the dataset: Dataset_Type_0
[INFO] Precision = 0.9870509429428476
[INFO] Recall = 0.9681209964378099
[INFO] mAP@0.5 = 0.9948627946127946
[INFO] mAP@0.5:0.95 = 0.8835221482148936
```
**Dataset 1**
```bash
[INFO] Evaluation Criteria: YOLOv8
[INFO] The name of the dataset: Dataset_Type_1
[INFO] Precision = 0.9951513000455711
[INFO] Recall = 1.0
[INFO] mAP@0.5 = 0.995
[INFO] mAP@0.5:0.95 = 0.9045463240080153
```
**Dataset 2**
```bash
[INFO] Evaluation Criteria: YOLOv8
[INFO] The name of the dataset: Dataset_Type_2
[INFO] Precision = 0.9694093325670848
[INFO] Recall = 0.9754387998143003
[INFO] mAP@0.5 = 0.9937606731643429
[INFO] mAP@0.5:0.95 = 0.8267212538523665
```
**Dataset 3**
```bash
[INFO] Evaluation Criteria: YOLOv8
[INFO] The name of the dataset: Dataset_Type_3
[INFO] Precision = 0.9813162211216122
[INFO] Recall = 0.9833111895034636
[INFO] mAP@0.5 = 0.995
[INFO] mAP@0.5:0.95 = 0.8580565242042372
```
**Dataset 4**
```bash
[INFO] Evaluation Criteria: YOLOv8
[INFO] The name of the dataset: Dataset_Type_4
[INFO] Precision = 0.9931101542689124
[INFO] Recall = 1.0
[INFO] mAP@0.5 = 0.995
[INFO] mAP@0.5:0.95 = 0.8592243727973665
```
**Dataset 5**
```bash
[INFO] Evaluation Criteria: YOLOv8
[INFO] The name of the dataset: Dataset_Type_5
[INFO] Precision = 0.997907120991472
[INFO] Recall = 1.0
[INFO] mAP@0.5 = 0.995
[INFO] mAP@0.5:0.95 = 0.9190032569339632
```

## Result:

Youtube: Coming soon ...

## Contact Info:
Roman.Parak@outlook.com

## Citation (BibTex)
```bash
@misc{RomanParak_Unity3D,
  author = {Roman Parak},
  title = {Analysis of training object detection models using different types of datasets},
  year = {2020-2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/rparak/Unity3D_Robotics_Overview}}
}
```

## License
[MIT](https://choosealicense.com/licenses/mit/)
