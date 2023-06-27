# Analysis of Training Object Detection Models using Different Types of Datasets

<p align="center">
  <img src="https://github.com/rparak/Object_Detection_Synthetic_Data//blob/main/images/Background/Image_11.png?raw=true" width="750" height="450">
</p>

## Requirements

**Software:**
```
Blender, Visual Studio Code with Python version >3.10 installed
```

**Supported on the following operating systems:**
```
Linux (recommended), macOS, Windows
```

**Python Libraries:**
```
See the "Installation Dependencies" section in the Readme file.
```

## Project Description

The project focuses on analyzing the training of object detection models using various types of datasets, including real data, augmented data, synthetic data, and hybrid data. Consequently, we compare these approaches to train the model and determine the most effective method for object detection in real-world applications.

The experiment was conducted in a robotics lab called Industry 4.0 Cell (I4C), located at Brno University of Technology. The main hardware used for the experiment was a PhoXi 3D Scanner M. The project primarily focuses on object detection in 2D space, so we utilized only the 2D camera of the scanner in our experiment. For future expansions of the project, we can also incorporate the third dimension to fully leverage the scanner's potential.

The experiment aims to detect two types of objects (see figure):
- The **T-joint** object made of aluminum material. Figure left.
- The **Metal-Blank** object made of aluminum material. Figure right.

<p align="center">
  <img src="https://github.com/rparak/Object_Detection_Synthetic_Data//blob/main/images/Objects/Object_T_Joint.png?raw=true" width="250" height="250">
  <img src="https://github.com/rparak/Object_Detection_Synthetic_Data//blob/main/images/Objects/Object_Metal_Blank.png?raw=true" width="250" height="250">
</p>

To determine the most effective object detection method, we created six types of datasets and trained them using the YOLOv8 model (Ultralytics HUB).

| Type of the Dataset  | Short Description |
| -------------------- | ------------- |
| Type-0 | A smaller dataset that contains real monochrome images from the Photoneo scanner. |
| Type-1 | Augmented real monochrome images used to increase the size of the Type-0 dataset. |
| Type-2 | A smaller dataset that contains synthetic monochrome images from the Blender camera. |
| Type-3 | Augmented synthetic monochrome images used to increase the size of the Type-2 dataset. |
| Type-4 | A larger dataset that contains synthetic monochrome images from the Blender camera. |
| Type-5 | A hybrid dataset that contains synthetic monochrome images from the Blender camera, as well as real monochrome images from the Photoneo scanner. |

The full dataset can be found here in a shared folder on Google Drive:

[Data.zip](https://drive.google.com/file/d/1VXEg9HtX9TzuSyUX8PPbrxeeKiCVEjNV/view?usp=drive_link)

and the training results, models etc. can also be found here on Google Drive:

[YOLO.zip](https://drive.google.com/file/d/1B_kyKrbVJNvrqnOPZnZo_mWCvWOtO58t/view?usp=drive_link)

For information on object detection using the YOLOv8 model, please visit the [Ultralytics YOLOv8](https://docs.ultralytics.com/tasks/detect/) website.

More detailed information about the data acquisition process, generating synthetic data using Blender, data augmentation, training the object detection model on a custom dataset, and last but not least, the evaluation of the experiment can be found in the individual sections of the README file below.

The project was realized at the Institute of Automation and Computer Science, Brno University of Technology, Faculty of Mechanical Engineering (NETME Centre - Cybernetics and Robotics Division).

## Project Hierarchy

**../Object_Detection_Synthetic_Data/CAD/**

3D models of individual objects.

**../Object_Detection_Synthetic_Data/Blender/**

Blender environment for access to synthetic data generation, robot workplace visualization, etc.

See the programmes below for more information.
```
$ ../src/Blender> ls
gen_object_bounding_box.py       gen_synthetic_data.py    save_image.py
gen_synthetic_background_data.py get_object_boundaries.py
```

**../Object_Detection_Synthetic_Data/src/**

The source code of the project with additional dependencies. The individual scripts for data collection, training, augmentation, validation of the results, etc. contain additional information.

The project also contains a Google Colab document that can be used to train the model in Collaboratory.
```
$ ../src/Training/Google_Colab> ls
YOLOv8_Train_Custom_Dataset.ipynb
```

**../Object_Detection_Synthetic_Data/Template/**

The **Template** section contains the required folders for the training and validation process. Both folders are explained below. To run the project smoothly, it is necessary to follow the structure of the individual folders.

Data
- The **Data** folder contains the data from the camera, the individual datasets and the results for each model (PyTorch, ONNX, etc.).
  
YOLO
- The **YOLO** folder contains the configuration file for training, the results of the training process and the models to be saved after training.
  
## Data Acquisition from the PhoXi 3D Scanner M

The Harvester Python library was used for image acquisition. Harvester is a library for Python that aims to ease the process of image acquisition in computer vision applications.
```
https://github.com/genicam/harvesters
```

**Python Support for Photoneo 3D Sensors using GenICam**
```
https://www.photoneo.com/support/
```

**The requirements to run the Python example with GenICam are:**
```
- Python 3.*
- PhoXi Control 1.8 or higher
```

**Examples located at (Windows):**
```
C:\Program Files\Photoneo\PhoXiControl-x.x.x\API\examples\GenTL\python
```

**Python Dependencies (packages)**
```
NumPy
$ ../user_name> pip3 install numpy

Open3D
$ ../user_name> pip3 install open3d
Note:  Only if 3D data processing is used.

harvesters
$ ../user_name> pip3 install harvesters
```

The program for acquiring the raw 2D image from the Photoneo sensor can be found below. This program is compatible with various types of Photoneo scanners, including XS, S, M, L, and more.

```
$ ../src/Camera/Collection> python scan.py 
```

After scanning the environment, we obtain the raw image without adjusting the contrast and brightness. To adjust the contrast 'alpha' and brightness 'beta' of each image, we utilize the histogram clip function.

A program to adjust the contrast {alpha} and brightness {beta} of the raw image can be found below:
```
$ ../src/Camera> python image_processing.py 
```

The results of the adjustment can be saved to the images folder using the program bellow:
```
$ ../src/Evaluation/Camera_Data> python save_histogram.py 
```

<p align="center">
  <img src="https://github.com/rparak/Object_Detection_Synthetic_Data//blob/main/images/Evaluation/Camera_Data/Histogram_Image_00013.png?raw=true" width="750" height="450">
</p>

## Synthetic Data Generation

Text ......

## Data Augmentation

Text ......

## Train YOLOv8 Object Detection on a Custom Dataset

Text ......

## Evaluation of the Experiment

Text ......

## Installation Dependencies

It will be useful for the project to create a virtual environment using Conda. Conda is an open source package management system and environment management system that runs on Windows, macOS, and Linux. Conda quickly installs, runs and updates packages and their dependencies.

**Set up a new virtual environment called {name} with python {version}**
```
$ ../user_name> conda create -n {name} python={version}
$ ../user_name> conda activate {name}
```

**Installation of packages needed for the project**
```
OpenCV
$ ../user_name> conda install -c conda-forge opencv

Matplotlib
$ ../user_name> conda install -c conda-forge matplotlib

SciencePlots
$ ../user_name> conda install -c conda-forge scienceplots

Pandas
$ ../user_name> conda install -c conda-forge pandas

Albumentations
$ ../user_name> conda install -c conda-forge albumentations

PyTorch, Torchvision, etc.
$ ../user_name> conda install pytorch::pytorch torchvision torchaudio -c pytorch
or 
$ ../user_name> conda install pytorch-nightly::pytorch torchvision torchaudio -c pytorch-nightly
```

**Installation of packages that are not available using conda-forge**
```
After conda env. will be activated, install pip.
$ ../user_name> conda activate {name}
$ ../user_name> conda install -c conda-forge pip

Locate the bin folder in the directory of the activated conda environment.
$ ../user_name> cd {directory_path}/bin

Installation of new packages such as Ultralytics.
$ ../user_name> pip install ultralytics
```

**Other useful commands for working with the Conda environment**
```
Deactivate environment.
$ ../user_name> conda deactivate

Remove environment.
$ ../user_name> conda remove -name {name} --all

To verify that the environment was removed, in your terminal window or an Anaconda Prompt, run.
$ ../user_name> conda info --envs

Rename the environment from the old name to the new one.
$ ../user_name> conda rename -n {old_name} {name_name}
```

## Video

Youtube: Coming soon ...

## Contact Info:
Roman.Parak@outlook.com

## Citation (BibTex)
```
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
