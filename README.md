# Analysis of Training Object Detection Models using Different Types of Datasets

## Requirements:

**Software:**
```bash
Blender, Visual Studio Code with Python version >3.10 installed
```

**Supported on the following operating systems:**
```bash
Linux (recommended), macOS, Windows
```

**Python Libraries:**
```bash
See the "Installation Dependencies" section in the Readme file.
```

## Project Description:

Text ...

## Project Hierarchy:

**../Object_Detection_Synthetic_Data/CAD/**

3D models of individual objects.

**../Object_Detection_Synthetic_Data/Blender/**

Blender environment for access to synthetic data generation, robot workplace visualization, etc.

See the programmes below for more information.
```bash
$ ../src/Blender> ls
gen_object_bounding_box.py       gen_synthetic_data.py    save_image.py
gen_synthetic_background_data.py get_object_boundaries.py
```

**../Object_Detection_Synthetic_Data/src/**

The source code of the project with additional dependencies. The individual scripts for data collection, training, augmentation, validation of the results, etc. contain additional information.

The project also contains a Google Colab document that can be used to train the model in Collaboratory.
```bash
$ ../src/Training/Google_Colab> ls
YOLOv8_Train_Custom_Dataset.ipynb
```

**../Object_Detection_Synthetic_Data/Template/**

The "Template" section contains the required folders for the training and validation process. Both folders are explained below. To run the project smoothly, it is necessary to follow the structure of the individual folders.

Data
- The "Data" folder contains the data from the camera, the individual datasets and the results for each model (PyTorch, ONNX, etc.).
  
YOLO
- The "YOLO" folder contains the configuration file for training, the results of the training process and the models to be saved after training.
  
## Installation Dependencies

It will be useful for the project to create a virtual environment using Conda. Conda is an open source package management system and environment management system that runs on Windows, macOS, and Linux. Conda quickly installs, runs and updates packages and their dependencies.

**Set up a new virtual environment called {name} with python {version}**
```bash
$ ../user_name> conda create -n {name} python={version}
$ ../user_name> conda activate {name}
```

**Installation of packages needed for the project**
```bash
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
```bash
After conda env. will be activated, install pip.
$ ../user_name> conda activate {name}
$ ../user_name> conda install -c conda-forge pip

Locate the bin folder in the directory of the activated conda environment.
$ ../user_name> cd {directory_path}/bin

Installation of new packages such as Ultralytics.
$ ../user_name> pip install ultralytics
```

**Other useful commands for working with the Conda environment**
```bash
Deactivate environment.
$ ../user_name> conda deactivate

Remove environment.
$ ../user_name> conda remove -name {name} --all

To verify that the environment was removed, in your terminal window or an Anaconda Prompt, run.
$ ../user_name> conda info --envs

Rename the environment from the old name to the new one.
$ ../user_name> conda rename -n {old_name} {name_name}
```

## Data Acquisition from the PhoXi 3D Scanner M

The Harvester Python library was used for image acquisition. Harvester is a Python library that aims to make the image acquisition process in your computer vision application easy.
```bash
https://github.com/genicam/harvesters
```

**Python Support for Photoneo 3D Sensors using GenICam**
```bash
https://www.photoneo.com/support/
```

**The requirements to run the Python example with GenICam are:**
```bash
- Python 3.*
- PhoXi Control 1.8 or higher
```

**Examples located at (Windows):**
```bash
C:\Program Files\Photoneo\PhoXiControl-x.x.x\API\examples\GenTL\python
```

**Python Dependencies (packages)**
```bash
NumPy
$ ../user_name> pip3 install numpy

Open3D
$ ../user_name> pip3 install open3d
Note:  Only if 3D data processing is used.

harvesters
$ ../user_name> pip3 install harvesters
```

The program for obtaining the raw image (2D) from the Photoneo sensor can be found below. This program can also be used for other types of Photoneo scanners.

```bash
$ ../src/Camera/Collection> python scan.py 
```

After scanning the environment, we get the raw image without adjusting the contrast and brightness. To adjust the contrast {alpha} and brightness {beta} of each image, we use the histogram clip function.

A program to adjust the contrast {alpha} and brightness {beta} of the raw image can be found below:
```bash
$ ../src/Camera> python image_processing.py 
```

The results of the adjustment can be saved to the images folder using the program bellow:
```bash
$ ../src/Evaluation/Camera_Data> python save_histogram.py 
```

<p align="center">
  <img src="https://github.com/rparak/Object_Detection_Synthetic_Data//blob/main/images/Background/Image_11.png?raw=true" width="700" height="500">
</p>

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
