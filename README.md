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

## Installation Dependencies

It will be useful for the project to create a virtual environment using Conda. Conda is an open source package management system and environment management system that runs on Windows, macOS, and Linux. Conda quickly installs, runs and updates packages and their dependencies.

**Set up a new virtual environment called {name} with python {version}**
```bash
$ ..\user_name> conda create -n {name} python={version}
$ ..\user_name> conda activate {name}
```

**Installation of packages needed for the project**
```bash
OpenCV
$ ..\user_name> conda install -c conda-forge opencv

Matplotlib
$ ..\user_name> conda install -c conda-forge matplotlib

SciencePlots
$ ..\user_name> conda install -c conda-forge scienceplots

Pandas
$ ..\user_name> conda install -c conda-forge pandas

Albumentations
$ ..\user_name> conda install -c conda-forge albumentations

PyTorch, Torchvision, etc.
$ ..\user_name> conda install pytorch::pytorch torchvision torchaudio -c pytorch
or 
$ ..\user_name> conda install pytorch-nightly::pytorch torchvision torchaudio -c pytorch-nightly
```

**Installation of packages that are not available using conda-forge**
```bash
After conda env. will be activated, install pip.
$ ..\user_name> conda activate {name}
$ ..\user_name> conda install -c conda-forge pip

Locate the bin folder in the directory of the activated conda environment.
$ ..\user_name> cd {directory_path}/bin

Installation of new packages such as Ultralytics.
$ ..\user_name> pip install ultralytics
```

**Other useful commands for working with the Conda environment**
```bash
Deactivate environment.
$ ..\user_name> conda deactivate

Remove environment.
$ ..\user_name> conda remove -name {name} --all

To verify that the environment was removed, in your terminal window or an Anaconda Prompt, run.
$ ..\user_name> conda info --envs

Rename the environment from the old name to the new one.
$ ..\user_name> conda rename -n {old_name} {name_name}
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
