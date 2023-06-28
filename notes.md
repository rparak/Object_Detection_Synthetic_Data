## Synthetic Data Generation

Generating synthetic data from Blender is a process that includes creating an environment similar to the real one, as well as rendering it to make it more realistic. The data generation process only applies to the 2D area of view, as I mentioned earlier. The process of generating synthetic images includes the function to random generation of the parameters of the camera as well as the object. In the camera the random function consist from the additional noise and illumination, and in the object consist from the position and rotation within the specific limits.

Synthetic data consists of an image and a corresponding label, which is defined by a bounding box around the scanned object.

The main part to find the corresponging label of the object was to solve the equation of projection matrix 'P':

```
    P = K x [R | t],
```

where 'K' is the instrict matrix that contains the intrinsic parameters, and '[R | t]' is the extrinsic 
matrix that is combination of rotation matrix 'R' and a translation vector 't'.

The process of generating synthetic images takes approximately 17.22s per image. The time depends on the available GPU, along with the settings in Blender.

**Information about the process of synthetic data generation**
```
The files related to Blender can be found within the folder, see bellow.
$ ../Blender> ls
Gen_Synthetic_Data.blend Object_Bounding_Box.blend Robot_Environment_View.blend

Related Python programs used to work with Blender files can be found in the folder below.
$ ../src/Blender> ls
gen_object_bounding_box.py       gen_synthetic_data.py    save_image.py
gen_synthetic_background_data.py get_object_boundaries.py

See the individual programs with the *.py extension for more information.
```

## Data Augmentation

Generates (augments) data from a small image dataset is simple process that includes augmentation of both image and label data. The process of generating augmented images takes approximately 0.185s per image.

We use the "Albumentations" library to generate augmented data. Information about the library (transformation functions, etc.) can be found here:

    [http://albumentations.ai](http://albumentations.ai)

**The transformation declaration used to augment the image/bounding box**
```
transformation = A.Compose([A.Affine(translate_px={'x': (-10, 10), 'y': (-10, 10)}, p = 0.75),
                            A.ColorJitter(brightness=(0.25, 1.5), contrast=(0.25, 1.5), saturation=(0.1, 1.0), 
                                            always_apply=True),
                            A.GaussianBlur(blur_limit=(5, 5), sigma_limit=(0.01, 1.0), p = 0.5),
                            A.RandomResizedCrop(height= 1544, width = 2064, scale = (0.95, 1.0), p = 0.5)], 
                            bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
```

**Information about the process of augmented data generation**
```
Augmented data can be easily generated using a Python program, see below. More information about the generation parameters can be found in the program. 
$ ../src/Augmentation> python generate.py
```

# Notes
Change {alpha}, {beta} to 'alpha', 'beta'