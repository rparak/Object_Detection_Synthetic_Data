# System (Default)
import sys
sys.path.append('..')
#   ../Lib/Utilities/File_IO
import Lib.Utilities.File_IO as File_IO

# Load a label (annotation) from a file.
label_data = File_IO.Load('/Users/rparak/Documents/GitHub/Blender_Synthetic_Data/images/Image_00001', 'txt', ' ')

for _, labe_data_i in enumerate(label_data):
    print(labe_data_i)