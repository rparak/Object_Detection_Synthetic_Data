# System (Default)
import sys
# Pandas (Data analysis and manipulation) [pip3 install pandas]
import pandas as pd
# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
# OS (Operating system interfaces)
import os

# https://github.com/ultralytics/yolov5/issues/411
# https://github.com/ultralytics/yolov5/issues/1246
# https://medium.com/@ytlei/understand-object-detection-metrics-cacd1335ec81
# https://blog.roboflow.com/releasing-a-new-yolov3-implementation/

def main():
    df = pd.read_csv('results.csv')

    # Assign data to variables.
    # ...
    epoch = df[df.columns[0]]

    # ...
    fig, ax = plt.subplots(2, 5)
    """
    for i, df_column_name in enumerate(df.columns[1::]):
        data_strip = df_column_name.strip()
    """

    # [1.0,0.85,0.75,1.0] -> orange
    # [0.2,0.4,0.6,0.75] -> blue
    fig, ax = plt.subplots()

    ax.plot(epoch, train['box_loss'], '-', color=[0.2,0.4,0.6,0.75], linewidth=2.0, marker = 'o', ms = 5.0, mfc = [1.0,1.0,1.0], markeredgecolor = [0.2,0.4,0.6], mew = 2.5)

    # Set parameters of the visualization.
    ax.set_title(f'text ...')
    ax.set_xlabel(r'x'); ax.set_ylabel(r'y')
    #   Other dependencies
    ax.grid(linewidth = 0.75, linestyle = '--')

    # Display the result
    plt.show()

    # Display the result
    #plt.show()
    
if __name__ == '__main__':
    sys.exit(main())