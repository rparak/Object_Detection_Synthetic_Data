# Matplotlib (Visualization) [pip3 install matplotlib]
import matplotlib.pyplot as plt
import numpy as np

box_desired = []
box_desired.append(0.95)
box_desired.append([0.95, 0.75])
box_desired.append([0.95, 0.65])
box_desired.append(0.91)
box_desired.append(0.9)
box_predicted = []
box_predicted.append(0.9)
box_predicted.append([0.66, 0.65])
box_predicted.append([0.85, 0.56])
box_predicted.append(0.71)
box_predicted.append(0.8)
num_of_data = []
num_of_data.append(1)
num_of_data.append([1, 2])
num_of_data.append([1, 2])
num_of_data.append(1)
num_of_data.append(1)

#print(box_desired)
#print(box_predicted)
#print(num_of_data)

n_images = np.linspace(0, len(num_of_data), len(num_of_data))

fig, ax = plt.subplots()
fig.suptitle(f'The name ...', fontsize = 20)

for i, (bd_i, bp_i) in enumerate(zip(box_desired, box_predicted)):
    if isinstance(bd_i, list):
        ax.scatter(i, np.sum(bd_i)/len(bd_i), c = 'b')
    else:
        ax.scatter(i, bd_i, c = 'b')

    if isinstance(bp_i, list):
        ax.scatter(i, np.sum(bp_i)/len(bp_i), c = 'r')
    else:
        ax.scatter(i, bp_i, c = 'r')

ax.plot(n_images, [0.9] * len(n_images))
plt.show()

