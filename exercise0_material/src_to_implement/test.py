import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import skimage
import random


path = os.getcwd()
path_to_img = path + "/src_to_implement/data/exercise_data/"
path_to_json = path + "/src_to_implement/data/lab.json"
# img_array = np.load(path_to_img + "0.npy")
# #plt.imshow(img_array)
# #plt.show()

# key = list
# value = list
with open(path_to_json) as json_file:
    data = json.load(json_file)

# key = data.keys()
# value = data.values()

# for keys in data:
#     print(keys, "   ", data[keys])
#     print(type(keys))

labels = list()
images = np.ndarray()
keys = data.keys()

# keys, values = data.keys(), data.values()
fig, axes = plt.subplots(2)
for i in keys:
    labels.append(data[i])
    #images.append(np.load(path_to_img + str(i) + ".npy"))
    #plt.subplot(5, 2, i + 1)
    images.append(np.load(path_to_img + str(i) + ".npy"))

class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}

print(class_dict[1])

for i in range(len(labels)):
    axes[i].imshow(images[i])
    axes[i].set_title(class_dict[labels[i]])

plt.show()
# plt.imshow(np.load(path_to_img + str(2) + ".npy"))

# plt.show()

# img_array = np.load(path_to_img + "0.npy")
# #img_array = skimage.transform.resize(img_array, (256, 128, 3))
# img_array = np.flip(img_array, axis=0)
# plt.imshow(img_array)
# plt.show()

# list_keys = list(keys)
# print(list_keys[1])

