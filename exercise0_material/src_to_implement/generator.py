import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import skimage
import skimage.transform
import random

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.
class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        # Define all members of your generator class object as global members here.
        # These need to include:
        # the batch size
        # the image size
        # flags for different augmentations and whether the data should be shuffled for each epoch
        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        self.file_path : str = file_path
        self.label_path : str = label_path

        self.batch_size : int = batch_size
        self.image_size : list = image_size

        self.rotation : bool = rotation
        self.mirroring : bool = mirroring
        self.shuffle : bool = shuffle

        self.data_dict = dict()

        with open(self.label_path) as json_labels:
            self.data_dict = json.load(json_labels)
        
        self.key = self.data_dict.keys()
        self.value = self.data_dict.values()
        self.list_keys = list(self.key)

        self.data_size = len(self.list_keys)

        self.epoch_iter = 0 


        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        
        #TODO: implement constructor


    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        #TODO: implement next method

        if self.shuffle: random.shuffle(self.list_keys)
        labels = list()
        images = list()
        i = 0
        while (len(labels) < self.batch_size):
            if not self.list_keys:
                i = self.regen_data()

            if self.list_keys[i] in labels:
                i += 1
                continue


            labels.append(self.data_dict[self.list_keys[i]])
            image = self.resize(np.load(self.file_path + self.list_keys[i] + ".npy"))
            self.list_keys.remove(self.list_keys[i])


            if self.mirroring or self.rotation:
                image = self.augment(image)

            images.append(image)

        return np.array(images), labels
    

    def regen_data(self):
        self.epoch_iter += 1
        if self.shuffle: 
            self.list_keys = list(self.key)
            random.shuffle(self.list_keys)
        else :
            self.list_keys = list(self.key)
        return 0


    def resize(self, image : np.ndarray):
        return skimage.transform.resize(image, self.image_size)


    def augment(self,img : np.array):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        if self.mirroring:
            type = random.randint(0, 3)
            if type == 1:
                return np.flip(img, 0)
            elif type == 2:
                return np.flip(img, 1)
            elif type == 3:
                return np.flip(np.flip(img, 0), 1)
            else:
                return img
        
        if self.rotation:
            type = random.randint(0, 3)
            if type == 1:
                return scipy.ndimage.rotate(img, 90)
            elif type == 2:
                return scipy.ndimage.rotate(img, 180)
            elif type == 3:
                return scipy.ndimage.rotate(img, 270)
            else:
                return img

        return img

    def current_epoch(self):
        return self.epoch_iter

    def class_name(self, x):
        return self.class_dict[x]
    
    def show(self):
        images, labels = self.next()
        fig, axes = plt.subplots(self.batch_size)
        for i in range(len(labels)):
            axes[i].imshow(images[i])
            axes[i].set_title(self.class_name(labels[i]))

        plt.show()

