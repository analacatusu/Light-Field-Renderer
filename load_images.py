from PIL import Image
from sys import exit
import os
from math import sqrt
import numpy as np

""" A class to create a LF from a directory of images already provided in the project.
    The class converts the images to a grey scale for simplicity and saves them in a new directory inside the project.
    If the folder with grey images already exists this part will be skipped and only the images paths will be loaded. 
    
    The class also creates a LF from the images and saves them inside lf_mat in a 4D array.
    
    Input: a path to the folder containing the .png images
    Optionals: new_directory -> name for the new directory, default: greyscale_rectified
    Outputs:a new folder with the converted .png images 
            a list of images paths
            a 4D array in order to work with LF """


class LoadLF:
    def _create_folder(self):
        if not os.path.exists(self.path) or not os.path.isdir(self.path):
            print("Insert a valid path to a directory ... Exiting ")
            exit(1)

        parent_dir = os.path.split(self.path)  # path[0]-> pana la rectified, path[1] = rectified
        parent_dir = os.path.join(parent_dir[0], self.new_directory)

        exists = False

        if not os.path.exists(parent_dir):
            print("Creating new directory and loading new content ...")
            os.makedirs(parent_dir)
            new_path = self.path
        else:
            print("Directory already exists. Importing images paths...")
            exists = True  # indicates that the folder with the black & white images already exists
            new_path = parent_dir

        list_new_paths = []
        for file in os.listdir(new_path):
            image_path = os.path.join(new_path, file)
            print("Loading image: " + image_path)
            if exists:  # appending the paths of the black & white images
                image = Image.open(image_path)
            else:  # converting the input images to black & white and saving them inside the new folder
                img_path_split = os.path.split(image_path)
                title = img_path_split[1]
                title = "greyscale_" + title
                img_new_path = os.path.join(parent_dir, title)
                image = Image.open(image_path).convert('L')
                image = image.resize((256, 256))
                image.save(img_new_path)
            list_new_paths.append(image)

        return list_new_paths

    # internal method to calculate the measurements for the 4D array
    def _calc_measurements(self):
        kl = int(sqrt(self.length_images))
        ij = self.length_images // kl
        return kl, ij

    # method to create a 4D array to work with LF
    def _create_lf_matrix(self):

        print("Creating the 4D LF array...")

        list_images = self._create_folder()
        self.length_images = len(list_images)

        if self.length_images == 0:
            print("Something went wrong when downloading the images... Check the directory")
            exit(1)

        self._find_resolution(list_images)

        im_nr_kl, im_nr_ij = self._calc_measurements()
        lf_matrix = np.zeros((im_nr_kl, im_nr_ij, self.height_images, self.width_images), dtype=int)  # kl,ij
        index_images = 0

        # storing the images in order
        for i in range(im_nr_kl):
            for j in range(im_nr_ij):
                lf_matrix[i][j] = np.array(list_images[index_images])
                index_images += 1

        print("Done")
        return lf_matrix

    # method to get the resolution of an image
    def _find_resolution(self, list_images):
        image = list_images[0]
        self.width_images, self.height_images = image.size

    def __init__(self, path, new_directory='grey_scale_rectified'):
        self.path = path
        self.new_directory = new_directory
        self.length_images = 0
        self.width_images = 0
        self.height_images = 0
        self.lf_mat = self._create_lf_matrix()
