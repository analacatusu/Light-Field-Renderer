import greyscale_images
import numpy as np
from PIL import Image
from glob import glob
from sys import exit
from math import sqrt


def _calculate_array_measurements(length, image):
    width, height = image.size
    kl = int(sqrt(length))
    ij = length // kl
    return width, height, kl, ij


def create_lf_matrix(list_images):
    width_im, height_im, im_nr_kl, im_nr_ij = _calculate_array_measurements(len(list_images), list_images[0])
    lf_matrix = np.zeros((im_nr_kl, im_nr_ij, height_im, width_im), dtype=int)

    index_images = 0
    for i in range(im_nr_kl):
        for j in range(im_nr_ij):
            lf_matrix[i][j] = np.array(list_images[index_images])
            index_images += 1

    return lf_matrix


if __name__ == '__main__':

    # insert path of the folder here. *.png used to extract all paths to all the .png images
    path = 'D:\intellij_workspace2\LFRenderer\\rectified\*.png'

    # list containing paths to all .png images inside the specified directory
    list_images_paths = glob(path)
    print(len(list_images_paths))

    if len(list_images_paths) == 0:
        print("Insert a valid path ... Exiting")
        exit(1)

    new_directory_name = ""  # insert here new directory name, otherwise default will be used
    list_new_images = greyscale_images.create_new_folder(list_images_paths)
    lf_mat = create_lf_matrix(list_new_images)
    r = Image.fromarray(lf_mat[0][0])
    r.show()
    r = Image.fromarray(lf_mat[16][16])
    r.show()
