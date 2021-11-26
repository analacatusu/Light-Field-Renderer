import timeit

""" This file is used for evaluation purposes.
    It checks in how many seconds the renderer runs and applies the shift sum filter at different resolutions.
"""

setup256 = """\
from PIL import Image
from sys import exit
import os
from math import sqrt
import numpy as np
class LoadLF:
    def _create_folder(self):
        if not os.path.exists(self.path) or not os.path.isdir(self.path):
            print("Insert a valid path to a directory ... Exiting ")
            exit(1)

        parent_dir = os.path.split(self.path)  # path[0]-> pana la rectified, path[1] = rectified
        parent_dir = os.path.join(parent_dir[0], self.new_directory)

        exists = False

        if not os.path.exists(parent_dir):
            # print("Creating new directory and loading new content ...")
            os.makedirs(parent_dir)
            new_path = self.path
        else:
            # print("Directory already exists. Importing images paths...")
            exists = True  # indicates that the folder with the black & white images already exists
            new_path = parent_dir

        list_new_paths = []
        for file in os.listdir(new_path):
            image_path = os.path.join(new_path, file)
            # print("Loading image: " + image_path)
            if exists:  # appending the paths of the black & white images
                image = Image.open(image_path)
            else:  # converting the input images to black & white and saving them inside the new folder
                img_path_split = os.path.split(image_path)
                title = img_path_split[1]
                title = "greyscale_" + title
                img_new_path = os.path.join(parent_dir, title)
                # image = Image.open(image_path)
                image = Image.open(image_path).convert('L')
                # if 'chess' in img_path_split[0]:
                image = image.resize((256, 256))
                # else:
                #     image = image.resize((256, 256))
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

        # print("Creating the 4D LF array...")

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

        # print("Done")
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
        
        
path_chess = 'chess'
images_chess = LoadLF(path_chess, new_directory="grey_scale_rectified_chess256")
"""

setup300 = """\
from PIL import Image
from sys import exit
import os
from math import sqrt
import numpy as np
class LoadLF:
    def _create_folder(self):
        if not os.path.exists(self.path) or not os.path.isdir(self.path):
            print("Insert a valid path to a directory ... Exiting ")
            exit(1)

        parent_dir = os.path.split(self.path)  # path[0]-> pana la rectified, path[1] = rectified
        parent_dir = os.path.join(parent_dir[0], self.new_directory)

        exists = False

        if not os.path.exists(parent_dir):
            # print("Creating new directory and loading new content ...")
            os.makedirs(parent_dir)
            new_path = self.path
        else:
            # print("Directory already exists. Importing images paths...")
            exists = True  # indicates that the folder with the black & white images already exists
            new_path = parent_dir

        list_new_paths = []
        for file in os.listdir(new_path):
            image_path = os.path.join(new_path, file)
            # print("Loading image: " + image_path)
            if exists:  # appending the paths of the black & white images
                image = Image.open(image_path)
            else:  # converting the input images to black & white and saving them inside the new folder
                img_path_split = os.path.split(image_path)
                title = img_path_split[1]
                title = "greyscale_" + title
                img_new_path = os.path.join(parent_dir, title)
                # image = Image.open(image_path)
                image = Image.open(image_path).convert('L')
                # if 'chess' in img_path_split[0]:
                image = image.resize((300, 300))
                # else:
                #     image = image.resize((256, 256))
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

        # print("Creating the 4D LF array...")

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

        # print("Done")
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
        
        
path_chess = 'chess'
images_chess = LoadLF(path_chess, new_directory="grey_scale_rectified_chess300")
"""

setup400 = """\
from PIL import Image
from sys import exit
import os
from math import sqrt
import numpy as np
class LoadLF:
    def _create_folder(self):
        if not os.path.exists(self.path) or not os.path.isdir(self.path):
            print("Insert a valid path to a directory ... Exiting ")
            exit(1)

        parent_dir = os.path.split(self.path)  # path[0]-> pana la rectified, path[1] = rectified
        parent_dir = os.path.join(parent_dir[0], self.new_directory)

        exists = False

        if not os.path.exists(parent_dir):
            # print("Creating new directory and loading new content ...")
            os.makedirs(parent_dir)
            new_path = self.path
        else:
            # print("Directory already exists. Importing images paths...")
            exists = True  # indicates that the folder with the black & white images already exists
            new_path = parent_dir

        list_new_paths = []
        for file in os.listdir(new_path):
            image_path = os.path.join(new_path, file)
            # print("Loading image: " + image_path)
            if exists:  # appending the paths of the black & white images
                image = Image.open(image_path)
            else:  # converting the input images to black & white and saving them inside the new folder
                img_path_split = os.path.split(image_path)
                title = img_path_split[1]
                title = "greyscale_" + title
                img_new_path = os.path.join(parent_dir, title)
                # image = Image.open(image_path)
                image = Image.open(image_path).convert('L')
                # if 'chess' in img_path_split[0]:
                image = image.resize((400, 400))
                # else:
                #     image = image.resize((256, 256))
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

        # print("Creating the 4D LF array...")

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

        # print("Done")
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
        
        
path_chess = 'chess'
images_chess = LoadLF(path_chess, new_directory="grey_scale_rectified_chess400")
"""


print("running LFRenderer")
print('=' * 40)

renderer300 = """\
import numpy as np
import matplotlib.pyplot as plt
from sys import exit
import scipy.interpolate as sp
from PIL import Image

class LFRenderer:

    # internal method used to multiply the rotation matrix with the rays directions
    def _multiply_matrices(self, matrix2):
        rot_matrix = self._rotation_matrix_z()
        row1, row2, row3 = rot_matrix[0, :], rot_matrix[1, :], rot_matrix[2, :]

        # reshape so that the multiply function works
        row1 = np.reshape(row1, (3, 1))
        row2 = np.reshape(row2, (3, 1))
        result1 = np.multiply(row1, matrix2)
        sum1 = result1.sum(axis=0)
        result2 = np.multiply(row2, matrix2)
        sum2 = result2.sum(axis=0)
        ray_directions = np.vstack([sum1, sum2])
        return ray_directions

    # internal method that returns a rotation matrix about the z axis
    def _rotation_matrix_z(self):
        theta = np.radians(self.camera_rot)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        return rot

    # internal method that returns the rays that describe the LF
    def _create_rays(self, ray_directions):
        # distances from the camera to the st and uv planes
        st_dist = 0 - self.camera_pos.item(2)
        uv_dist = st_dist + self.distance

        cam_pos_2 = np.array([[self.camera_pos.item(0)], [self.camera_pos.item(1)]])

        # intersection of the ray directions and the st and uv planes in order to create rays
        st = cam_pos_2 + st_dist * ray_directions
        uv = cam_pos_2 + uv_dist * ray_directions
        dim1, dim2 = st.shape
        ones = np.ones((1, dim2))
        row1_st = st[0, :]
        row2_st = st[1, :]
        row1_uv = uv[0, :]
        row2_uv = uv[1, :]
        row1_ones = ones[0, :]
        # stack the rays in an array
        rays = np.matrix([row1_st, row2_st, row1_uv, row2_uv, row1_ones])
        return rays

    # internal method that calculates some grid points in order to compose the LF rays
    def _calculate_meshgrid(self):
        # check whether the resolution for the LF is alright
        # width, height = self.resolution_image
        #
        # if width < self.resolution_lf or height < self.resolution_lf:
        #     print("LF resolution is too high .. Exiting")
        #     exit(1)

        # make resolution number of points between -0.6 and 0.6
        self.ray_vec = np.linspace(-0.6, 0.6, self.resolution_lf)

        # create the grid with the points
        ray_y, ray_x = np.meshgrid(self.ray_vec, self.ray_vec)
        size = ray_x.size

        # we use xyz although we fixed z so we need that coordinate too
        ones = np.ones(size)

        # stack the points in an array
        positions = np.vstack([ray_x.ravel(), ray_y.ravel(), ones])

        # plt.plot(ray_x, ray_y, 'ro', markersize=1)
        # plt.show()
        return positions

    def _calculate_rays(self):
        # calculate points to use for the LF
        ray_dir = self._calculate_meshgrid()
        # rotate them according to the camera rotation
        rays_rotated = self._multiply_matrices(ray_dir)
        # create the LF rays in st, uv planes
        rays = self._create_rays(rays_rotated)
        return rays

    # internal function to create an unoptimized plenoptic intrinsic matrix H
    # process taken from "Decoding, Calibration and Rectification for Lenselet-Based Plenoptic Cameras"
    # written by Dansereau et al
    def _create_intrinsic_matrix(self):
        h_intrinsic = np.matrix([[0.008, 0, 0, 0, -0.036],
                                 [0, 0.008, 0, 0, -0.036],  # 0.036 0.036  0.08 -1.2
                                 [0, 0, 0.055, 0, -7],   # 0.055 0.055 # up down
                                 [0, 0, 0, 0.055, -7],   # -7 -7 # left right
                                 [0, 0, 0, 0, 1]])
        return h_intrinsic

    def _calculate_continuous_indices(self):
        # in case we are out of range in the intrinsic matrix
        # comment this line when computing min/max value
        self._rearrange_h_intrinsic()

        h_inverse = np.linalg.inv(self.h_intrinsic)
        n_continuous = h_inverse * self.lf_rays

        return n_continuous

    def _create_array_of_points(self, size):
        points = np.arange(1, size + 1)
        return points

    def _rearrange_h_intrinsic(self):
        if self.h_intrinsic[2, 4] < self.min_value:
            self.h_intrinsic[2, 4] = self.min_value
        if self.h_intrinsic[3, 4] < self.min_value:
            self.h_intrinsic[3, 4] = self.min_value
        if self.h_intrinsic[2, 4] > self.max_value:
            self.h_intrinsic[2, 4] = self.max_value
        if self.h_intrinsic[3, 4] > self.max_value:
            self.h_intrinsic[3, 4] = self.max_value

    def interpolate(self, lf_mat):
        lf_mat_shape = lf_mat.shape
        fst, snd, third, forth = lf_mat_shape
        fst_array = self._create_array_of_points(fst)
        snd_array = self._create_array_of_points(snd)
        third_array = self._create_array_of_points(third)
        forth_array = self._create_array_of_points(forth)
        points = np.array((fst_array, snd_array, third_array, forth_array))

        interp_func = sp.RegularGridInterpolator(points, lf_mat, bounds_error=False, fill_value=1.)
        # interp_func = sp.RegularGridInterpolator(points, lf_mat)  # uncomment this function for computing min/max value

        points_to_interp = (self.continuous_indices[0], self.continuous_indices[1], self.continuous_indices[2],
                            self.continuous_indices[3])
        try:
            outframe = interp_func(points_to_interp)
        except:
            print("There was a problem with the interpolation")
            # uncomment next 2 lines for computing min/max value
            # print(self.h_intrinsic[2, 4])
            # print(self.h_intrinsic[3, 4])
            exit(1)

        try:
            outframe_arranged = np.array(outframe).reshape(self.resolution_lf, self.resolution_lf)
        except:
            print("There was a problem with the reshaping of the light field")
            exit(1)

        outframe_arranged = outframe_arranged.astype(int)
        lf_im = Image.fromarray((outframe_arranged).astype(np.uint8))
        lf_im = lf_im.convert('L')
        # lf_im.show()

        return lf_im

    def recalculate_interpolation(self, factor=0., up_down=False, reset=False, mouse=False, lr=0., ud=0.):
        if reset:
            self.h_intrinsic[2, 4] = -7
            self.h_intrinsic[3, 4] = -7

        if mouse:
            self.h_intrinsic[2, 4] = ud
            self.h_intrinsic[3, 4] = lr

        if up_down:
            self.h_intrinsic[2, 4] = self.h_intrinsic[2, 4] + factor
        else:
            self.h_intrinsic[3, 4] = self.h_intrinsic[3, 4] + factor

        self.continuous_indices = self._calculate_continuous_indices()
        lf_im = self.interpolate(self.lf_mat)
        return lf_im

    def compute_min_value(self, factor=0.2):
        while True:
            self.h_intrinsic[2, 4] = self.h_intrinsic[2, 4] - factor
            self.h_intrinsic[3, 4] = self.h_intrinsic[3, 4] - factor
            print(self.h_intrinsic)
            self.continuous_indices = self._calculate_continuous_indices()
            lf_im =self.interpolate(self.lf_mat)

    def compute_max_value(self, factor=0.2):
        while True:
            self.h_intrinsic[2, 4] = self.h_intrinsic[2, 4] + factor
            self.h_intrinsic[3, 4] = self.h_intrinsic[3, 4] + factor
            print(self.h_intrinsic)
            self.continuous_indices = self._calculate_continuous_indices()
            lf_im = self.interpolate(self.lf_mat)

    def __init__(self, resolution_image, resolution_lf, camera_pos, camera_rot, distance, lf_mat):
        self.resolution_image = resolution_image
        self.resolution_lf = resolution_lf
        self.camera_pos = camera_pos
        self.camera_rot = camera_rot
        self.distance = distance
        self.ray_vec = 0
        self.min_value = -9.2
        self.max_value = -5
        self.lf_rays = self._calculate_rays()
        self.lf_mat = lf_mat
        self.h_intrinsic = self._create_intrinsic_matrix()
        self.continuous_indices = self._calculate_continuous_indices()
        self.lf_image = self.interpolate(lf_mat)


resolution_im = (800, 800)
resolution_lf = 300
cam_pos = np.array((0, 0, 0))
cam_rot = 0
distance = 8
renderer = LFRenderer(resolution_im, resolution_lf, cam_pos, cam_rot, distance, images_chess.lf_mat)
"""

renderer400 = """\
import numpy as np
import matplotlib.pyplot as plt
from sys import exit
import scipy.interpolate as sp
from PIL import Image

class LFRenderer:

    # internal method used to multiply the rotation matrix with the rays directions
    def _multiply_matrices(self, matrix2):
        rot_matrix = self._rotation_matrix_z()
        row1, row2, row3 = rot_matrix[0, :], rot_matrix[1, :], rot_matrix[2, :]

        # reshape so that the multiply function works
        row1 = np.reshape(row1, (3, 1))
        row2 = np.reshape(row2, (3, 1))
        result1 = np.multiply(row1, matrix2)
        sum1 = result1.sum(axis=0)
        result2 = np.multiply(row2, matrix2)
        sum2 = result2.sum(axis=0)
        ray_directions = np.vstack([sum1, sum2])
        return ray_directions

    # internal method that returns a rotation matrix about the z axis
    def _rotation_matrix_z(self):
        theta = np.radians(self.camera_rot)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        return rot

    # internal method that returns the rays that describe the LF
    def _create_rays(self, ray_directions):
        # distances from the camera to the st and uv planes
        st_dist = 0 - self.camera_pos.item(2)
        uv_dist = st_dist + self.distance

        cam_pos_2 = np.array([[self.camera_pos.item(0)], [self.camera_pos.item(1)]])

        # intersection of the ray directions and the st and uv planes in order to create rays
        st = cam_pos_2 + st_dist * ray_directions
        uv = cam_pos_2 + uv_dist * ray_directions
        dim1, dim2 = st.shape
        ones = np.ones((1, dim2))
        row1_st = st[0, :]
        row2_st = st[1, :]
        row1_uv = uv[0, :]
        row2_uv = uv[1, :]
        row1_ones = ones[0, :]
        # stack the rays in an array
        rays = np.matrix([row1_st, row2_st, row1_uv, row2_uv, row1_ones])
        return rays

    # internal method that calculates some grid points in order to compose the LF rays
    def _calculate_meshgrid(self):
        # check whether the resolution for the LF is alright
        # width, height = self.resolution_image
        #
        # if width < self.resolution_lf or height < self.resolution_lf:
        #     print("LF resolution is too high .. Exiting")
        #     exit(1)

        # make resolution number of points between -0.6 and 0.6
        self.ray_vec = np.linspace(-0.6, 0.6, self.resolution_lf)

        # create the grid with the points
        ray_y, ray_x = np.meshgrid(self.ray_vec, self.ray_vec)
        size = ray_x.size

        # we use xyz although we fixed z so we need that coordinate too
        ones = np.ones(size)

        # stack the points in an array
        positions = np.vstack([ray_x.ravel(), ray_y.ravel(), ones])

        # plt.plot(ray_x, ray_y, 'ro', markersize=1)
        # plt.show()
        return positions

    def _calculate_rays(self):
        # calculate points to use for the LF
        ray_dir = self._calculate_meshgrid()
        # rotate them according to the camera rotation
        rays_rotated = self._multiply_matrices(ray_dir)
        # create the LF rays in st, uv planes
        rays = self._create_rays(rays_rotated)
        return rays

    # internal function to create an unoptimized plenoptic intrinsic matrix H
    # process taken from "Decoding, Calibration and Rectification for Lenselet-Based Plenoptic Cameras"
    # written by Dansereau et al
    def _create_intrinsic_matrix(self):
        h_intrinsic = np.matrix([[0.008, 0, 0, 0, -0.036],
                                 [0, 0.008, 0, 0, -0.036],  # 0.036 0.036  0.08 -1.2
                                 [0, 0, 0.055, 0, -7],   # 0.055 0.055 # up down
                                 [0, 0, 0, 0.055, -7],   # -7 -7 # left right
                                 [0, 0, 0, 0, 1]])
        return h_intrinsic

    def _calculate_continuous_indices(self):
        # in case we are out of range in the intrinsic matrix
        # comment this line when computing min/max value
        self._rearrange_h_intrinsic()

        h_inverse = np.linalg.inv(self.h_intrinsic)
        n_continuous = h_inverse * self.lf_rays

        return n_continuous

    def _create_array_of_points(self, size):
        points = np.arange(1, size + 1)
        return points

    def _rearrange_h_intrinsic(self):
        if self.h_intrinsic[2, 4] < self.min_value:
            self.h_intrinsic[2, 4] = self.min_value
        if self.h_intrinsic[3, 4] < self.min_value:
            self.h_intrinsic[3, 4] = self.min_value
        if self.h_intrinsic[2, 4] > self.max_value:
            self.h_intrinsic[2, 4] = self.max_value
        if self.h_intrinsic[3, 4] > self.max_value:
            self.h_intrinsic[3, 4] = self.max_value

    def interpolate(self, lf_mat):
        lf_mat_shape = lf_mat.shape
        fst, snd, third, forth = lf_mat_shape
        fst_array = self._create_array_of_points(fst)
        snd_array = self._create_array_of_points(snd)
        third_array = self._create_array_of_points(third)
        forth_array = self._create_array_of_points(forth)
        points = np.array((fst_array, snd_array, third_array, forth_array))

        interp_func = sp.RegularGridInterpolator(points, lf_mat, bounds_error=False, fill_value=1.)
        # interp_func = sp.RegularGridInterpolator(points, lf_mat)  # uncomment this function for computing min/max value

        points_to_interp = (self.continuous_indices[0], self.continuous_indices[1], self.continuous_indices[2],
                            self.continuous_indices[3])
        try:
            outframe = interp_func(points_to_interp)
        except:
            print("There was a problem with the interpolation")
            # uncomment next 2 lines for computing min/max value
            # print(self.h_intrinsic[2, 4])
            # print(self.h_intrinsic[3, 4])
            exit(1)

        try:
            outframe_arranged = np.array(outframe).reshape(self.resolution_lf, self.resolution_lf)
        except:
            print("There was a problem with the reshaping of the light field")
            exit(1)

        outframe_arranged = outframe_arranged.astype(int)
        lf_im = Image.fromarray((outframe_arranged).astype(np.uint8))
        lf_im = lf_im.convert('L')
        # lf_im.show()

        return lf_im

    def recalculate_interpolation(self, factor=0., up_down=False, reset=False, mouse=False, lr=0., ud=0.):
        if reset:
            self.h_intrinsic[2, 4] = -7
            self.h_intrinsic[3, 4] = -7

        if mouse:
            self.h_intrinsic[2, 4] = ud
            self.h_intrinsic[3, 4] = lr

        if up_down:
            self.h_intrinsic[2, 4] = self.h_intrinsic[2, 4] + factor
        else:
            self.h_intrinsic[3, 4] = self.h_intrinsic[3, 4] + factor

        self.continuous_indices = self._calculate_continuous_indices()
        lf_im = self.interpolate(self.lf_mat)
        return lf_im

    def compute_min_value(self, factor=0.2):
        while True:
            self.h_intrinsic[2, 4] = self.h_intrinsic[2, 4] - factor
            self.h_intrinsic[3, 4] = self.h_intrinsic[3, 4] - factor
            print(self.h_intrinsic)
            self.continuous_indices = self._calculate_continuous_indices()
            lf_im =self.interpolate(self.lf_mat)

    def compute_max_value(self, factor=0.2):
        while True:
            self.h_intrinsic[2, 4] = self.h_intrinsic[2, 4] + factor
            self.h_intrinsic[3, 4] = self.h_intrinsic[3, 4] + factor
            print(self.h_intrinsic)
            self.continuous_indices = self._calculate_continuous_indices()
            lf_im = self.interpolate(self.lf_mat)

    def __init__(self, resolution_image, resolution_lf, camera_pos, camera_rot, distance, lf_mat):
        self.resolution_image = resolution_image
        self.resolution_lf = resolution_lf
        self.camera_pos = camera_pos
        self.camera_rot = camera_rot
        self.distance = distance
        self.ray_vec = 0
        self.min_value = -9.2
        self.max_value = -5
        self.lf_rays = self._calculate_rays()
        self.lf_mat = lf_mat
        self.h_intrinsic = self._create_intrinsic_matrix()
        self.continuous_indices = self._calculate_continuous_indices()
        self.lf_image = self.interpolate(lf_mat)


resolution_im = (800, 800)
resolution_lf = 400
cam_pos = np.array((0, 0, 0))
cam_rot = 0
distance = 8
renderer = LFRenderer(resolution_im, resolution_lf, cam_pos, cam_rot, distance, images_chess.lf_mat)
"""


renderer500 = """\
import numpy as np
import matplotlib.pyplot as plt
from sys import exit
import scipy.interpolate as sp
from PIL import Image

class LFRenderer:

    # internal method used to multiply the rotation matrix with the rays directions
    def _multiply_matrices(self, matrix2):
        rot_matrix = self._rotation_matrix_z()
        row1, row2, row3 = rot_matrix[0, :], rot_matrix[1, :], rot_matrix[2, :]

        # reshape so that the multiply function works
        row1 = np.reshape(row1, (3, 1))
        row2 = np.reshape(row2, (3, 1))
        result1 = np.multiply(row1, matrix2)
        sum1 = result1.sum(axis=0)
        result2 = np.multiply(row2, matrix2)
        sum2 = result2.sum(axis=0)
        ray_directions = np.vstack([sum1, sum2])
        return ray_directions

    # internal method that returns a rotation matrix about the z axis
    def _rotation_matrix_z(self):
        theta = np.radians(self.camera_rot)
        c, s = np.cos(theta), np.sin(theta)
        rot = np.array(((c, -s, 0), (s, c, 0), (0, 0, 1)))
        return rot

    # internal method that returns the rays that describe the LF
    def _create_rays(self, ray_directions):
        # distances from the camera to the st and uv planes
        st_dist = 0 - self.camera_pos.item(2)
        uv_dist = st_dist + self.distance

        cam_pos_2 = np.array([[self.camera_pos.item(0)], [self.camera_pos.item(1)]])

        # intersection of the ray directions and the st and uv planes in order to create rays
        st = cam_pos_2 + st_dist * ray_directions
        uv = cam_pos_2 + uv_dist * ray_directions
        dim1, dim2 = st.shape
        ones = np.ones((1, dim2))
        row1_st = st[0, :]
        row2_st = st[1, :]
        row1_uv = uv[0, :]
        row2_uv = uv[1, :]
        row1_ones = ones[0, :]
        # stack the rays in an array
        rays = np.matrix([row1_st, row2_st, row1_uv, row2_uv, row1_ones])
        return rays

    # internal method that calculates some grid points in order to compose the LF rays
    def _calculate_meshgrid(self):
        # check whether the resolution for the LF is alright
        # width, height = self.resolution_image
        #
        # if width < self.resolution_lf or height < self.resolution_lf:
        #     print("LF resolution is too high .. Exiting")
        #     exit(1)

        # make resolution number of points between -0.6 and 0.6
        self.ray_vec = np.linspace(-0.6, 0.6, self.resolution_lf)

        # create the grid with the points
        ray_y, ray_x = np.meshgrid(self.ray_vec, self.ray_vec)
        size = ray_x.size

        # we use xyz although we fixed z so we need that coordinate too
        ones = np.ones(size)

        # stack the points in an array
        positions = np.vstack([ray_x.ravel(), ray_y.ravel(), ones])

        # plt.plot(ray_x, ray_y, 'ro', markersize=1)
        # plt.show()
        return positions

    def _calculate_rays(self):
        # calculate points to use for the LF
        ray_dir = self._calculate_meshgrid()
        # rotate them according to the camera rotation
        rays_rotated = self._multiply_matrices(ray_dir)
        # create the LF rays in st, uv planes
        rays = self._create_rays(rays_rotated)
        return rays

    # internal function to create an unoptimized plenoptic intrinsic matrix H
    # process taken from "Decoding, Calibration and Rectification for Lenselet-Based Plenoptic Cameras"
    # written by Dansereau et al
    def _create_intrinsic_matrix(self):
        h_intrinsic = np.matrix([[0.008, 0, 0, 0, -0.036],
                                 [0, 0.008, 0, 0, -0.036],  # 0.036 0.036  0.08 -1.2
                                 [0, 0, 0.055, 0, -7],   # 0.055 0.055 # up down
                                 [0, 0, 0, 0.055, -7],   # -7 -7 # left right
                                 [0, 0, 0, 0, 1]])
        return h_intrinsic

    def _calculate_continuous_indices(self):
        # in case we are out of range in the intrinsic matrix
        # comment this line when computing min/max value
        self._rearrange_h_intrinsic()

        h_inverse = np.linalg.inv(self.h_intrinsic)
        n_continuous = h_inverse * self.lf_rays

        return n_continuous

    def _create_array_of_points(self, size):
        points = np.arange(1, size + 1)
        return points

    def _rearrange_h_intrinsic(self):
        if self.h_intrinsic[2, 4] < self.min_value:
            self.h_intrinsic[2, 4] = self.min_value
        if self.h_intrinsic[3, 4] < self.min_value:
            self.h_intrinsic[3, 4] = self.min_value
        if self.h_intrinsic[2, 4] > self.max_value:
            self.h_intrinsic[2, 4] = self.max_value
        if self.h_intrinsic[3, 4] > self.max_value:
            self.h_intrinsic[3, 4] = self.max_value

    def interpolate(self, lf_mat):
        lf_mat_shape = lf_mat.shape
        fst, snd, third, forth = lf_mat_shape
        fst_array = self._create_array_of_points(fst)
        snd_array = self._create_array_of_points(snd)
        third_array = self._create_array_of_points(third)
        forth_array = self._create_array_of_points(forth)
        points = np.array((fst_array, snd_array, third_array, forth_array))

        interp_func = sp.RegularGridInterpolator(points, lf_mat, bounds_error=False, fill_value=1.)
        # interp_func = sp.RegularGridInterpolator(points, lf_mat)  # uncomment this function for computing min/max value

        points_to_interp = (self.continuous_indices[0], self.continuous_indices[1], self.continuous_indices[2],
                            self.continuous_indices[3])
        try:
            outframe = interp_func(points_to_interp)
        except:
            print("There was a problem with the interpolation")
            # uncomment next 2 lines for computing min/max value
            # print(self.h_intrinsic[2, 4])
            # print(self.h_intrinsic[3, 4])
            exit(1)

        try:
            outframe_arranged = np.array(outframe).reshape(self.resolution_lf, self.resolution_lf)
        except:
            print("There was a problem with the reshaping of the light field")
            exit(1)

        outframe_arranged = outframe_arranged.astype(int)
        lf_im = Image.fromarray((outframe_arranged).astype(np.uint8))
        lf_im = lf_im.convert('L')
        # lf_im.show()

        return lf_im

    def recalculate_interpolation(self, factor=0., up_down=False, reset=False, mouse=False, lr=0., ud=0.):
        if reset:
            self.h_intrinsic[2, 4] = -7
            self.h_intrinsic[3, 4] = -7

        if mouse:
            self.h_intrinsic[2, 4] = ud
            self.h_intrinsic[3, 4] = lr

        if up_down:
            self.h_intrinsic[2, 4] = self.h_intrinsic[2, 4] + factor
        else:
            self.h_intrinsic[3, 4] = self.h_intrinsic[3, 4] + factor

        self.continuous_indices = self._calculate_continuous_indices()
        lf_im = self.interpolate(self.lf_mat)
        return lf_im

    def compute_min_value(self, factor=0.2):
        while True:
            self.h_intrinsic[2, 4] = self.h_intrinsic[2, 4] - factor
            self.h_intrinsic[3, 4] = self.h_intrinsic[3, 4] - factor
            print(self.h_intrinsic)
            self.continuous_indices = self._calculate_continuous_indices()
            lf_im =self.interpolate(self.lf_mat)

    def compute_max_value(self, factor=0.2):
        while True:
            self.h_intrinsic[2, 4] = self.h_intrinsic[2, 4] + factor
            self.h_intrinsic[3, 4] = self.h_intrinsic[3, 4] + factor
            print(self.h_intrinsic)
            self.continuous_indices = self._calculate_continuous_indices()
            lf_im = self.interpolate(self.lf_mat)

    def __init__(self, resolution_image, resolution_lf, camera_pos, camera_rot, distance, lf_mat):
        self.resolution_image = resolution_image
        self.resolution_lf = resolution_lf
        self.camera_pos = camera_pos
        self.camera_rot = camera_rot
        self.distance = distance
        self.ray_vec = 0
        self.min_value = -9.2
        self.max_value = -5
        self.lf_rays = self._calculate_rays()
        self.lf_mat = lf_mat
        self.h_intrinsic = self._create_intrinsic_matrix()
        self.continuous_indices = self._calculate_continuous_indices()
        self.lf_image = self.interpolate(lf_mat)


resolution_im = (800, 800)
resolution_lf = 500
cam_pos = np.array((0, 0, 0))
cam_rot = 0
distance = 8
renderer = LFRenderer(resolution_im, resolution_lf, cam_pos, cam_rot, distance, images_chess.lf_mat)
"""
print('running LFFilters')
print('=' * 40)

filters = """\
import numpy as np
import scipy.interpolate as sp
from sys import exit
from PIL import Image
import math


class LFFilters:

    # def _shift_back_in_range(self, vec):
    #     if vec.ndim != 2:
    #         print("Incorrect dimension for shift.. Exiting")
    #         exit(1)
    #     for k in range(vec.shape[0]):
    #         for l in range(vec.shape[1]):
    #             if vec[k][l] > (vec.shape[0] - 1) or vec[k][l] < 1.:
    #                 vec[k][l] = 0.
    #     return vec

    def _create_array_of_points(self, start, end):
        points = np.arange(start, end)
        return points

    # def _create_array_of_points_symm(self, size):
    #     points = np.zeros(size).astype(int)
    #     val = size-1
    #     for i in range(size):
    #         if i == 0:
    #             points[i] = 0
    #         else:
    #             points[i] = val
    #             val -= 1
    #     return points

    def create_filter_shift_sum(self, slope):
        # print(slope)
        tv_slope = slope
        su_slope = slope
        fst_dim, snd_dim, thrd_dim, forth_dim = self.lf_mat.shape
        uu, vv = np.meshgrid(self._create_array_of_points(0, thrd_dim), self._create_array_of_points(0, forth_dim))
        v_vec = np.linspace(-0.3, 0.3, fst_dim) * tv_slope * fst_dim
        u_vec = np.linspace(-0.3, 0.3, snd_dim) * su_slope * snd_dim

        points = np.array((self._create_array_of_points(0, thrd_dim), self._create_array_of_points(0, forth_dim)))

        for i in range(fst_dim):
            # print(".", end="")
            v_offset = v_vec[i]
            for j in range(snd_dim):
                u_offset = u_vec[j]
                curr_slice = self.lf_mat[i][j]
                points_to_interp = ((vv + v_offset), (uu + u_offset))
                interp_func = sp.RegularGridInterpolator(points, curr_slice, bounds_error=False, fill_value=0.)
                curr_slice_shift = interp_func(points_to_interp)
                self.lf_mat[i][j] = curr_slice_shift

        sum_along_col = np.sum(self.lf_mat, axis=1)
        sum_along_row = np.sum(sum_along_col, axis=0)

        # img_out = np.divide(sum_along_row[0], sum_along_row[1])
        img_out = np.divide(sum_along_row, fst_dim*snd_dim)

        # img_out_arranged = np.array(img_out).reshape(400, 400)
        lf_im = Image.fromarray((img_out).astype(np.uint8))
        lf_im = lf_im.convert('L')
        lf_im = lf_im.crop((self.px_crop, self.px_crop, self.dim_im-self.px_crop, self.dim_im-self.px_crop))

        return lf_im

    def __init__(self, lf_mat, slopes=[-0.45]):
        self.lf_mat = lf_mat
        self.dim_im = lf_mat.shape[3]
        self.px_crop = 10
        # print(lf_mat.shape)
        self.max_uv = 300
        # self.slope = -0.6 * self.max_uv/400
        self.slope = slopes
        self.aspect_4D = np.ones(4)
        self.planar_bw = 0.06
        self.list_img_filter = [self.create_filter_shift_sum(slope=self.slope[i]) for i in range(len(slopes))]


slopes = [-0.45, -1.6, 2]
filters_im = LFFilters(images_chess.lf_mat, slopes=slopes)
"""

print("running gif")
print('=' * 40)

gif256 = """\
from renderer import LFRenderer
import numpy as np
import PIL
import sys
import os

resolution_im = (800, 800)
resolution_lf = 300
cam_pos = np.array((0, 0, 0))
cam_rot = 0
distance = 8

renderer = LFRenderer(resolution_im, resolution_lf, cam_pos, cam_rot, distance, images_chess.lf_mat)

images = []
images.append(renderer.lf_image)
steps = 6
image_nr = 0

for step in range(steps):
    lf_im = renderer.recalculate_interpolation(factor=-0.2)
    lf_im = lf_im.convert("P")
    images.append(lf_im)

for step in range(steps):
    lf_im = renderer.recalculate_interpolation(factor=0.2, up_down=True)
    lf_im = lf_im.convert("P")
    images.append(lf_im)

for step in range(steps):
    lf_im = renderer.recalculate_interpolation(factor=0.2)
    lf_im = lf_im.convert("P")
    images.append(lf_im)

for step in range(steps):
    lf_im = renderer.recalculate_interpolation(factor=-0.2, up_down=True)
    lf_im = lf_im.convert("P")
    images.append(lf_im)

images[0].save('out.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=70, loop=0)

"""

gif300 = """\
from renderer import LFRenderer
import numpy as np
import PIL
import sys
import os

resolution_im = (800, 800)
resolution_lf = 300
cam_pos = np.array((0, 0, 0))
cam_rot = 0
distance = 8

renderer = LFRenderer(resolution_im, resolution_lf, cam_pos, cam_rot, distance, images_chess.lf_mat)

images = []
images.append(renderer.lf_image)
steps = 6
image_nr = 0

for step in range(steps):
    lf_im = renderer.recalculate_interpolation(factor=-0.2)
    lf_im = lf_im.convert("P")
    images.append(lf_im)

for step in range(steps):
    lf_im = renderer.recalculate_interpolation(factor=0.2, up_down=True)
    lf_im = lf_im.convert("P")
    images.append(lf_im)

for step in range(steps):
    lf_im = renderer.recalculate_interpolation(factor=0.2)
    lf_im = lf_im.convert("P")
    images.append(lf_im)

for step in range(steps):
    lf_im = renderer.recalculate_interpolation(factor=-0.2, up_down=True)
    lf_im = lf_im.convert("P")
    images.append(lf_im)

images[0].save('out.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=70, loop=0)

"""

gif400 = """\
from renderer import LFRenderer
import numpy as np
import PIL
import sys
import os

resolution_im = (800, 800)
resolution_lf = 300
cam_pos = np.array((0, 0, 0))
cam_rot = 0
distance = 8

renderer = LFRenderer(resolution_im, resolution_lf, cam_pos, cam_rot, distance, images_chess.lf_mat)

images = []
images.append(renderer.lf_image)
steps = 6
image_nr = 0

for step in range(steps):
    lf_im = renderer.recalculate_interpolation(factor=-0.2)
    lf_im = lf_im.convert("P")
    images.append(lf_im)

for step in range(steps):
    lf_im = renderer.recalculate_interpolation(factor=0.2, up_down=True)
    lf_im = lf_im.convert("P")
    images.append(lf_im)

for step in range(steps):
    lf_im = renderer.recalculate_interpolation(factor=0.2)
    lf_im = lf_im.convert("P")
    images.append(lf_im)

for step in range(steps):
    lf_im = renderer.recalculate_interpolation(factor=-0.2, up_down=True)
    lf_im = lf_im.convert("P")
    images.append(lf_im)

images[0].save('out.gif',
               save_all=True, append_images=images[1:], optimize=False, duration=70, loop=0)

"""

result_1 = timeit.timeit(renderer300, setup256, number=1)
result_2 = timeit.timeit(renderer300, setup300, number=1)
result_3 = timeit.timeit(renderer300, setup400, number=1)
result_4 = timeit.timeit(renderer400, setup256, number=1)
result_5 = timeit.timeit(renderer400, setup300, number=1)
result_6 = timeit.timeit(renderer400, setup400, number=1)
result_7 = timeit.timeit(renderer500, setup256, number=1)
result_8 = timeit.timeit(renderer500, setup300, number=1)
result_9 = timeit.timeit(renderer500, setup400, number=1)
result_10 = timeit.timeit(filters, setup256, number=1)
result_11 = timeit.timeit(filters, setup300, number=1)
result_12 = timeit.timeit(filters, setup400, number=1)
result_13 = timeit.timeit(gif256, setup256, number=1)
result_14 = timeit.timeit(gif300, setup300, number=1)
result_15 = timeit.timeit(gif400, setup400, number=1)
result_16 = timeit.timeit(setup256, number=1)
result_17 = timeit.timeit(setup300, number=1)
result_18 = timeit.timeit(setup400, number=1)
print("LFRenderer300_256:\t{}".format(result_1))
print("LFRenderer300_300:\t{}".format(result_2))
print("LFRenderer300_400:\t{}".format(result_3))
print("LFRenderer400_256:\t{}".format(result_4))
print("LFRenderer400_300:\t{}".format(result_5))
print("LFRenderer400_400:\t{}".format(result_6))
print("LFRenderer500_256:\t{}".format(result_7))
print("LFRenderer500_300:\t{}".format(result_8))
print("LFRenderer500_400:\t{}".format(result_9))
print("LFFilters256:\t{}".format(result_10))
print("LFFilters300:\t{}".format(result_11))
print("LFFilters400:\t{}".format(result_12))
print("LFGif256:\t{}".format(result_13))
print("LFGif300:\t{}".format(result_14))
print("LFGif400:\t{}".format(result_15))
print("LFRead256:\t{}".format(result_16))
print("LFRead300:\t{}".format(result_17))
print("LFRead400:\t{}".format(result_18))

