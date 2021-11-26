from load_images import LoadLF
import numpy as np
import matplotlib.pyplot as plt
from sys import exit
import scipy.interpolate as sp
from PIL import Image
""" A class to calculate the rays and indices in order to display the LF 
    Inputs: resolution_image = width and height of the images 
            resolution_lf = resolution of the output
            camera_pos = camera position in xyz coordinates
            camera_rot = rotation of the camera as integer
            distance = distance between plane st and uv that describe the LF rays
    Outputs: a rendered image from the LF
    Optional: uncomment everything that is related to computing the min/max value of the field of view for the LF
              in case one ones to use the GUI with a different light field than the ones that are provided
"""


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

        # comment the following function when computing min/max value
        interp_func = sp.RegularGridInterpolator(points, lf_mat, bounds_error=False, fill_value=1.)
        # uncomment the following function for computing min/max value
        # interp_func = sp.RegularGridInterpolator(points, lf_mat)

        points_to_interp = (self.continuous_indices[0], self.continuous_indices[1], self.continuous_indices[2],
                            self.continuous_indices[3])
        try:
            outframe = interp_func(points_to_interp)
        except:
            # comment this line when computing min/max value
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
        lf_im = Image.fromarray(outframe_arranged.astype(np.uint8))
        lf_im = lf_im.convert('L')
        # lf_im.show()

        return lf_im

    # function used in the GUI to render multiple images from the LF in order to concatenate them
    # and build a 3D model
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

    # function to compute border values for the range of the LF
    def compute_min_value(self, factor=0.2):
        while True:
            self.h_intrinsic[2, 4] = self.h_intrinsic[2, 4] - factor
            self.h_intrinsic[3, 4] = self.h_intrinsic[3, 4] - factor
            print(self.h_intrinsic)
            self.continuous_indices = self._calculate_continuous_indices()
            lf_im =self.interpolate(self.lf_mat)

    # function to compute border values for the range of the LF
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


if __name__ == '__main__':
    resolution_im = (800, 800)
    resolution_lf = 400
    cam_pos = np.array((0, 0, 0))
    cam_rot = 0
    distance = 8
    path_chess = 'D:\intellij_workspace2\LFRenderer\\stanford\\rectified_chess'
    images_chess = LoadLF(path_chess, new_directory="grey_scale_rectified_chess")
    renderer = LFRenderer(resolution_im, resolution_lf, cam_pos, cam_rot, distance, images_chess.lf_mat)
    # uncomment the next two lines one at a time in order to compute the min/max values.
    # renderer.compute_min_value()
    # renderer.compute_max_value()

