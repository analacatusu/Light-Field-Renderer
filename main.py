from load_images import LoadLF
from PIL import Image
import numpy as np
from renderer import LFRenderer
import sys
import os

DATA_DIR = "D:\intellij_workspace2\LFRenderer\\stanford"
# path_chess = 'D:\intellij_workspace2\LFRenderer\\stanford\\rectified_chess'  # insert path of the folder here.
# path_lego = 'D:\intellij_workspace2\LFRenderer\\stanford\\rectified_lego'
# path_fish_multi = 'D:\intellij_workspace2\LFRenderer\\stanford\\rectified_fish_multi'
# path_fish_eye = 'D:\intellij_workspace2\LFRenderer\\stanford\\rectified_fish_eye'
# path_fibers = 'D:\intellij_workspace2\LFRenderer\\stanford\\rectified_fibers'
# path_ball = 'D:\intellij_workspace2\LFRenderer\\stanford\\rectified_ball'

# an LFLoad object that contains the specified path, the name of the new directory,
# the list of images paths and the 4D LF array
# optionally insert new folder name here for the black & white images
images_chess = LoadLF(os.path.join(DATA_DIR,'rectified_chess'), new_directory="grey_scale_rectified_chess")
images_lego = LoadLF(os.path.join(DATA_DIR,'rectified_lego'), new_directory="grey_scale_rectified_lego")
images_fish_multi = LoadLF(os.path.join(DATA_DIR,'rectified_fish_multi'), new_directory="grey_scale_rectified_fish_multi")
images_fish_eye = LoadLF(os.path.join(DATA_DIR,'rectified_fish_eye'), new_directory="grey_scale_rectified_fish_eye")
images_fibers = LoadLF(os.path.join(DATA_DIR,'rectified_fibers'), new_directory="grey_scale_rectified_fibers")
images_ball = LoadLF(os.path.join(DATA_DIR,'rectified_ball'), new_directory="grey_scale_rectified_ball")


# accessing data in the 4D LF array and displaying it


# run with load_images2.py
# downside with generators: can't access otherwise -> we must iterate
# repetition = 0
# for element in images.lf_mat:
#     if repetition == 0 or repetition == 288:
#         im1 = Image.fromarray(element)
#         im1.show()
#         print(im1.size)
#     repetition += 1


# run with load_images.py
# im1 = Image.fromarray(images.lf_mat[0][0])
# im1.show()
# print(im1.size)
# im2 = Image.fromarray(images.lf_mat[16][16])
# im2.show()


# width, height = images.resolution
# print(width)  # 1400
# print(height)  # 800
# standard: width x height


# setting up camera
print(images_chess.lf_mat.shape)
resolution_im = (800, 800)
resolution_lf = 300
cam_pos = np.array((0, 0, 0))
cam_rot = 0
distance = 8

# renderer = LFRenderer(resolution_im, resolution_lf, cam_pos, cam_rot, distance, images_chess.lf_mat)
renderer = LFRenderer(resolution_im, resolution_lf, cam_pos, cam_rot, distance, images_lego.lf_mat)
# renderer = LFRenderer(resolution_im, resolution_lf, cam_pos, cam_rot, distance, images_chess.lf_mat)
# renderer = LFRenderer(resolution_im, resolution_lf, cam_pos, cam_rot, distance, images_fish_eye.lf_mat)
# renderer = LFRenderer(resolution_im, resolution_lf, cam_pos, cam_rot, distance, images_fish_multi.lf_mat)
# renderer = LFRenderer(resolution_im, resolution_lf, cam_pos, cam_rot, distance, images_ball.lf_mat)
print("size: {} bytes".format(sys.getsizeof(renderer)))
print()
# renderer.interpolate(images.lf_mat)
# renderer.recalculate_interpolation(mouse=True, lr=-9.2, ud=-7.593999999)
# print("done")
