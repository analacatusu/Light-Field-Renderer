from load_images import LoadLF
import numpy as np
import scipy.interpolate as sp
from PIL import Image
import os

""" A class to apply the shift sum filter to a LF. 
    The shift sum filter focuses on a region of interest in the image,
    a so-called slope, whilst the rest is blurred out.
    Inputs: lf_mat = light field matrix
            slopes = a list of slopes where to apply the filter
    Outputs: a filtered image/ filtered images
"""

DATA_DIR = "D:\intellij_workspace2\LFRenderer\\stanford"

class LFFilters:

    # helper function that is used to generate a grid of points
    def _create_array_of_points(self, start, end):
        points = np.arange(start, end)
        return points

    # applying the shift sum filter to the LF with a given slope
    def create_filter_shift_sum(self, slope):
        tv_slope = slope
        su_slope = slope
        fst_dim, snd_dim, thrd_dim, forth_dim = self.lf_mat.shape
        uu, vv = np.meshgrid(self._create_array_of_points(0, thrd_dim), self._create_array_of_points(0, forth_dim))
        v_vec = np.linspace(-0.3, 0.3, fst_dim) * tv_slope * fst_dim
        u_vec = np.linspace(-0.3, 0.3, snd_dim) * su_slope * snd_dim

        points = np.array((self._create_array_of_points(0, thrd_dim), self._create_array_of_points(0, forth_dim)))

        print("Applying shift sum filter to LF")
        print("Shifting images to depth: " + str(slope))
        print("Loading ", end="")

        for i in range(fst_dim):
            print(".", end="")
            v_offset = v_vec[i]
            for j in range(snd_dim):
                u_offset = u_vec[j]
                curr_slice = self.lf_mat[i][j]
                points_to_interp = ((vv + v_offset), (uu + u_offset))
                interp_func = sp.RegularGridInterpolator(points, curr_slice, bounds_error=False, fill_value=0.)
                curr_slice_shift = interp_func(points_to_interp)
                im = Image.fromarray(curr_slice_shift)
                self.lf_mat[i][j] = curr_slice_shift

        print(" Done")

        sum_along_col = np.sum(self.lf_mat, axis=1)
        sum_along_row = np.sum(sum_along_col, axis=0)

        img = Image.fromarray(sum_along_row)
        img_out = np.divide(sum_along_row, fst_dim*snd_dim)

        lf_im = Image.fromarray(img_out)
        lf_im = lf_im.convert('L')
        lf_im = lf_im.crop((self.px_crop, self.px_crop, self.dim_im-self.px_crop, self.dim_im-self.px_crop))

        self.save_filt(lf_im)
        # lf_im.show()
        return lf_im

    # save the filtered image inside the project
    def save_filt(self, lf_im):
        lf_im.save(f"filtered image at slope {self.slope[self.index]}.png")
        self.index += 1

    def __init__(self, lf_mat, slopes=[-0.45]):
        self.index = 0

        self.lf_mat = lf_mat
        self.dim_im = lf_mat.shape[3]
        self.px_crop = 10
        self.max_uv = 300
        self.slope = slopes
        self.aspect_4D = np.ones(4)
        self.planar_bw = 0.06
        self.list_img_filter = [self.create_filter_shift_sum(slope=self.slope[i]) for i in range(len(slopes))]


if __name__ == '__main__':
    path = os.path.join(DATA_DIR, "rectified_lego")
    # path = 'D:\intellij_workspace2\LFRenderer\\stanford\\rectified_lego'
    images = LoadLF(path, new_directory="grey_scale_rectified_lego")
    slopes = [-0.45, 1.6]
    filters = LFFilters(images.lf_mat, slopes=slopes)
