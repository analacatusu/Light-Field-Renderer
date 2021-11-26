import sys
import os
import numpy as np
from PIL import Image
from load_images import LoadLF
from renderer import LFRenderer
from filters import LFFilters
from PIL.ImageQt import ImageQt
from PyQt5.QtWidgets import *
from PyQt5.QtCore import Qt, QTimer
from PyQt5 import QtGui
import open_gif
import cv2

""" A class that creates a GUI in order to visualize LF and the shift-sum filter for refocusing.
    Inputs: LFRenderer, LFFilters
    Output: A GUI
"""


font = QtGui.QFont("Arial", 9)


class Window(QWidget):

    def __init__(self):
        super().__init__()
        self.setGeometry(350, 150, 700, 500)  # first 2 position, second 2 size
        self.setWindowTitle("Light Field GUI")
        self.h_slider = QSlider(Qt.Horizontal)
        self.v_slider = QSlider(Qt.Vertical)
        self.upload_btn = QPushButton("Upload")
        self.upload_btn.setFont(font)
        self.exit_btn = QPushButton("Exit")
        self.exit_btn.setFont(font)
        self.start_btn = QPushButton("Start")
        self.start_btn.setFont(font)
        self.stop_btn = QPushButton("Stop")
        self.stop_btn.setFont(font)
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.setFont(font)
        self.export_btn = QPushButton("Export")
        self.export_btn.setFont(font)
        self.upload_btn_filt = QPushButton("Upload")
        self.upload_btn_filt.setFont(font)
        self.exit_btn_filt = QPushButton("Exit")
        self.exit_btn_filt.setFont(font)
        self.timer = QTimer()
        self.v_slider_filt = QSlider(Qt.Vertical)

        self.previous_value_lr = -7
        self.previous_value_tb = -7
        self.step = 0.2
        self.count = 0
        self.count_max = 7
        self.semaphore = {"left_start": True, "up": False, "right": False, "down": False, "left_end": False}

        self.parent_dir = 'D:\intellij_workspace2\LFRenderer\\stanford'
        self.path_chess = os.path.join(self.parent_dir, 'rectified_chess')
        self.path_lego = os.path.join(self.parent_dir, 'rectified_lego')
        self.path_ball = os.path.join(self.parent_dir, 'rectified_ball')
        self.path_fish_multi = os.path.join(self.parent_dir, "rectified_fish_multi")
        self.path_fish_eye = os.path.join(self.parent_dir, "rectified_fish_eye")
        self.images_chess = LoadLF(self.path_chess, new_directory="grey_scale_rectified_chess")
        self.images_lego = LoadLF(self.path_lego, new_directory="grey_scale_rectified_lego")
        self.images_ball = LoadLF(self.path_ball, new_directory="grey_scale_rectified_ball")
        self.images_fish_multi = LoadLF(self.path_fish_multi, new_directory="grey_scale_rectified_fish_multi")
        self.images_fish_eye = LoadLF(self.path_fish_eye, new_directory="grey_scale_rectified_fish_eye")

        self.resolution_im = (800, 800)
        self.resolution_lf = 400
        self.cam_pos = np.array((0, 0, 0))
        self.cam_rot = 0
        self.distance = 8
        self.renderer_chess = LFRenderer(self.resolution_im, self.resolution_lf, self.cam_pos, self.cam_rot, self.distance,
                                         self.images_chess.lf_mat)
        self.min_value = self.renderer_chess.min_value
        self.max_value = self.renderer_chess.max_value

        self.previous_value_h_slider = 10
        self.previous_value_v_slider = 10

        self.slope_1 = -0.45
        self.slope_2 = -1.6
        self.slope_3 = 2.0
        self.slope_list = [str(self.slope_1), str(self.slope_2), str(self.slope_3)]
        self.slope_combo = QComboBox()
        self.slope_combo.addItems(self.slope_list)
        self.colors_list = ["grayscale", "green", "red", "yellow", "orange", "violet", "blue"]
        self.colors_combo = QComboBox()
        self.colors_combo.addItems(self.colors_list)
        self.stretched = False
        self.setMouseTracking(True)
        self.ui()

    def map_intervals(self, x, a, b):
        c = self.max_value
        d = self.min_value
        dist_fst_interval = b - a
        dist_snd_interval = d - c
        div = float(dist_fst_interval / dist_snd_interval)
        result = x / div + (c - (a / div))
        return result

    def mouseMoveEvent(self, event):
        if 50 <= event.x() <= 450 and 25 <= event.y() <= 425:
            lr = self.map_intervals(event.x(), 50, 450)
            ud = self.map_intervals(event.y(), 25, 425)
            self.image = self.renderer_chess.recalculate_interpolation(mouse=True, lr=lr, ud=ud)
            self._set_image()

    # here we use and declare widgets
    def ui(self):
        main_layout = QVBoxLayout()
        self.exit_btn.clicked.connect(self.exit_func)
        self.start_btn.clicked.connect(self.start_func)
        self.stop_btn.clicked.connect(self.stop_func)
        self.upload_btn.clicked.connect(self.open_dir)
        self.reset_btn.clicked.connect(self.reset_func)
        self.export_btn.clicked.connect(self.export_func)
        self.timer.setInterval(50)  # ms
        self.timer.timeout.connect(self.change_3d_view)

        # tab: visualize LF
        self.image = self.renderer_chess.lf_image
        self.qt_image = ImageQt(self.image)
        self.qt_image = QtGui.QImage(self.qt_image)
        self.pixmap = QtGui.QPixmap.fromImage(self.qt_image)
        self.im = QLabel()
        self.im.setPixmap(self.pixmap)

        hbox_main = QHBoxLayout()
        self.hbox = QHBoxLayout()
        vbox = QVBoxLayout()
        hbox2 = QHBoxLayout()

        self.h_slider.setMaximum(21)
        self.h_slider.setMinimum(0)
        self.h_slider.setTickPosition(QSlider.TicksAbove)
        self.h_slider.setValue(10)
        self.h_slider.valueChanged.connect(self.move_left_right)

        self.v_slider.setMinimum(0)
        self.v_slider.setMaximum(21)
        self.v_slider.setTickPosition(QSlider.TicksLeft)
        self.v_slider.setValue(10)
        self.v_slider.valueChanged.connect(self.move_top_bottom)

        gbox_automatic = QGroupBox("Automatic Process")
        gbox_automatic.resize(5, 5)
        hbox_automatic = QHBoxLayout()
        vbox_automatic = QVBoxLayout()

        gbox_color = QGroupBox("Choose colormap")
        gbox_color.resize(10, 10)
        label_color = QLabel("Choose a color to export the gif with a colormap")
        vbox_color = QVBoxLayout()
        vbox_color.addStretch()
        vbox_color.addWidget(label_color)
        vbox_color.addWidget(self.colors_combo)
        vbox_color.addStretch()
        gbox_color.setLayout(vbox_color)

        self.information = QLabel("Press the Start Button to visualize the light field automatically")
        vbox_automatic.addWidget(self.information)
        hbox_automatic.addStretch()
        hbox_automatic.addWidget(self.start_btn)
        hbox_automatic.addWidget(self.stop_btn)
        hbox_automatic.addStretch()
        vbox_automatic.addLayout(hbox_automatic)
        gbox_automatic.setLayout(vbox_automatic)

        gbox_instructions = QGroupBox("Instructions")
        gbox_instructions.resize(7, 7)
        vbox_instructions = QVBoxLayout()
        label1 = QLabel("Use this Interface to visualize multiple Light Fields.")
        label2 = QLabel("You can tap on the Upload button to change the current Light Field.")
        label3 = QLabel("You can move the Light Field by using your keyboard after tapping on one of the sliders.")
        label4 = QLabel("You can start an Automatic Process and sit back to visualize the Light Field.")
        label5 = QLabel("You can also use your mouse to drag the Light Field.")
        vbox_instructions.addWidget(label1)
        vbox_instructions.addWidget(label2)
        vbox_instructions.addWidget(label3)
        vbox_instructions.addWidget(label4)
        vbox_instructions.addWidget(label5)
        gbox_instructions.setLayout(vbox_instructions)

        vbox_rightside = QVBoxLayout()
        vbox_rightside.addWidget(gbox_color)
        vbox_rightside.addWidget(gbox_automatic)
        vbox_rightside.addWidget(gbox_instructions)

        self.hbox.addWidget(self.v_slider)
        self.hbox.addWidget(self.im)
        vbox.addStretch()
        vbox.addLayout(self.hbox)
        vbox.addWidget(self.h_slider)
        vbox.addStretch()
        hbox2.addStretch()
        hbox2.addWidget(self.upload_btn)
        hbox2.addWidget(self.exit_btn)
        hbox2.addWidget(self.reset_btn)
        hbox2.addWidget(self.export_btn)
        hbox2.addStretch()
        vbox.addLayout(hbox2)
        vbox.addStretch()
        hbox_main.addLayout(vbox)
        hbox_main.addLayout(vbox_rightside)

        # tab: apply filter
        hbox_main_filt = QHBoxLayout()
        vbox_filt_left = QVBoxLayout()
        vbox_filt_right = QVBoxLayout()
        gbox_slope = QGroupBox("Choose Slope")
        gbox_slope.resize(20, 20)
        gbox_slope_information = QGroupBox("Information")
        gbox_slope_information.resize(20, 20)
        vbox_slope = QVBoxLayout()
        vbox_combo = QVBoxLayout()
        vbox_info_slope = QVBoxLayout()
        hbox_btns = QHBoxLayout()
        self.hbox_label = QHBoxLayout()
        self.hbox_shifted_im = QHBoxLayout()
        self.exit_btn_filt.clicked.connect(self.exit_func)
        self.upload_btn_filt.clicked.connect(self.upload_func)
        self.label_instructions = QLabel("Press upload and wait")
        self.label_instructions.setFont(font)
        self.hbox_shifted_im.addWidget(self.label_instructions)
        label_slopes = QLabel("Choose a slope from the list to apply filter there.")
        label_information = QLabel("The slope with the value -0.45 focuses on the center of the Light Field.")
        label_information1 = QLabel("If there is an object there, it will appear in focus.")
        label_information2 = QLabel("Smaller slopes focus on the upper part of the Light Field.")
        label_information3 = QLabel("Greater slopes focus on the lower part of the Light Field.")

        hbox_btns.addStretch()
        hbox_btns.addWidget(self.upload_btn_filt)
        hbox_btns.addWidget(self.exit_btn_filt)
        hbox_btns.addStretch()

        self.v_slider_filt.setMinimum(0)
        self.v_slider_filt.setMaximum(2)
        self.v_slider_filt.setTickPosition(QSlider.TicksLeft)
        self.v_slider_filt.setValue(1)
        self.v_slider_filt.valueChanged.connect(self.change_shift)
        self.hbox_label.addWidget(self.v_slider_filt)
        self.hbox_label.addStretch()
        self.hbox_label.addLayout(self.hbox_shifted_im)
        self.hbox_label.addStretch()
        vbox_slope.addLayout(self.hbox_label)
        vbox_slope.addLayout(hbox_btns)
        vbox_filt_left.addLayout(vbox_slope)

        hbox_save_btn = QHBoxLayout()
        self.save_btn_slope = QPushButton("Save")
        self.save_btn_slope.setEnabled(False)
        self.save_btn_slope.clicked.connect(self.save_slope)
        hbox_save_btn.addWidget(self.save_btn_slope)
        hbox_save_btn.addStretch()
        vbox_combo.addStretch()
        vbox_combo.addWidget(label_slopes)
        vbox_combo.addWidget(self.slope_combo)
        vbox_combo.addLayout(hbox_save_btn)
        vbox_combo.addStretch()
        gbox_slope.setLayout(vbox_combo)
        vbox_filt_right.addWidget(gbox_slope)

        vbox_info_slope.addWidget(label_information)
        vbox_info_slope.addWidget(label_information1)
        vbox_info_slope.addWidget(label_information2)
        vbox_info_slope.addWidget(label_information3)
        vbox_info_slope.addStretch()
        gbox_slope_information.setLayout(vbox_info_slope)
        vbox_filt_right.addWidget(gbox_slope_information)

        hbox_main_filt.addLayout(vbox_filt_left)
        hbox_main_filt.addLayout(vbox_filt_right)

        # declare tabs
        self.tabs = QTabWidget()

        self.tab_visualize_lf = QWidget()
        self.tab_filter = QWidget()

        self.tabs.addTab(self.tab_visualize_lf, "Visualize LF")
        self.tabs.addTab(self.tab_filter, "Apply Filters")

        self.tab_visualize_lf.setLayout(hbox_main)
        self.tab_filter.setLayout(hbox_main_filt)

        main_layout.addWidget(self.tabs)
        self.setLayout(main_layout)

        self.show()

    # function to display the LF
    def _set_image(self):
        self.qt_image = ImageQt(self.image)
        self.qt_image = QtGui.QImage(self.qt_image)
        self.pixmap = QtGui.QPixmap.fromImage(self.qt_image)
        self.hbox.removeWidget(self.im)
        self.im.deleteLater()
        self.im = None
        self.im = QLabel()
        self.im.setPixmap(self.pixmap)
        self.hbox.addWidget(self.im)

    # function to change the viewing angle of the LF
    def move_left_right(self):
        value = self.h_slider.value()

        if value - self.previous_value_h_slider < 0:
            factor = (self.previous_value_h_slider - value) / 5.0
        else:
            factor = -(value - self.previous_value_h_slider) / 5.0

        self.previous_value_lr += factor
        self.previous_value_h_slider = value
        self.image = self.renderer_chess.recalculate_interpolation(factor)
        self._set_image()

    # function to change the viewing angle of the LF
    def move_top_bottom(self):
        value = self.v_slider.value()

        if value - self.previous_value_v_slider < 0:
            factor = (self.previous_value_v_slider - value) / 5.0
        else:
            factor = -(value - self.previous_value_v_slider) / 5.0

        self.previous_value_tb += factor
        self.previous_value_v_slider = value
        self.image = self.renderer_chess.recalculate_interpolation(factor, up_down=True)
        self._set_image()

    # function to quit the GUI
    def exit_func(self):
        m_box = QMessageBox.question(self, "Warning", "Are you sure you want to quit?",
                                     QMessageBox.Yes | QMessageBox.No)
        if m_box == QMessageBox.Yes:
            sys.exit(0)

    # function to reset the viewing angle of the LF
    def reset_func(self):
        self.image = self.renderer_chess.recalculate_interpolation(reset=True)
        self._set_image()
        self.h_slider.setValue(10)
        self.v_slider.setValue(10)

    # function to upload a new LF
    def open_dir(self):
        url = QFileDialog.getExistingDirectory(self, "Open a Directory",
                                               "D:\intellij_workspace2\LFRenderer\\stanford\\",
                                               QFileDialog.ShowDirsOnly | QFileDialog.DontResolveSymlinks)
        if not url:
            return

        if 'chess' in url:
            self.renderer_chess = LFRenderer(self.resolution_im, self.resolution_lf, self.cam_pos, self.cam_rot,
                                             self.distance, self.images_chess.lf_mat)
        elif 'lego' in url:
            self.renderer_chess = LFRenderer(self.resolution_im, self.resolution_lf, self.cam_pos, self.cam_rot,
                                             self.distance, self.images_lego.lf_mat)
        elif 'ball' in url:
            self.renderer_chess = LFRenderer(self.resolution_im, self.resolution_lf, self.cam_pos, self.cam_rot,
                                             self.distance, self.images_ball.lf_mat)
        elif 'fish_multi' in url:
            self.renderer_chess = LFRenderer(self.resolution_im, self.resolution_lf, self.cam_pos, self.cam_rot,
                                             self.distance, self.images_fish_multi.lf_mat)
        elif 'fish_eye' in url:
            self.renderer_chess = LFRenderer(self.resolution_im, self.resolution_lf, self.cam_pos, self.cam_rot,
                                             self.distance, self.images_fish_eye.lf_mat)
        else:
            return

        self.h_slider.setValue(10)
        self.v_slider.setValue(10)
        self.semaphore = {"left_start": True, "up": False, "right": False, "down": False, "left_end": False}
        self.image = self.renderer_chess.lf_image
        self._set_image()

    def _apply_colormap(self, lf_im, lut):
        lf_im = np.array(lf_im)
        lf_im = cv2.merge((lf_im, lf_im, lf_im))

        colored_im = cv2.LUT(lf_im, lut)
        colored_im = Image.fromarray(np.uint8(colored_im))
        colored_im = colored_im.convert('P')

        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return colored_im

    # function to export gifs
    def export_func(self):
        color = self.colors_combo.currentText()

        black = np.zeros((1, 1, 3), np.uint8)
        black[:] = (13, 12, 12)

        white = np.zeros((1, 1, 3), np.uint8)
        white[:] = (253, 250, 253)

        if color == "grayscale":
            color = np.zeros((1, 1, 3), np.uint8)
            color[:] = (107, 107, 107)

        elif color == "green":
            color = np.zeros((1, 1, 3), np.uint8)
            color[:] = (23, 165, 37)

        elif color == "blue":
            color = np.zeros((1, 1, 3), np.uint8)
            color[:] = (18, 98, 235)

        elif color == "violet":
            color = np.zeros((1, 1, 3), np.uint8)
            color[:] = (129, 74, 138)

        elif color == "yellow":
            color = np.zeros((1, 1, 3), np.uint8)
            color[:] = (232, 232, 0)

        elif color == "orange":
            color = np.zeros((1, 1, 3), np.uint8)
            color[:] = (249, 168, 7)

        else:  # red
            color = np.zeros((1, 1, 3), np.uint8)
            color[:] = (209, 20, 20)

        # append colors
        lut = np.concatenate((black, color, white), axis=0)

        # resize lut to 256 values
        lut = cv2.resize(lut, (1, 256), interpolation=cv2.INTER_CUBIC)

        self.renderer_chess.recalculate_interpolation(reset=True)

        lf_im = self.renderer_chess.lf_image
        colored_im = self._apply_colormap(lf_im, lut)

        images = [colored_im]

        steps = 6
        print("Loading", end="")
        for step in range(steps):
            print(".", end="")
            lf_im = self.renderer_chess.recalculate_interpolation(factor=-0.2)
            colored_im = self._apply_colormap(lf_im, lut)
            images.append(colored_im)

        for step in range(steps):
            print(".", end="")
            lf_im = self.renderer_chess.recalculate_interpolation(factor=0.2, up_down=True)
            colored_im = self._apply_colormap(lf_im, lut)
            images.append(colored_im)

        for step in range(steps):
            print(".", end="")
            lf_im = self.renderer_chess.recalculate_interpolation(factor=0.2)
            colored_im = self._apply_colormap(lf_im, lut)
            images.append(colored_im)

        for step in range(steps):
            print(".", end="")
            lf_im = self.renderer_chess.recalculate_interpolation(factor=-0.2, up_down=True)
            colored_im = self._apply_colormap(lf_im, lut)
            images.append(colored_im)

        images[0].save('lf_gif.gif',
                       save_all=True, append_images=images[1:], optimize=False, duration=70, loop=0)
        print("Gif generated")

        info_box = QMessageBox.information(self, "GIF",
                                           "Successfully exported gif into the project. Do you want to see the gif now?",
                                           QMessageBox.Yes | QMessageBox.No)
        if info_box == QMessageBox.Yes:
            open_gif.run_gif('lf_gif.gif')

    # function that starts an automatic process
    def start_func(self):
        self.timer.start()
        self.information.setText("Press the Stop Button to stop the automated view")
        self.setMouseTracking(False)

    # function that stops an automatic process
    def stop_func(self):
        self.timer.stop()
        self.information.setText("Press the Start Button to visualize the light field automatically")
        self.setMouseTracking(True)

    # internal function that is used to calculate how much to change the viewing angle of the LF
    # and makes sure that we don't get out of range
    def _verify(self, step, up_down=False):
        if up_down:
            if step > 0:
                if (self.previous_value_tb + step) >= self.max_value:
                    factor = -step
                else:
                    factor = step
            else:
                if (self.previous_value_tb + step) >= self.min_value:
                    factor = step
                else:
                    factor = -step
            self.previous_value_tb += factor
        else:
            if step > 0:
                if (self.previous_value_lr + step) >= self.max_value:
                    factor = -step
                else:
                    factor = step
            else:
                if (self.previous_value_lr + step) >= self.min_value:
                    factor = step
                else:
                    factor = -step
            self.previous_value_lr += factor

        return factor

    def _perform_changes(self, curr_semaphore_key, next_semaphore_key, direction, change_count_max=7):
            self.direction = direction
            if self.count == self.count_max:
                self.count = 0
                self.count_max = change_count_max
                self.semaphore[curr_semaphore_key] = False
                self.semaphore[next_semaphore_key] = True
            else:
                self.count += 1

    # function used for the automatic process to choose where to change the viewing angle of the LF
    def _choose_direction(self):
        # left: 0, right: 1, down: 2, up: 3
        if self.semaphore["left_start"]:
            self._perform_changes("left_start", "up", 0, 9)
        elif self.semaphore["up"]:
            self._perform_changes("up", "right", 3, 13)
        elif self.semaphore["right"]:
            self._perform_changes("right", "down", 1, 9)
        elif self.semaphore["down"]:
            self._perform_changes("down", "left_end", 6)
        else:
            self._perform_changes("left_end", "left_start", 0, 4)

    # helper function to actually change the viewing angle of the LF
    def change_3d_view(self):
        self._choose_direction()
        up_down = False
        if self.direction == 0:  # left
            factor = self._verify(self.step)
        elif self.direction == 1:  # right
            factor = self._verify(-self.step)
        elif self.direction == 3:  # down
            factor = self._verify(self.step, up_down=True)
            up_down = True
        else:  # up
            factor = self._verify(-self.step, up_down=True)
            up_down = True

        if up_down:
            self.image = self.renderer_chess.recalculate_interpolation(factor, up_down=True)
        else:
            self.image = self.renderer_chess.recalculate_interpolation(factor)

        self._set_image()

    # function to set the filtered LF
    def _set_filt_im(self, index):
        self.image_filt = self.filter_lego.list_img_filter[index]
        self.qt_image_filt = ImageQt(self.image_filt)
        self.qt_image_filt = QtGui.QImage(self.qt_image_filt)
        self.pixmap_filt = QtGui.QPixmap.fromImage(self.qt_image_filt)
        self.im_filt = QLabel()
        self.im_filt.setPixmap(self.pixmap_filt)

        if self.label_instructions:
            self.hbox_shifted_im.removeWidget(self.label_instructions)
            self.label_instructions.deleteLater()
            self.label_instructions = None

        self.hbox_shifted_im.addStretch()
        self.hbox_shifted_im.addWidget(self.im_filt)
        self.hbox_shifted_im.addStretch()

    # function to upload the filtered images
    def upload_func(self):
        self.filter_lego = LFFilters(self.images_lego.lf_mat, slopes=[self.slope_1, self.slope_2, self.slope_3])
        self._set_filt_im(index=0)
        self.upload_btn_filt.setEnabled(False)
        self.save_btn_slope.setEnabled(True)

    # function used when the user changes the slope in the combo box
    def save_slope(self):
        value = self.slope_combo.currentText()
        index = self.return_index_combo(float(value))
        if self.im_filt:
            self.im_filt.deleteLater()
            self.hbox_shifted_im.removeWidget(self.im_filt)
        self._set_filt_im(index=index)

    def return_index(self, value):
        if value == 0:
            return 2
        elif value == 1:
            return 0
        else:
            return 1

    def return_index_combo(self, value):
        if value < -0.45:
            return 1
        elif value > -0.45:
            return 2
        else:
            return 0

    # function used when the user changes the slope with the sliders
    def change_shift(self):
        value = int(self.v_slider_filt.value())
        index = self.return_index(value)

        if self.im_filt:
            self.im_filt.deleteLater()
            self.hbox_shifted_im.removeWidget(self.im_filt)

        self._set_filt_im(index=index)


def main_func():
    app = QApplication(sys.argv)
    window = Window()
    sys.exit(app.exec_())


if __name__ == '__main__':
    main_func()
