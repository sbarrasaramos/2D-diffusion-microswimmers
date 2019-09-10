from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

from utils import *
from preprocessing.trajectories import TrajectorySequence

import numpy as np
import glob
import cv2
import pims
import trackpy as tp
import matplotlib.pyplot as plt

class ImageSequence:
    particlesize = 9
    particleminmass = 2e3
    minponts = 50

    def __init__(self, path, fps, lens_magnification):
        """

        :param path:
        :param fps:
        :param lens_magnification:
        """

        self.path = path
        self.files = [f for f in glob.glob(self.path + '**/*.tif', recursive=True)]
        self.image_sequence = pims.ImageSequence(path + '**/*.tif', as_grey=True)
        self.size = self.image_sequence.frame_shape
        self.frame_number = len(self.files)
        self.fps = fps
        self.lens_magnification = lens_magnification
        self.particle_trajectory_list = None
        self.algae_trajectory_list = None
        self.minpoints = None
        self.cmin = 255
        self.cmax = 0

    #     self.width = 1
    #     self.width(1)
    #     print(self.width)
    #     print(self.width())
    #
    # @property
    # def width(self):
    #     return self._width.copy()
    #
    # @width.setter
    # def width(self, value):
    #     if value % 2 == 0:
    #         raise ValueError()
    #     self._width = value

    def __call__(self):
        """

        :return:
        """
        pass

    def process_stack(self):
        im_filtered = []
        create_directory(self.path + '\\Bandpass')

        def im_filtered_generator():
            for f in glob.glob(self.path + '**\\*.tif', recursive=True):
                yield self.bandpass_filter(read_image(f))

        for im in im_filtered_generator():
            pass

        for counter, im in enumerate(im_filtered_generator()):
            im_filtered = self.equalization_contrast_denoising(im)
            cv2.imwrite(self.path + '\\Bandpass\\frame_{:04d}.tif'.format(counter),
                        im_filtered.astype(np.uint8))

        return ImageSequence(self.path + '\\Bandpass', self.fps, self.lens_magnification)

    def bandpass_filter(self, src):
        a = cv2.bilateralFilter(src, 3, 3, 3, 3).astype(np.floating)
        b = cv2.bilateralFilter(src, 40, 40, 40, 40).astype(np.floating)
        c = a - b
        self.cmin = min(self.cmin, np.amin(c))
        self.cmax = max(self.cmax, np.amax(c))
        return c

    def equalization_contrast_denoising(self, c):
        c = (c - self.cmin*np.ones_like(c))*255/(self.cmax-self.cmin)
        upper_threshold = 175
        c = c*255/upper_threshold
        for x in np.nditer(c, op_flags=['readwrite']):
            if x[...] > 255:
                x[...] = 255
        clahe = cv2.createCLAHE(clipLimit=6.0, tileGridSize=(8, 8))
        c = clahe.apply(c.astype(np.uint8))
        c = cv2.fastNlMeansDenoising(c, None, 10, 7, 21)
        _, c = cv2.threshold(c, 80, 255, cv2.THRESH_TOZERO)
        return c

    def remove_background(self, option='median', background=None, beta=75):
        """

        :param option:
        :param background:
        :param beta:
        :return:
        """
        if background not in ['mean', 'median']:
            self.find_background(option)
        ref_image = read_image(self.path + '\\' + option + '.tif')
        create_directory(self.path + '\\NoBackground_{0:s}'.format(option))
        counter = 0
        for f in glob.glob(self.path + '**\\*.tif', recursive=True):
            counter += 1
            im_diff = read_image(f) - ref_image.astype(np.uint8) + beta
            cv2.imwrite(self.path + '\\NoBackground_{0:s}'.format(option) + '\\frame_{:04d}.tif'.format(counter),
                        im_diff.astype(np.uint8))
        return ImageSequence(self.path + '\\NoBackground_{0:s}'.format(option), self.fps, self.lens_magnification)

    def find_background(self, option='median'):
        """

        :param option:
        :return:
        """
        random_files = self.files.copy()
        np.random.shuffle(random_files)
        if option in ['mean', 'median']:
            ref_image = np.median(
                np.stack(
                    [f for f in ImageSequence.__subset_stat(random_files, option, 301)],
                    axis=2),
                axis=2)
            cv2.imwrite(self.path + '\\{0:s}.tif'.format(option), ref_image.astype(np.uint8))
        return option

    @staticmethod
    def __subset_stat(files, option='median', step=101):
        """

        :param files:
        :param option:
        :param step:
        :return:
        """
        size = len(files)
        for i in range(0, size, step):
            print('\r{:.1f}% done'.format(i / size * 100), end='')
            if option == 'median':
                yield np.median(np.stack([read_image(f) for f in files[i:i + step - 1]], axis=2), axis=2)
            elif option == 'mean':
                yield np.mean(np.stack([read_image(f) for f in files[i:i + step - 1]], axis=2), axis=2)
        print('\r')

    def crop_images(self, x_range, y_range, frame_range):
        """

        :param x_range:
        :param y_range:
        :param frame_range:
        :return:
        """
        files = []
        for hundreds in range(frame_range[0], frame_range[-1]):
            files.extend(
                [f for f in glob.glob(self.path + '**\\*{:02d}[0-9][0-9]*.tif'.format(hundreds), recursive=True)])
        create_directory(self.path + '\\cropped')
        counter = frame_range[0] * 100
        for f in files:
            cv2.imwrite(self.path + '\\cropped\\frame_{:04d}.tif'.format(counter),
                        read_image(f)[y_range[0]:y_range[-1], x_range[0]:x_range[-1]])
            counter += 1
        return ImageSequence(self.path + '\\cropped', self.fps, self.lens_magnification)

    def generate_trajectories(self, subject='particles', plot=True, export=True, invert=True, memory=50):
        """

        :param subject:
        :param plot:
        :param export:
        :param invert:
        :param memory:
        :return:
        """
        fv = tp.batch(self.image_sequence, self.particlesize, minmass=self.particleminmass, invert=invert)
        t = tp.link_df(fv, 5, memory=memory)
        if subject == 'particles':
            self.particle_trajectory_list = TrajectorySequence(t)
        elif subject == 'algae':
            self.algae_trajectory_list = TrajectorySequence(t)
        else:
            raise Exception('The argument subject of the current method '
                            '(generate_trajectories) must be either particles or algae')
        if plot:
            tp.plot_traj(t, label=True)
            plt.show()
        if export:
            t.to_csv(self.path + '\\t_' + subject + '.csv', index=None, header=True)
        if subject == 'particles':
            return self.particle_trajectory_list
        if subject == 'algae':
            return self.algae_trajectory_list

    def delete_spurious_trajectories(self, subject='particles', plot=True, export=True):
        """

        :param subject:
        :return:
        """
    ### Deletion of spurious trajectories under 'minpoints' points ###
        if subject == 'particles':
            editable_list = self.particle_trajectory_list
        elif subject == 'algae':
            editable_list = self.algae_trajectory_list
        else:
            raise Exception('The argument subject of the current method '
                            '(delete_spurious_trajectories) must be either particles or algae')

        editable_list.set_all_trajectories_dataframe(
            tp.filter_stubs(
                editable_list.all_trajectories_dataframe, self.minpoints
            )
        )
        editable_list.update_item_list()
        if plot:
            tp.plot_traj(editable_list.all_trajectories_dataframe, label=True)
            plt.show()
        if export:
            editable_list.all_trajectories_dataframe.to_csv(self.path + '\\t_' + subject + '.csv', index=None, header=True)
        return editable_list

    def delete_algae_from_particles_trajectories(self, quantile=0.9, plot=True, export=True):
        self.particle_trajectory_list.set_all_trajectories_dataframe(tp.filtering.bust_clusters(
            self.particle_trajectory_list.all_trajectories_dataframe,
            quantile=quantile,
            threshold=None))
        self.particle_trajectory_list.update_item_list()
        if plot:
            tp.plot_traj(self.particle_trajectory_list.all_trajectories_dataframe, label=True)
            plt.show()
        if export:
            self.particle_trajectory_list.all_trajectories_dataframe.to_csv(
                self.path + '\\t_particles.csv',
                index=None,
                header=True
            )
        return self.particle_trajectory_list



