from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import os
from os.path import isfile, join
import functools
import winsound
import cv2
import numpy as np

# im = cv2.imread("myimage.png", 0)

read_image = functools.partial(cv2.imread, flags=cv2.IMREAD_GRAYSCALE)
finish_signal = functools.partial(winsound.Beep, frequency=2500, dwDuration=200)


def create_directory(path):
    """

    :param path:
    :return:
    """
    try:
        os.makedirs(path, exist_ok = True)
    except OSError as e:
        # print(path + ': ')
        print('This directory (' + path + ') already exists. Do you want to overwrite it? (Y/N)')
        answer = input()
        if answer == 'Y':
            os.makedirs(path, exist_ok = True)
        else:
            exit()
    return path


def convert_frames_to_video(path_in, path_out, video_fps):
    """

    :param path_in:
    :param path_out:
    :param video_fps:
    :return:
    """
    frame_array = []
    files = [f for f in os.listdir(path_in) if isfile(join(path_in, f))]

    for i in range(len(files)):
        filename = path_in + files[i]
        # reading each file
        img = cv2.imread(filename)
        height, width, layers = img.shape
        size = (width, height)
        # inserting the frames into an image array
        frame_array.append(img)

    out = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*'DIVX'), video_fps, size)

    for i in range(len(frame_array)):
        # writing to a image array
        out.write(frame_array[i])
    out.release()


def correlation(x, y, t=0):
    """

    :param x:
    :param y:
    :param t:
    :return:
    """
    result = np.correlate(x[t:], y[:y.index[-1]-t], mode='full')
    return result[result.size // 2:]


def autocorrelation(x, t=0):
    """

    :param x:
    :param t:
    :return:
    """
    return correlation(x[t:], x[:x.index[-1]-t])


def correlation_plot(vel_df1, which1, color1, label1, vel_df2=None, which2=None, color2=None, label2=None, axes=None):
    """

    :param vel_df1:
    :param which1:
    :param color1:
    :param label1:
    :param vel_df2:
    :param which2:
    :param color2:
    :param label2:
    :param axes:
    :return:
    """
    x = vel_df1[which1]
    x.reset_index()
    axes[0].plot(x, '-' + color1, label=which1 + '_' + label1)
    if vel_df2 is not None:
        y = vel_df2[which2]
        y.reset_index()
        size = min(len(x), len(y))
        x = x[0:size - 1]
        y = y[0:size - 1]
        axes[0].plot(y, '-' + color2, label=which2 + '_' + label2)
        z = correlation(x, y, t=0)
    else:
        z = autocorrelation(x, t=0)

    axes[0].legend()

    # Correlation
    axes[1].plot(z / float(z.max()), '-' + color1)
    axes[1].legend()
    axes[1].set_title("Correlation")




