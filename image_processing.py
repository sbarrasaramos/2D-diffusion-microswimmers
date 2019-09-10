from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

from preprocessing.images import ImageSequence
import warnings

#########################################################
from preprocessing.trajectories import TrajectorySequence
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
#########################################################

def image_processing():
    warnings.filterwarnings("ignore")
    # path = '..\\Data\\Example3'
    # path = '..\\Data\\190509\\Chlamy\\60x'
    # path = '..\\Data\\190613\\BillesHC_ChlamyHC_2p0ul_50fps_40x_C001H001S0001'
    # path = '..\\Data\\190613\\Later_BillesHC_ChlamyHC_2p0ul_50fps_10x_C001H001S0001'
    # path = '..\\Data\\190617\\NexDevice_Billes_Chlamy_1p5ul_50fps_40x_C001H001S0001'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190715\\ChlamyBilles_60x_2_C001H001S0001'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190726\\H2O_0%_18p5um_C001H001S0001'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190715\\ChlamyBilles_60x_C001H001S0001\\NoBackground_median\\Clear'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190726\\H20_0%_15um_60x_3_BG_C001H001S0001'
    path = 'D:\\Microswimmers2D_SBR\\Data\\190808\\ANH_Chlamy_beads_ThicknessLessThan18pt_60x_C001H001S0001'

# ------------------------------------------------------
    """ PARAMETERS """
    fps = 50
    lens_magnification = 40
    width = 1024
    height = 1024

# ------------------------------------------------------
    """ OBTAINING IMAGE SET """
    stack = ImageSequence(path, fps, lens_magnification)

    """ Removing the background of the image sequence """
    stack_nobackground = stack.remove_background(option='median')

    # """ Cropping """
    # current_frames = [1, 7]
    # x_pixels = [237, 763]
    # y_pixels = [490, 1001]
    # stack_nb_cropped = stack_nobackground.crop_images(x_pixels, y_pixels, current_frames)
    # # stack_cropped = stack.crop_images(x_pixels, y_pixels, current_frames)

# ------------------------------------------------------
    """ IMAGE PROCESSING """
    # stack_nb_processed = stack_nobackground.process_stack()
    # stack_nb_processed = stack_nb_cropped.process_stack()

#########################################################

    # with open(path + '\\NoBackground_median\\Bandpass\\t_particles.csv') as csv_file:
    #     stack_particles = TrajectorySequence(
    #         pd.read_csv(csv_file,
    #                     delimiter=',',
    #                     dtype={'frame': int, 'particle': int, 'y': float, 'x': float},
    #                     usecols=['y', 'x', 'frame', 'particle'])
    #     )
    #
    # tp.plot_traj(stack_particles.all_trajectories_dataframe, label=True)
    # plt.show()

if __name__=="__main__":
    image_processing()