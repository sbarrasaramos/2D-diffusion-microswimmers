from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

from preprocessing.trajectories import TrajectorySequence
# from utils import *
# import warnings

# import functools
# from multiprocessing import Pool
# from concurrent.futures import ProcessPoolExecutor

import pandas as pd
# import numpy as np
import trackpy as tp
import matplotlib.pyplot as plt

# import glob
# import time

def whatever():
    path = 'D:\\Microswimmers2D_SBR\\Data\\190808\\ANH_Chlamy_beads_Thickness18pt_60x_C001H001S0001\\NoBackground_median\\Clear'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190808\\ANH_Chlamy_beads_ThicknessLessThan18pt_60x_C001H001S0001\\NoBackground_median\\Clear'

    with open(path + '\\t_particles.csv') as csv_file:
        stack_particles = TrajectorySequence(
            pd.read_csv(csv_file,
                        delimiter=',',
                        dtype={'frame': int, 'particle': int, 'y': float, 'x': float},
                        usecols=['y', 'x', 'frame', 'particle'])
        )

    f_all = stack_particles.all_trajectories_dataframe
    f200 = f_all['frame'] <= 200
    f_200 = f_all[f200]

    tp.plot_traj(f_200, label=True)
    plt.show()

if __name__=="__main__":
    whatever()