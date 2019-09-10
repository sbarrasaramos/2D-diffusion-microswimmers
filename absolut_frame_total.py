from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

from preprocessing.trajectories import TrajectorySequence
from utils import *
import warnings

import functools
# from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
import trackpy as tp
import matplotlib.pyplot as plt

import glob
import time

num_cores = 2  # number of cores on your machine


def parallelize_dataframe(df_split, func, leap):
    part = functools.partial(func, leap=leap)
    with ProcessPoolExecutor(num_cores) as pool:
        df = pd.concat(pool.map(part, df_split))
    return df

def func(select, leap=None):
    select.set_index('frame',
                     inplace=True,
                     drop=False
                     )
    select[['dy', 'dx']] = select[['y', 'x']].shift(periods=-leap) - select[['y', 'x']]
    return select


def abs_frame():
    warnings.filterwarnings("ignore")
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190726\\H2O_0%_18p5um_C001H001S0001'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190715\\ChlamyBilles_60x_2_C001H001S0001'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190715\\ChlamyBilles_60x_C001H001S0001'
    path = 'D:\\Microswimmers2D_SBR\\Data\\190808\\ANH_Chlamy_beads_ThicknessLessThan18pt_60x_C001H001S0001'
    # ------------------------------------------------------
    """ PARAMETERS """
    fps = 50
    lens_magnification = 60
    width = 512
    height = 512
    # ------------------------------------------------------
    with open(path + '\\NoBackground_median\\Clear\\t_particles.csv') as csv_file:
        stack_particles = TrajectorySequence(
            pd.read_csv(csv_file,
                        delimiter=',',
                        dtype={'frame': int, 'particle': int, 'y': float, 'x': float},
                        usecols=['y', 'x', 'frame', 'particle'])
        )
    # ------------------------------------------------------

    itertime = [2, 4, 6, 8, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 76, 82, 88, 94, 100]
    for t in itertime:
        time_lag = t / 100
        leap = int(fps * time_lag)

        df = stack_particles.all_trajectories_dataframe[['y', 'x', 'frame', 'particle']]
        tso = TrajectorySequence(df)

        df_split = []
        for part in tso.item_list:
            df_split.append(tso.trajectory_dict[part].coord_dataframe)

        """ COMENTA HDP """
        ##################################
        df2 = parallelize_dataframe(df_split, func, leap)
        ##################################

        df2.dropna(inplace=True)

        create_directory(
            path + '\\NoBackground_median\\Clear\\Dt{:03d}'.format(t)
        )

        df2.to_csv(
            path + '\\NoBackground_median\\Clear\\Dt{:03d}\\t_abs.csv'.format(t),
            index=None,
            header=True
        )

        col_data = pd.DataFrame(columns=['dy', 'dx'])

        counter_main = 0
        for f in glob.glob(
                path + '\\NoBackground_median\\Clear\\Dt{:03d}\\t_abs*.csv'.format(t)
        ):
            counter_main += 1
            with open(f) as csv_file:
                df = pd.read_csv(csv_file,
                                 delimiter=',',
                                 dtype={'dy': float, 'dx': float},
                                 usecols=['dy', 'dx'])
                col_data = pd.concat([col_data, df[['dy', 'dx']]])

        col_data.to_csv(
            path + '\\NoBackground_median\\Clear\\Dt{:03d}\\abs_dydx.csv'.format(t),
            index=False,
            header=True
        )


if __name__ == "__main__":
    abs_frame()