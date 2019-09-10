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


def parallelize_dataframe(df_split, func, algadf, leap):
    part = functools.partial(func, algadf=algadf, leap=leap)
    # pool = Pool(num_cores)
    # df = pd.concat(pool.map(part, df_split))
    with ProcessPoolExecutor(num_cores) as pool:
        df = pd.concat(pool.map(part, df_split))
    # pool.close()
    # pool.join()
    return df


def func(select, algadf=None, leap=None):
    select.set_index('frame',
                     inplace=True,
                     drop=False
                     )
    select[['dy', 'dx']] = select[['y', 'x']] - algadf[['y', 'x']]
    select['r2'] = select['dx'].apply(lambda x: x ** 2) + select['dy'].apply(lambda x: x ** 2)
    select[['dy', 'dx']] = select[['y', 'x']].shift(periods=-leap) - select[['y', 'x']]
    return select


def abs_frame():
    warnings.filterwarnings("ignore")
    # path = '..\\Data\\Example3'
    # path = '..\\Data\\190617\\NexDevice_Billes_Chlamy_1p5ul_50fps_40x_C001H001S0001'
    path = 'D:\\Microswimmers2D_SBR\\Data\\190715\\ChlamyBilles_60x_2_C001H001S0001'
    # ------------------------------------------------------
    """ PARAMETERS """
    fps = 50
    lens_magnification = 40
    width = 1024
    height = 1024
    # ------------------------------------------------------
    # with open(path + '\\NoBackground_median\\Bandpass\\Clear\\t_algae.csv') as csv_file:
    #     stack_algae = TrajectorySequence(
    #         pd.read_csv(csv_file,
    #                     delimiter=',',
    #                     dtype={'frame': int, 'particle': int, 'y': float, 'x': float},
    #                     usecols=['y', 'x', 'frame', 'particle'])
    #     )
    #
    # """ Application of the derivative filter to the algae trajectories"""
    #
    # stack_algae.filter_trajectories(
    #     path + '\\NoBackground_median\\Bandpass\\Clear', lag=30, subject='algae'
    # )

    with open(path + '\\NoBackground_median\\Bandpass\\Clear\\t_particles.csv') as csv_file:
        stack_particles = TrajectorySequence(
            pd.read_csv(csv_file,
                        delimiter=',',
                        dtype={'frame': int, 'particle': int, 'y': float, 'x': float},
                        usecols=['y', 'x', 'frame', 'particle'])
        )

    with open(path + '\\NoBackground_median\\Bandpass\\Clear\\t_algae_filtered.csv') as csv_file:
        filtered_algae_trajectories = TrajectorySequence(
            pd.read_csv(csv_file,
                        delimiter=',',
                        dtype={'frame': int, 'particle': int, 'y': float, 'x': float},
                        usecols=['y', 'x', 'frame', 'particle'])
        )

    # tp.plot_traj(filtered_algae_trajectories.all_trajectories_dataframe, label=True)
    # plt.show()

    # ------------------------------------------------------

    itertime = [2, 4, 6, 8, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 76, 82, 88, 94, 100]
    for t in itertime:
        time_lag = t / 100
        leap = int(fps * time_lag)

        alga_frame_list = filtered_algae_trajectories.all_trajectories_dataframe.frame.unique()

        df = stack_particles.all_trajectories_dataframe[['y', 'x', 'frame', 'particle']]
        alga_frames = df['frame'].isin(alga_frame_list)
        df = df[alga_frames]
        tso = TrajectorySequence(df)
        df_split = []
        for part in tso.item_list:
            df_split.append(tso.trajectory_dict[part].coord_dataframe)

        # ###################
        # start = time.time()
        # ###################

        for counter_main, alga in enumerate(filtered_algae_trajectories.item_list):
            """ Finding the alga trajectory angle """
            algadf = filtered_algae_trajectories.trajectory_dict[alga].coord_dataframe
            algadf.set_index('frame',
                             inplace=True,
                             drop=False
                             )

            print('\r')
            print('##########################')
            print('alga = {:04d}'.format(alga))

            """ COMENTA HDP """
            ##################################
            df2 = parallelize_dataframe(df_split, func, algadf, leap)
            ##################################

            # phi = 0.3 --> r = 218.89 px --> r2 = 47914.7433
            # phi = 0.7 --> r = 143.3 px --> r2 = 20534.89
            # phi = 2.7 --> r = 72.96 px --> r2 = 5323.86037

            circle = df2['r2'] <= 5323.86037
            df2 = df2[circle]
            df2.dropna(inplace=True)

            # ###################
            # end = time.time()
            # print(end - start)
            # ###################

            create_directory(
                path + '\\NoBackground_median\\Bandpass\\Clear\\Dt{:03d}'.format(t)
            )

            df2.to_csv(
                path + '\\NoBackground_median\\Bandpass\\Clear\\Dt{:03d}\\t_abs_{:04d}.csv'.format(
                    t, filtered_algae_trajectories.item_list[counter_main]
                ),
                index=None,
                header=True
            )

            # tp.plot_traj(df2, label=True) #, axes=ax
            # plt.show()

        col_data = pd.DataFrame(columns=['dy', 'dx'])

        counter_main = 0
        for f in glob.glob(
                path + '\\NoBackground_median\\Bandpass\\Clear\\Dt{:03d}\\t_abs*.csv'.format(t)
        ):
            counter_main += 1
            with open(f) as csv_file:
                df = pd.read_csv(csv_file,
                                 delimiter=',',
                                 dtype={'dy': float, 'dx': float},
                                 usecols=['dy', 'dx'])
                col_data = pd.concat([col_data, df[['dy', 'dx']]])

        col_data.to_csv(
            path + '\\NoBackground_median\\Bandpass\\Clear\\Dt{:03d}\\abs_dydx.csv'.format(t),
            index=False,
            header=True
        )


if __name__ == "__main__":
    abs_frame()