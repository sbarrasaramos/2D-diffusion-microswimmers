from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

from preprocessing.trajectories import TrajectorySequence
from utils import *
import warnings

import functools
# from multiprocessing import Pool
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import numpy as np
# import trackpy as tp
# import matplotlib.pyplot as plt

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
    select[['y', 'x']] = select[['y', 'x']] - algadf[['y', 'x']]
    select['r2'] = select['x'].apply(lambda x: x ** 2) + select['y'].apply(lambda x: x ** 2)
    select = pd.concat([select, select[['y', 'x']].shift(periods=-leap) - select[['y', 'x']]], axis=1)
    select.columns = ['y', 'x', 'frame', 'particle', 'r2', 'dy', 'dx']
    select['theta_p'] = np.arctan2(select.dy.values, select.dx.values)
    select['alpha'] = select['theta_p'] - algadf['theta_a']
    select['d'] = (select.dx.values ** 2 + select.dy.values ** 2) ** 0.5
    select['dy_turningrf'] = select['d'] * select['alpha'].apply(lambda x: np.sin(x))
    select['dx_turningrf'] = select['d'] * select['alpha'].apply(lambda x: np.cos(x))

    return select


def turning_alga_frame():
    warnings.filterwarnings("ignore")
    path = '..\\Data\\Example3'
    # path = '..\\Data\\190617\\NexDevice_Billes_Chlamy_1p5ul_50fps_40x_C001H001S0001'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190715\\ChlamyBilles_60x_2_C001H001S0001'
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
    """ Application of the derivative filter to the algae trajectories """
    #
    # stack_algae.filter_trajectories(
    #     path + '\\NoBackground_median\\Bandpass\\Clear', lag=30, subject='algae'
    # )

    """ Reading of the trajectories data (algae and tracers) and writing them on two separate dataframes """
    with open(path + '\\NoBackground_median\\cropped\\Bandpass\\t_particles.csv') as csv_file:
        stack_particles = TrajectorySequence(
            pd.read_csv(csv_file,
                        delimiter=',',
                        dtype={'frame': int, 'particle': int, 'y': float, 'x': float},
                        usecols=['y', 'x', 'frame', 'particle'])
        )

    with open(path + '\\NoBackground_median\\cropped\\Bandpass\\t_algae_filtered.csv') as csv_file:
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

        """ Obtaining a list of the IDs of the algae  --> PUTO USA ITEM_LIST """
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
            algadf = pd.concat([algadf, algadf[['y', 'x']].shift(periods=-leap) - algadf[['y', 'x']]], axis=1)
            algadf.columns = ['y', 'x', 'frame', 'particle', 'dy', 'dx']
            algadf['theta_a'] = np.arctan2(algadf.dy.values, algadf.dx.values)

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

            # phi = 0.3 --> r = 93.03 mu m
            # phi = 0.7 --> r = 60.90 mu m
            # phi = 2.7 --> r = 31.01 mu m

            circle = df2['r2'] <= 20534.89
            df2 = df2[circle]
            df2.dropna(inplace=True)

            # ###################
            # end = time.time()
            # print(end - start)
            # ###################

            create_directory(
                path + '\\NoBackground_median\\cropped\\Bandpass\\test07\\Dt{:03d}'.format(t)
            )

            df2.to_csv(
                path + '\\NoBackground_median\\cropped\\Bandpass\\test07\\Dt{:03d}\\t_turning_circle_{:04d}.csv'.format(
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
                path + '\\NoBackground_median\\cropped\\Bandpass\\test07\\Dt{:03d}\\t_turning_circle*.csv'.format(t)
        ):
            counter_main += 1
            with open(f) as csv_file:
                df = pd.read_csv(csv_file,
                                 delimiter=',',
                                 dtype={'dy_turningrf': float, 'dx_turningrf': float},
                                 usecols=['dy_turningrf', 'dx_turningrf'])
                col_data = pd.concat([col_data, df[['dy_turningrf', 'dx_turningrf']]])

        col_data.to_csv(
            path + '\\NoBackground_median\\cropped\\Bandpass\\test07\\Dt{:03d}\\turning_dydx.csv'.format(t),
            index=False,
            header=True
        )


if __name__ == "__main__":
    turning_alga_frame()