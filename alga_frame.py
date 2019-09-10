from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

from preprocessing.trajectories import TrajectorySequence
import warnings

import pandas as pd
import trackpy as tp

import matplotlib.pyplot as plt

import glob


def to_alga_frame():
    warnings.filterwarnings("ignore")
    path = '..\\Data\\190628\\1B_Chlamy_Billes_1um_60x_C001H001S0001'
# ------------------------------------------------------
    """ PARAMETERS """
    fps = 50
    lens_magnification = 60
    width = 1024
    height = 1024
# ------------------------------------------------------
    with open(path + '\\NoBackground_median\\cropped\\Bandpass\\t_algae.csv') as csv_file:
        stack_algae = TrajectorySequence(
            pd.read_csv(csv_file,
                        delimiter=',',
                        dtype={'frame': int, 'particle': int, 'y': float, 'x': float},
                        usecols=['y', 'x', 'frame', 'particle'])
        )

    """ Application of the derivative filter to the algae trajectories"""

    filtered_algae_trajectories = stack_algae.filter_trajectories(
        path + '\\NoBackground_median\\cropped\\Bandpass', lag=30, subject='algae'
    )

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
# ------------------------------------------------------

    df = stack_particles.all_trajectories_dataframe[['y', 'x', 'frame', 'particle']]
    alga_frame_list = list(filtered_algae_trajectories.all_trajectories_dataframe['frame'])
    alga_frame_list = list(set(alga_frame_list))

    alga_frames = df['frame'].isin(alga_frame_list)
    df = df[alga_frames]

    counter_main = 0
    for alga in filtered_algae_trajectories.item_list:
        ts = TrajectorySequence(df)
        algatr = filtered_algae_trajectories.trajectory_dict[alga]
        algadf = algatr.coord_dataframe
        algadf.set_index('frame',
                         inplace=True,
                         drop=False
                         )
        print('\r')
        print('##########################')
        print('alga = {:04d}'.format(alga))
        for part in ts.item_list:
            select = ts.trajectory_dict[part].coord_dataframe
            select.set_index('frame',
                             inplace=True,
                             drop=False
                             )
            select[['y', 'x']] = select[['y', 'x']] - algadf[['y', 'x']]
            select['r2'] = select['x'].apply(lambda x: x**2) + select['y'].apply(lambda x: x**2)
            shifted_select = select.loc[2:].copy()
            shifted_select.index = shifted_select.index - 2
            select['dy'] = shifted_select['y'] - select['y']
            select['dx'] = shifted_select['x'] - select['x']
            select['d2'] = select['dx'].apply(lambda x: x**2) + select['dy'].apply(lambda x: x**2)
            select.index = select.index + 2
            select['frame'] = select['frame'].apply(lambda x: x + 2)
            ts.trajectory_dict[part].coord_dataframe = select

        # phi = 0.7 --> r = 143.3
        circle = ts.all_trajectories_dataframe['r2'] <= 20534.89

        df2 = ts.all_trajectories_dataframe[circle]

        df2.dropna(inplace=True)

        df2.to_csv(
            path + '\\NoBackground_median\\cropped\\Bandpass\\t_circle_{:04d}.csv'.format(filtered_algae_trajectories.item_list[counter_main]),
            index=None,
            header=True
        )

        tp.plot_traj(df2, label=True) #, axes=ax
        plt.show()

        # hist = df2[['dx', 'dy', 'd2']].hist(bins=100)
        # plt.show()

        counter_main += 1

    all_data_dy = pd.DataFrame()
    all_data_dx = pd.DataFrame()

    col_data = pd.DataFrame(columns=['dy', 'dx'])

    counter_main = 0
    for f in glob.glob(path + '\\NoBackground_median\\cropped\\Bandpass\\t_circle*.csv'):
        counter_main += 1
        with open(f) as csv_file:
            df = pd.read_csv(csv_file,
                             delimiter=',',
                             dtype={'dy': float, 'dx': float},
                             usecols=['dy', 'dx'])
            # all_data_dy['dy_{:04d}'.format(counter_main)] = df['dy']
            # all_data_dx['dx_{:04d}'.format(counter_main)] = df['dx']
            col_data = pd.concat([col_data, df[['dy', 'dx']]])

    col_data.to_csv(
        path + '\\NoBackground_median\\cropped\\Bandpass\\dydx.csv',
        index=False,
        header=False
    )

if __name__=="__main__":
    to_alga_frame()