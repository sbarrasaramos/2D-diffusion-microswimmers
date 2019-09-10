from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import copy
import numpy as np
import pandas as pd
import trackpy as tp
import matplotlib.pyplot as plt
from scipy.optimize import least_squares


class Trajectory:
    def __init__(self, coord_dataframe: pd.DataFrame):
        # super().__init__(columns=columns)
        self.coord_dataframe = coord_dataframe
        self.item = self.coord_dataframe['particle'].iloc[0]
        if not self.coord_dataframe.loc[self.coord_dataframe['particle'] != self.item].empty:
            raise Exception('A trajectory must contain a single particle')

    def __call__(self):
        pass

    def filtered_coordinates(self):
        """
        Filters function according to its own derivative
        :return:
        """
        row_list = self.coord_dataframe.index.values[1:]
        for row in row_list:
            self.coord_dataframe.loc[
                row, 'y'
            ] = self.coord_dataframe.loc[row - 1, 'y'] + self.coord_dataframe.loc[row - 1, 'vy'] * 0.02
            self.coord_dataframe.loc[
                row, 'x'
            ] = self.coord_dataframe.loc[row - 1, 'x'] + self.coord_dataframe.loc[row - 1, 'vx'] * 0.02

    def mean_vel(self, lag):
        aux = self.coord_dataframe[['dy' + str(lag), 'dx' + str(lag)]].copy()
        for i in range(lag - 1):
            aux += self.coord_dataframe[['dy' + str(lag), 'dx' + str(lag)]].shift(i + 1)
        return aux

    def linear_interpolation(self):
        item_zero = self.coord_dataframe['frame'].iloc[0]
        item_last = self.coord_dataframe['frame'].iloc[-1]
        missing_items = list(set(map(int, np.linspace(item_zero, item_last, 1 + item_last - item_zero)))
                             .difference(self.coord_dataframe['frame']))
        for item in missing_items:
            self.coord_dataframe = self.coord_dataframe.append([{'frame': item}], ignore_index=False)
            self.coord_dataframe.index += 1  # shifting index
        self.coord_dataframe.sort_values('frame', inplace=True)

        self.coord_dataframe = self.coord_dataframe.astype(float).interpolate(method='linear', axis=0)

    def derivative_filter(self, lag=20):
        """

        :param lag:
        :return:
        """

        self.linear_interpolation()
        self.coord_dataframe.sort_values('frame',
                                       inplace=True)
        self.coord_dataframe[['frame', 'particle']] = self.coord_dataframe[['frame', 'particle']].astype(int)

        self.coord_dataframe.set_index('frame',
                                     inplace=True,
                                     drop=False)

        shifted_coord_df = self.coord_dataframe.loc[lag:].copy()

        shifted_coord_df.index = shifted_coord_df.index - lag

        diff = (shifted_coord_df - self.coord_dataframe)[['y', 'x']]
        self.coord_dataframe[['dy' + str(lag), 'dx' + str(lag)]] = diff.apply(lambda x: x / (lag * 0.02), axis=1)

        self.coord_dataframe[['vy', 'vx']] = self.mean_vel(lag=lag).apply(lambda x: x / lag)

        self.coord_dataframe[
            'vmod'
        ] = self.coord_dataframe['vy'].apply(lambda x: x ** 2) + self.coord_dataframe['vx'].apply(lambda x: x ** 2)

        self.coord_dataframe['vmod'] = self.coord_dataframe['vmod'].apply(np.sqrt)
        self.coord_dataframe['atan'] = np.rad2deg(np.arctan2(self.coord_dataframe['vy'], self.coord_dataframe['vx']))
        self.coord_dataframe = self.coord_dataframe.dropna()

        self.filtered_coordinates()

        return self

    @staticmethod
    def shape_function(alpha, x):
        return alpha[3] * x ** 0.5 + alpha[2] * x ** 2 + alpha[1] * x + alpha[0]

    @staticmethod
    def residue_shape_function(alpha, x, data):
        return Trajectory.shape_function(alpha, x) - data

class TrajectorySequence:

    def __init__(self, all_trajectories_dataframe):
        """

        :param all_trajectories_dataframe:
        """
        self.trajectory_dict = None
        self.item_list = None
        self.set_all_trajectories_dataframe(all_trajectories_dataframe)

    def __call__(self):
        """

        :return:
        """
        pass

    def __copy__(self):
        return TrajectorySequence(self.all_trajectories_dataframe)

    def set_all_trajectories_dataframe(self, all_trajectories_dataframe):
        self.trajectory_dict = {}
        self.item_list = pd.unique(list(all_trajectories_dataframe['particle'])).tolist()
        for item in self.item_list:
            self.trajectory_dict[item] = Trajectory(
                all_trajectories_dataframe.loc[all_trajectories_dataframe['particle'] == item]
            )

    @property
    def all_trajectories_dataframe(self):
        return pd.concat([value.coord_dataframe for value in self.trajectory_dict.values()])

    def update_item_list(self):
        """

        :return:
        """
        self.item_list = pd.unique(list(self.all_trajectories_dataframe['particle'])).tolist()
        return self.item_list

    def filter_trajectories(self, path, subject='particles', lag=20, plot=True, export=True, inplace=False):
        """

        :param lag:
        :return:
        """
        if inplace == False:
            sequence = self.__copy__()
        elif inplace == True:
            sequence = self

        for part in sequence.item_list:
            sequence.trajectory_dict[part].derivative_filter(lag=lag)
        if plot:
            tp.plot_traj(sequence.all_trajectories_dataframe, label=False)
            plt.show()
        if export:
            sequence.all_trajectories_dataframe.to_csv(
                path + '\\t_' + subject + '_filtered.csv',
                index=None,
                header=True
            )
        return sequence

    def find_interactions(self, stack_algae, x_pixels, y_pixels, distance_criterion=75):
        for part in self.item_list:
            self.trajectory_dict[part].coord_dataframe[
                'mindist'
            ] = np.sqrt((x_pixels[1] - x_pixels[0]) ** 2 + (y_pixels[1] - y_pixels[0]) ** 2)
            for alga in stack_algae.item_list:
                self.trajectory_dict[part].coord_dataframe[
                    ['disty', 'distx']
                ] = self.trajectory_dict[part].coord_dataframe[
                        ['y', 'x']
                    ] - stack_algae.trajectory_dict[alga].coord_dataframe[
                        ['y', 'x']
                    ]

                self.trajectory_dict[part].coord_dataframe[
                    'distmod'
                ] = self.trajectory_dict[part].coord_dataframe[
                        'disty'
                    ].apply(lambda x: x ** 2) + self.trajectory_dict[part].coord_dataframe[
                        'distx'
                    ].apply(lambda x: x ** 2)

                self.trajectory_dict[part].coord_dataframe[
                    'distmod'
                ] = self.trajectory_dict[part].coord_dataframe[
                    'distmod'
                ].apply(lambda x: np.sqrt(x))

                self.trajectory_dict[part].coord_dataframe[
                    'mindist'
                ] = self.trajectory_dict[part].coord_dataframe[['mindist', 'distmod']].min(axis=1)

        far = self.all_trajectories_dataframe['mindist'] >= distance_criterion

        return TrajectorySequence(self.all_trajectories_dataframe[far])

    def fit_trajectories(self): # Para que servia esto...?
        """

        :return:
        """
        for part in self.item_list:
            alpha0 = np.ones(4)
            frames = self.trajectory_dict[part].coord_dataframe['frame']
            item_zero = frames.iloc[0]
            item_last = frames.iloc[-1]
            missing_items = list(set(map(int, np.linspace(item_zero, item_last, 1 + item_last - item_zero)))
                                 .difference(frames))
            missing_items.append(-1)
            bottom = self.trajectory_dict[part].coord_dataframe['frame'] < missing_items[0]
            top = self.trajectory_dict[part].coord_dataframe['frame'] > missing_items[0]
            df_1 = self.trajectory_dict[part].coord_dataframe[bottom]
            df_2 = self.trajectory_dict[part].coord_dataframe[top]
            x_1 = df_1['x']
            y_1 = df_1['y']
            x_2 = df_2['x']
            y_2 = df_2['y']

            res_robust_1 = least_squares(
                Trajectory.residue_shape_function,
                alpha0,
                loss='soft_l1',
                f_scale=0.1,
                args=(x_1, y_1)
            )
            res_robust_2 = least_squares(
                Trajectory.residue_shape_function,
                alpha0,
                loss='soft_l1',
                f_scale=0.1,
                args=(x_2, y_2)
            )
            self.trajectory_dict[part].coord_dataframe[
                'y'
            ] = self.trajectory_dict[part].coord_dataframe[
                'x'
            ].apply(lambda xx: Trajectory.shape_function(res_robust_1.x, xx))

            self.trajectory_dict[part + 1000] = self.trajectory_dict[part]

            self.trajectory_dict[part + 1000].coord_dataframe[
                'y'
            ] = self.trajectory_dict[part].coord_dataframe[
                'x'
            ].apply(lambda xx: Trajectory.shape_function(res_robust_2.x, xx))
        return self.all_trajectories_dataframe

    "Methods out of use"
    # def atan_filter(self):
    #     """
    #
    #     :return:
    #     """
    #   atan_std = {}
    #     for part in self.item_list:
    #         atan_std[part] = np.std(list(self.trajectory_dict[part].coord_dataframe['atan']))
    #     med = np.median(list(atan_std.values()))
    #     low_atan_std = {k: v for k, v in atan_std.items() if v <= med}
    #     coord_df = [trajectory.coord_dataframe for trajectory in self.trajectory_dict.values()]
    #     low_atan_selection = self.all_trajectories_dataframe['particle'].isin(list(low_atan_std.keys()))
    #     self.all_trajectories_dataframe = self.all_trajectories_dataframe.loc[low_atan_selection]