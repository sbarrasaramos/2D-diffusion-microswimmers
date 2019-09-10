from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import warnings

import pandas as pd
import numpy as np

import csv
import re

def mean_square_displacement():
    warnings.filterwarnings("ignore")
    # path = '..\\Data\\Example3'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190715\\ChlamyBilles_60x_2_C001H001S0001'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190726\\H2O_0%_18p5um_C001H001S0001'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190715\\ChlamyBilles_60x_C001H001S0001'
    path = 'D:\\Microswimmers2D_SBR\\Data\\190808\\ANH_Chlamy_beads_Thickness18pt_60x_C001H001S0001'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190808\\ANH_Chlamy_beads_ThicknessLessThan18pt_60x_C001H001S0001'

# ------------------------------------------------------
    """ PARAMETERS """
    fps = 50
    lens_magnification = 60
    width = 1024
    height = 1024

# ------------------------------------------------------
    itertime = [2, 4, 6, 8, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 76, 82, 88, 94, 100]
    # itertime = [2, 4, 8, 16, 40, 58]
    for t in itertime:
        # ###########################################################################
        # path2 = path + '\\NoBackground_median\\Bandpass\\Clear\\phi27\\Dt{:03d}\\abs_dydx.csv'.format(t)
        #
        # # open your csv and read as a text string
        # with open(path2, 'r') as f:
        #     my_csv_text = f.read()
        #
        # find_str = '\n,'
        # replace_str = '\n'
        # # substitute
        # my_csv_text = re.sub(find_str, replace_str, my_csv_text)
        #
        # find_str = ',,'
        # replace_str = ','
        # # substitute
        # my_csv_text = re.sub(find_str, replace_str, my_csv_text)
        #
        # find_str = 'dx,dx_turningrf,dy,dy_turningrf'
        # replace_str = 'dx,dy'
        # # substitute
        # my_csv_text = re.sub(find_str, replace_str, my_csv_text)
        #
        # # open new file and save
        # with open(path2, 'w') as f:
        #     f.write(my_csv_text)

        ###########################################################################
        with open(
                path + '\\NoBackground_median\\Clear\\Dt{:03d}\\abs_dydx.csv'.format(t)
                # path + '\\NoBackground_median\\cropped\\Bandpass\\phi03\\Dt{:03d}\\turning_dydx.csv'.format(t)
                  ) as csv_file:
            col_data = pd.read_csv(csv_file,
                                   delimiter=',',
                                   dtype={'dx': float, 'dy': float},
                                   usecols=['dx', 'dy']
                                   )

        ############################################################################
        """ <Dx2=f(Dt)> """
        col_data['dy2'] = col_data['dy'].apply(lambda x: np.abs(x**2)) # - 0.044))
        col_data['dx2'] = col_data['dx'].apply(lambda x: np.abs(x**2)) # - 0.21))
        # col_data['d'] = col_data['dx2'] + col_data['dy2']
        # col_data['d'] = col_data['d'].apply(lambda x: np.sqrt(x))

        Dx2 = ((0.425*2/3)**2)*(col_data['dy2'].sum() + col_data['dx2'].sum()) / col_data.shape[0]
        # D = (((0.425*2/3)**2)*(col_data['dx2'].sum()) / col_data.shape[0])**0.5
        print(Dx2)
        # print(D)

        ############################################################################

if __name__ == "__main__":
    mean_square_displacement()