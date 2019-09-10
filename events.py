from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import warnings

import pandas as pd
import numpy as np

import csv
import re

def events():
    warnings.filterwarnings("ignore")
    # path = '..\\Data\\Example3'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190808\\ANH_Chlamy_beads_Thickness18pt_60x_C001H001S0001'
    path = 'D:\\Microswimmers2D_SBR\\Data\\190808\\ANH_Chlamy_beads_ThicknessLessThan18pt_60x_C001H001S0001'

# ------------------------------------------------------
    itertime = [2, 4, 6, 8, 10, 16, 22, 28, 34, 40, 46, 52, 58, 64, 70, 76, 82, 88, 94, 100]
    for t in itertime:
        with open(
                path + '\\NoBackground_median\\Clear\\Dt{:03d}\\abs_dydx.csv'.format(t)
                # path + '\\NoBackground_median\\cropped\\Bandpass\\phi27\\Dt{:03d}\\turning_dydx.csv'.format(t)
                  ) as csv_file:
            col_data = pd.read_csv(csv_file,
                                   delimiter=',',
                                   dtype={'dx': float},
                                   usecols=['dx']
                                   )

        ev = col_data.shape[0]
        print(ev)

if __name__ == "__main__":
    events()