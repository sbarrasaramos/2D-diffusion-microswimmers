from __future__ import division, unicode_literals, print_function  # for compatibility with Python 2 and 3

import warnings

import pandas as pd
import numpy as np
import scipy

import matplotlib

import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

from scipy.stats import norm

matplotlib.rcParams['text.usetex'] = True


def distribution(x, lim_g, lim_pl, mu, sigma, exponent, coef_g, coef_pl, alpha, k):
    if x <= lim_g:
        dist = coef_g * gausspdf(x, mu, sigma)
    elif x >= lim_pl:
        dist = coef_pl * powerlawpdf(x, exponent, k)
    else:
        dist = coef_pl * x_powerlawpdf(
            x, exponent, lim_g, lim_pl, alpha, k
        ) + coef_g * x_gausspdf(
            x, mu, sigma, lim_g, lim_pl, alpha
        )
    return dist


def gausspdf(x, mu, sigma):
    return 2 * norm(mu, sigma).pdf(x)


def x_gausspdf(x, mu, sigma, lim_g, lim_pl, alpha):
    return (1 - ((x - lim_g) / (lim_pl - lim_g)) ** alpha) * gausspdf(x, mu, sigma)


def powerlawpdf(x, exponent, k):
    return (x - k) ** exponent


def x_powerlawpdf(x, exponent, lim_g, lim_pl, alpha, k):
    return ((x - lim_g) / (lim_pl - lim_g)) ** alpha * powerlawpdf(x, exponent, k)


def get_scaling_coef(muh0, sigmah0, exponent, lim_g, lim_pl, alpha, k, coef_g):
    g1, g1e = scipy.integrate.quad(gausspdf, 0, lim_g, args=(muh0, sigmah0))
    pl1, pl1e = scipy.integrate.quad(powerlawpdf, lim_pl, 1000, args=(exponent, k))
    g2, g2e = scipy.integrate.quad(x_gausspdf, lim_g, lim_pl, args=(muh0, sigmah0, lim_g, lim_pl, alpha))
    pl2, pl2e = scipy.integrate.quad(x_powerlawpdf, lim_g, lim_pl, args=(exponent, lim_g, lim_pl, alpha, k))

    coef_pl = (1 - coef_g * g1 - coef_g * g2) / (pl1 + pl2)
    return coef_pl


def series_statistics():
    warnings.filterwarnings("ignore")
    # path = '..\\Data\\Example3'
    path = 'D:\\Microswimmers2D_SBR\\Data\\190808\\ANH_Chlamy_beads_Thickness18pt_60x_C001H001S0001'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190808\\ANH_Chlamy_beads_ThicknessLessThan18pt_60x_C001H001S0001'

    # ------------------------------------------------------
    ############################################################################
    """ Figure settings """
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax.margins(x=0)

    plt.ylim([5 * 10 ** (-3), 1 * 10 ** (0)])
    plt.xlim([0.6, 4.0])
    # plt.xlim([5 * 10 ** (-2), 5 * 10 ** (0)])

    plt.title('Absolute reference frame')
    plt.xlabel('$\|\Delta x\| / (<\Delta x ^ 2>)^{1/2} = 2.42 = 1.04$')
    plt.ylabel('$P(\|\Delta x\|, \Delta t) / P(0, \Delta t)$')

    """ Reading data """
    ###########################################################################
    itertime = [2, 4, 8, 16, 40]
    iterfactor = [
        0.28121261171254236,
        0.5130885574168105,
        0.9395741289390831,
        1.6757245894968638,
        3.415859754082798
    ]

    for counter, t in enumerate(itertime):
        with open(path + '\\NoBackground_median\\Clear\\Dt{:03d}\\abs_dydx.csv'.format(t)) as csv_file:
            # with open(path + '\\NoBackground_median\\cropped\\Bandpass\\phi07\\Dt004\\turning_dydx.csv') as csv_file:
            col_data = pd.read_csv(csv_file,
                                   delimiter=',',
                                   dtype={'dx': float, 'dy': float},
                                   usecols=['dx', 'dy']
                                   )
        ############################################################################
        """ Absolute values, to um """
        col_data['dx_um'] = col_data['dx'].apply(lambda x: np.abs(x) * 0.425 / 3 * 2)
        # col_data.append(colpre_data['dy'].apply(lambda x: np.abs(x) * 0.425 / 3 * 2))

        ############################################################################
        """ Calculate pdf (absolute values) """
        x = (np.histogram(col_data['dx_um'], bins=30))[1]
        x1 = x[1:]
        x2 = x[:-1]
        x = (x1 + x2) / 2
        xp = x
        w = abs(x1 - x2)
        px = (np.histogram(col_data['dx_um'], bins=30))[0] / len(col_data['dx_um']) / w[0]

        normal = norm(0, 3.4).pdf(xp)/norm(0, 3.4).pdf(0)
        xp = xp/iterfactor[counter]
        px = px/px[0]
        powerlaw = xp[13:] ** (-4) * 2

        ############################################################################
        """ Plot histogram and pdf """
        # plt.hist(col_data['dx_um'], 100, density=True, label='Histogram')
        # plt.plot(xp, px, label='Prob density')
        # plt.semilogy(xp, px, ':', label='Prob density $\Delta_t =$ {:03d} s'.format(t))
        plt.loglog(xp, px, 's', markersize=3, label='Prob density $\Delta_t =$ {:03d} s'.format(t))

    plt.loglog(xp, normal, 'k:', label='Gaussian distrbution')
    plt.loglog(xp[13:], powerlaw, 'k--', label='$ \propto \Delta x ^{- a}$')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    series_statistics()