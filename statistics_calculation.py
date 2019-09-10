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
        dist = coef_g*gausspdf(x, mu, sigma)
    elif x >= lim_pl:
        dist = coef_pl*powerlawpdf(x, exponent, k)
    else:
        dist = coef_pl*x_powerlawpdf(
            x, exponent, lim_g, lim_pl, alpha, k
        ) + coef_g*x_gausspdf(
            x, mu, sigma, lim_g, lim_pl, alpha
        )
    return dist


def gausspdf(x, mu, sigma):
    return 2 * norm(mu, sigma).pdf(x)


def x_gausspdf(x, mu, sigma, lim_g, lim_pl, alpha):
    return (1 - ((x-lim_g)/(lim_pl-lim_g))**alpha) * gausspdf(x, mu, sigma)


def powerlawpdf(x, exponent, k):
    return (x-k)**exponent


def x_powerlawpdf(x, exponent, lim_g, lim_pl, alpha, k):
    return ((x - lim_g)/(lim_pl-lim_g))**alpha * powerlawpdf(x, exponent, k)


def get_scaling_coef(muh0, sigmah0, exponent, lim_g, lim_pl, alpha, k, coef_g):
    g1, g1e = scipy.integrate.quad(gausspdf, 0, lim_g, args=(muh0, sigmah0))
    pl1, pl1e = scipy.integrate.quad(powerlawpdf, lim_pl, 1000, args=(exponent, k))
    g2, g2e = scipy.integrate.quad(x_gausspdf, lim_g, lim_pl, args=(muh0, sigmah0, lim_g, lim_pl, alpha))
    pl2, pl2e = scipy.integrate.quad(x_powerlawpdf, lim_g, lim_pl, args=(exponent, lim_g, lim_pl, alpha, k))

    coef_pl = (1 - coef_g*g1 - coef_g*g2)/(pl1 + pl2)
    return coef_pl


def statistics_calculation():
    warnings.filterwarnings("ignore")
    path = '..\\Data\\Example3'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190808\\ANH_Chlamy_beads_Thickness18pt_60x_C001H001S0001'
    # path = 'D:\\Microswimmers2D_SBR\\Data\\190808\\ANH_Chlamy_beads_ThicknessLessThan18pt_60x_C001H001S0001'

    # ------------------------------------------------------
    """ Reading data """
    ###########################################################################

    # with open(path + '\\NoBackground_median\\Clear\\Dt010\\abs_dydx.csv') as csv_file:
    with open(path + '\\NoBackground_median\\cropped\\Bandpass\\phi07\\Dt004\\turning_dydx.csv') as csv_file:
        col_data = pd.read_csv(csv_file,
                               delimiter=',',
                               dtype={'dx': float, 'dy': float},
                               usecols=['dx', 'dy']
                               )
    ############################################################################
    """ Absolute values, to um """
    col_data['dx_um'] = col_data['dx'].apply(lambda x: np.abs(x) * 0.425) #/3*2)
    # col_data['dy_um'] = col_data['dy'].apply(lambda x: x * 0.425/3*2)
    ############################################################################
    """ Calculate pdf (absolute values) """
    x = (np.histogram(col_data['dx_um'], bins=500))[1]
    x1 = x[1:]
    x2 = x[:-1]
    x = (x1 + x2) / 2
    xp = x
    w = abs(x1 - x2)
    px = (np.histogram(col_data['dx_um'], bins=500))[0] / len(col_data['dx_um']) / w[0]

    ############################################################################
    """ Figure settings """
    fig, ax = plt.subplots()
    plt.subplots_adjust(left=0.25, bottom=0.25)
    ax.margins(x=0)

    plt.ylim([5 * 10 ** (-2), 5 * 10 ** (0)])
    plt.xlim([0, 0.8])
    # plt.xlim([5 * 10 ** (-2), 5 * 10 ** (0)])

    plt.title('Absolute reference frame')
    plt.xlabel('$\|\Delta x\| (\mu m)$')
    plt.ylabel('$P(\|\Delta x\|, \Delta t=0.76s)$')

    ############################################################################
    """ Plot histogram and pdf """
    # plt.hist(col_data['dx_um'], 100, density=True, label='Histogram')
    # plt.plot(xp, px, label='Prob density')
    plt.semilogy(xp, px, label='Prob density')
    # plt.loglog(xp, px, label='Prob density')

    ############################################################################
    """ Calculate ad hoc pdf (Gauss + PowerLaw) """
    col_data['norm'] = col_data['dx'].apply(lambda x: x * 0.425) #/3*2)

    muh0 = 0
    sigmah0 = np.std(col_data['norm'])
    print('-----')
    print(sigmah0)
    print('-----')

    k = 0
    alpha = 1
    lim_g = 2
    lim_pl = 3
    exponent = -5

    coef_g = px[5] / (2 * norm(muh0, sigmah0).pdf(xp[5]))
    coef_pl = get_scaling_coef(muh0, sigmah0, exponent, lim_g, lim_pl, alpha, k, coef_g)

    toto = [distribution(
        v,
        lim_g=lim_g,
        lim_pl=lim_pl,
        mu=muh0,
        sigma=sigmah0,
        exponent=exponent,
        coef_g=coef_g,
        coef_pl=coef_pl,
        alpha=alpha,
        k=k
    ) for v in xp]

    l, = plt.plot(xp, toto, '--r', label='Gaussian distribution') # label='Ad hoc distribution')
    # l, = plt.loglog(xp, toto, '--r', label='Gaussian distribution') # label='Ad hoc distribution')

    plt.legend()

    axcolor = 'lightgoldenrodyellow'

    # noinspection PyTypeChecker
    axlimg = plt.axes([0.25, 0.03, 0.25, 0.03], facecolor=axcolor)
    slimg_disp = Slider(axlimg, 'Lim Gauss', 1.0, 3.0, valinit=lim_g, valstep=0.01)

    # noinspection PyTypeChecker
    axlimpl = plt.axes([0.65, 0.03, 0.25, 0.03], facecolor=axcolor)
    slimpl_disp = Slider(axlimpl, 'Lim PL', 2.0, 4.0, valinit=lim_pl, valstep=0.01)

    # noinspection PyTypeChecker
    axsigmah = plt.axes([0.25, 0.07, 0.25, 0.03], facecolor=axcolor)
    ssigmah_factor = Slider(axsigmah, 'Sigma factor', 0.0, 2.0, valinit=1.0, valstep=0.01)

    # noinspection PyTypeChecker
    axalphah = plt.axes([0.65, 0.07, 0.25, 0.03], facecolor=axcolor)
    salphah = Slider(axalphah, 'Trans', 0, 10.0, valinit=alpha, valstep=0.01)

    # noinspection PyTypeChecker
    axexph = plt.axes([0.25, 0.11, 0.25, 0.03], facecolor=axcolor)
    sexph = Slider(axexph, 'Power law', -6.0, -3.0, valinit=exponent, valstep=0.1)

    # noinspection PyTypeChecker
    axkh = plt.axes([0.65, 0.11, 0.25, 0.03], facecolor=axcolor)
    skh = Slider(axkh, 'k', -0.1, 0.1, valinit=k, valstep=0.001)

    def update(val):
        sigmah = sigmah0*ssigmah_factor.val
        lim_gh = slimg_disp.val
        lim_plh = slimpl_disp.val
        exponenth = sexph.val
        alphah = salphah.val
        kh = skh.val

        coef_gh = px[5] / (2 * norm(muh0, sigmah).pdf(xp[5]))
        coef_plh = get_scaling_coef(muh0, sigmah, exponenth, lim_gh, lim_plh, alphah, kh, coef_gh)

        l.set_ydata(
            [distribution(
                v,
                lim_g=lim_gh,
                lim_pl=lim_plh,
                mu=muh0,
                sigma=sigmah,
                exponent=exponenth,
                coef_g=coef_gh,
                coef_pl=coef_plh,
                alpha=alphah,
                k=kh
            ) for v in xp]
        )
        fig.canvas.draw_idle()

        # print(sigmah)
        # print(coef_gh)
        # print(coef_plh)

    ssigmah_factor.on_changed(update)
    slimg_disp.on_changed(update)
    slimpl_disp.on_changed(update)
    sexph.on_changed(update)
    salphah.on_changed(update)
    skh.on_changed(update)

    # noinspection PyTypeChecker
    resetax = plt.axes([0.05, 0.15, 0.1, 0.04])
    button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

    def reset(event):
        ssigmah_factor.reset()
        slimg_disp.reset()
        slimpl_disp.reset()
        sexph.reset()
        salphah.reset()
        skh.reset()
    button.on_clicked(reset)

    plt.show()


if __name__ == "__main__":
    statistics_calculation()
