from typing import Any, Union

import scipy.stats as st
from scipy.special import erf
import numpy as np

from scipy.stats import beta
from scipy.stats import norm

class norm_powertails_gen(st.rv_continuous):

    def _pdf(self, x, mu, sigma, a, b, lim, sc):

        rv_norm = norm(mu, sigma)
        rv_beta = norm(a, b, lim, sc)

        y = x - lim
        sup = (x > (mu+lim))
        inf = (x < (mu-lim))
        center = (abs(x-mu) <= lim)
        pdf = rv_beta.pdf(y)*sup + rv_beta.pdf(-y)*inf + rv_norm.pdf(x)*center

        return pdf

    def _argcheck(self, mu, sigma, a, b, lim, sc):

        sigma_bool = sigma > 0
        lim_bool = mu < lim < 10
        a_bool = 3 < a < 5
        b_bool = b > 0

        all_bool = sigma_bool & lim_bool & a_bool & b_bool

        return all_bool

    # def _pdf(self, x, sigma, limit, a):
    #
    #     mu = (-1+erf(-limit/(sigma*np.sqrt(2))))*((1-a)*sigma*np.sqrt(2*np.pi))/(2*np.exp(-limit/(2*sigma**2)))-limit
    #     b = 0.5*(a-1)*(limit+mu)**(a-1)*(erf(limit/(np.sqrt(2)*sigma))+1)
    #
    #     pdf_norm = 1 / (sigma * np.sqrt(2 * np.pi)) * np.exp(-(x - mu) ** 2 / (2 * sigma ** 2))
    #     pdf_power = b * x ** (-a)
    #     pdf = pdf_power # *(abs(x) > (mu + limit)) + pdf_norm*(abs(x) < (mu + limit))
    #
    #     return pdf
    #
    # def _argcheck(self, sigma, limit, a):
    #
    #     sigma_bool = sigma > 0
    #     limit_bool = 0 < limit < 10
    #     a_bool = a > 1
    #
    #     all_bool = sigma_bool & limit_bool & a_bool
    #
    #     return all_bool