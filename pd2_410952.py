"""
Program is calculating the R0 based on simplified equation modeling initial spreading of the virus:
n_e(t) = n_e(0)*R0**t
n_e(t): number of confirmed cases in time t
Solution is based on weighted linear least square fit to logarithmized version of the problem.
"""


import pandas as pd
from matplotlib import pylab
import numpy as np
from scipy.linalg import lstsq


def read_28_days(country):
    """
    :param country: String :return: np.array Using public data regarding CoVid-19 cases worldwide returns a np.array
    of number of cases from first 28 days since the first case in given country
    """
    df = pd.read_csv('https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data'
                     '/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv')
    df = df.loc[df['Country/Region'] == country, :].sum(axis=0)[4:]  # get all data in given country
    df = df[df != 0][:28]  # retrieving first 28 days since discovering virus in the country
    return np.array(df, dtype='float64')

def weighted_linear_lstsq(l, r, w):
    """
    Compute weighted linear least squares solution to equation Ax = b given an one dimensional array of weights
    :param l: np.array, A in Ax = b equation
    :param r: np.array, b in Ax = b equation
    :param w: np.array, size = (1,n) where n is an integer
    :return: a, b: floats in linear equation y = ax + b, weighted least-squares solution
    """
    w = np.diag(w)
    wl = np.dot(w, l)
    wr = np.dot(w, r)  # redefining problem of least square with weights
    a, b = lstsq(a=wl, b=wr)[0]
    return float(a), float(b)

country = input('Provide a country name: ')
n_e = read_28_days(country)
t = np.ones((28,2))  # preparing array of time in days and ones needed for the linear equation
t[:,0] = np.arange(1,29)
logn_e = np.log(np.resize(n_e,(28,1)))

# t*logr0 + logn_e0 = logn_e(t)
logr0, logn_e0 = weighted_linear_lstsq(t,logn_e,n_e)

r0 = np.exp(logr0)
n_e0 = np.exp(logn_e0)

# visualizing data and solution
pylab.plot(n_e, 'x', label='data')
pylab.plot(n_e0*r0**t[:,0],label='weighted fit')
pylab.xlabel('days')
pylab.ylabel('confirmed cases')
pylab.legend()
pylab.show()


