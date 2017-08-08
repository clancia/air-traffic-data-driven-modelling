#!/usr/bin/env python3

import atddm
import pandas as pd
# import numpy as np
# from datetime import time
from math import sqrt
from constants import AIRPORTS, COLORS, TZONES, CODES
import seaborn as sns
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from scipy import stats
# import pdb

sns.set(style="whitegrid", context='paper')


def rgbfy(code):
    return list(map(lambda x: x/255, COLORS[code]))


BEGDT = pd.Timestamp(atddm.constants.BEGDT)
ENDDT = pd.Timestamp(atddm.constants.ENDDT)
INTERVAL = 10
ALPHA = 0.01

zval = stats.norm.ppf(1-ALPHA/2)
dd = atddm.load(subset=CODES)
m3_bin = {}
nairp = len(CODES)
CODES.sort()

for code in CODES:
    indx = pd.date_range(start=BEGDT, end=ENDDT,
                         freq=str(INTERVAL)+'min',
                         tz=TZONES[code])
    m3_bin[code] = atddm.binarrivals(dd[code].M3_FL240,
                                     interval=INTERVAL,
                                     tz=TZONES[code])[indx].fillna(0)

ci = zval/sqrt(len(indx))

lag1acf = pd.Series(index=CODES)
adftest = pd.DataFrame(index=CODES, columns=['adf', 'adf*', 'p-val'])

f, axes = plt.subplots(nairp//2, 2, sharey=False)

for ax, code in zip(axes.flatten(), CODES):
    ts = m3_bin[code].fillna(0)
    freq = 60*ts.index.freq.delta.components.hours +\
        ts.index.freq.delta.components.minutes
    ndays = 3
    lagmax = ndays*24*60//freq
    ts = ts.diff().dropna()
    test = adfuller(ts[:24*60//freq], regression='nc')
    adftest.loc[code, :] = [test[0], test[4]['1%'], test[1]]
    lag1acf.loc[code] = ts[:24*60//freq].autocorr(lag=1)
    autocorr = [ts.autocorr(lag=i) for i in range(lagmax+1)]
    ax.fill_between(range(-3, lagmax+3), -ci, ci, color=rgbfy(code), alpha=.15)
    ax.axhline(ci, color=rgbfy(code), lw=0.25, ls='--')
    ax.axhline(-ci, color=rgbfy(code), lw=0.25, ls='--')
    ax.axhline(0, color=rgbfy(code))
    # ax.plot(range(len(autocorr)), autocorr, color=rgbfy(code), lw=0.5)
    markerline, stemlines, baseline = ax.stem(range(len(autocorr)),
                                              autocorr,
                                              markerfmt=' ',
                                              basefmt=' ')
    plt.setp(stemlines, 'color', rgbfy(code))
    ax.set_xlim(-2.5, lagmax+2.5)
    ax.set_ylim(-0.2, 0.2)
    ndays += 1
    ax.set_xticks([i*24*60//freq for i in range(ndays)])
    ax.set_xticklabels(['{:d} days'.format(i) for i in range(ndays)])
    ax.set_title('{:s} (ICAO: {:s})'.format(AIRPORTS[code], code))
    ax.set_ylabel('ACF({:s})'.format(code))

f.set_size_inches(2*nairp, 1.5*nairp)
f.savefig('./plots/Autocorr.png', dpi=300, bbox_inches='tight')

print('lag-1 autocorrelations')
print(lag1acf)
print('\n' + '*'*30 + '\n')
print('Dickey-Fuller test')
print(adftest)
