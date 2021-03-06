#!/usr/bin/env python3

import atddm
import pandas as pd
# import numpy as np
from datetime import time
from math import sqrt
from constants import AIRPORTS, COLORS, TZONES, CODES, BEGDT, ENDDT
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats

TRGT = 'talk'

if TRGT == 'talk':
    sns.set(context='talk')
    PRFIX = './../publications/talk_plots/'
else:
    sns.set(style="whitegrid", context='paper')
    PRFIX = './../plots/'


def rgbfy(code):
    return list(map(lambda x: x/255, COLORS[code]))


BEGDT = pd.Timestamp(BEGDT)
ENDDT = pd.Timestamp(ENDDT)
INTERVAL = 10
ALPHA = 0.01
YMAX = 13

zval = stats.norm.ppf(1-ALPHA/2)
dd = atddm.load(subset=CODES)
m3_bin = {}
nairp = len(CODES)
CODES.sort()

for code in CODES:
    indx = pd.date_range(start=BEGDT, end=ENDDT,
                         freq=str(INTERVAL)+'min', tz=TZONES[code])
    m3_bin[code] = atddm.binarrivals(dd[code].M3_FL240,
                                     interval=INTERVAL,
                                     tz=TZONES[code])[indx].fillna(0)
daily = {}
for code in CODES:
    daily[code] = atddm.daily_avg(m3_bin[code], tz=TZONES[code])

ci = zval/sqrt(len(indx))

if TRGT == 'talk':
    f, axes = plt.subplots(2, nairp//2, sharex=True, sharey=False)
else:
    f, axes = plt.subplots(nairp//2, 2, sharex=True, sharey=False)

for ax, code in zip(axes.flatten(), CODES):
    df = daily[code]
    # AVERAGE ARRIVALS
    freq = 60*df.index.freq.delta.components.hours +\
        df.index.freq.delta.components.minutes
    # periods = int(TIMEZONES[code]/pd.Timedelta(freq, 'm'))
    # mu = np.roll(df.mu, periods)
    mu = df.mu
    # sem = np.roll(df.stermn, periods)
    sem = df.stermn
    ax.plot(range(len(df)), mu, color=rgbfy(code))
    ax.fill_between(range(len(df)), mu - zval*sem, mu + zval*sem,
                    color=rgbfy(code), alpha=.25)
    times = [time(i, j).strftime('%H:%M') for i in range(24)
             for j in range(0, 60, freq)]
    xticks = [(2+3*i)*60//freq for i in range(8)]
    if TRGT == 'talk':
        ax.set_title('{:s}'.format(code))
    else:
        ax.set_title('{:s} (ICAO: {:s})'.format(AIRPORTS[code], code))
ax.set_ylim(-0.5, YMAX)
ax.set_xlim(0, 24*60//freq)
ax.set_xticks(xticks)
ax.set_xticklabels([times[i] for i in xticks])

for ax in axes[:, 0]:
    ax.set_ylabel('Avg # arrivals by {:d} mins'.format(INTERVAL))

for ax in axes[-1]:
    for tick in ax.get_xticklabels():
        tick.set_rotation(30)

if TRGT == 'talk':
    f.set_size_inches(24, 10)
else:
    f.set_size_inches(2*nairp, 1.5*nairp)
f.savefig(PRFIX+'AvgArrivals.png', dpi=300, bbox_inches='tight')
