#!/usr/bin/env python3

import atddm
import pandas as pd
# import numpy as np
from datetime import time
from constants import AIRPORTS, COLORS, TZONES, CODES, BEGDT, ENDDT
import seaborn as sns
import matplotlib.pyplot as plt

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
# ENDDT = BEGDT + pd.Timedelta(21, 'D')
INTERVAL = 10

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

if TRGT == 'talk':
    f, axes = plt.subplots(2, nairp//2, sharey=False)
else:
    f, axes = plt.subplots(nairp//2, 2, sharey=False)

for ax, code in zip(axes.flatten(), CODES):
    ts = m3_bin[code].fillna(0)
    ts = ts.diff().dropna()
    freq = 60*ts.index.freq.delta.components.hours +\
        ts.index.freq.delta.components.minutes
    day0 = ts.index[0]
    days = int(len(ts)/(24*60/freq))
    begdays = [day0 + pd.Timedelta(i, 'D') for i in range(0, days)]
    enddays = [d + pd.Timedelta(1, 'D') - pd.Timedelta(10, 'm')
               for d in begdays]
    for bd, ed in zip(begdays, enddays):
        ax.plot(range(24*60//freq), ts.loc[bd:ed], color='k', alpha=0.05)
    times = [time(i, j).strftime('%H:%M') for i in range(24)
             for j in range(0, 60, freq)]
    xticks = [(2+3*i)*60//freq for i in range(8)]
    ax.set_xlim(0, 24*60//freq)
    ax.set_xticks(xticks)
    ax.set_xticklabels([times[i] for i in xticks])
    ax.set_ylabel('1st order differenced daily demands')
    if TRGT == 'talk':
        ax.set_title('{:s}'.format(code))
    else:
        ax.set_title('{:s} (ICAO: {:s})'.format(AIRPORTS[code], code))

if TRGT == 'talk':
    f.set_size_inches(24, 10)
else:
    f.set_size_inches(2*nairp, 1.5*nairp)
f.savefig(PRFIX+'DailyDemand.png', dpi=300, bbox_inches='tight')
