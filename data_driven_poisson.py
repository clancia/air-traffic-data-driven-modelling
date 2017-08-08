#!/usr/bin/env python3

"""
This script defines a Poisson model with piecewise constant lambda over the day

# Fit Procedure
The time series of aggregated number of arrivals at FL240 is investigated for
changes in the arrivals regime. The arrival process is modelled as a Poisson
process with piecewise constant arrival speed.

The number of arrivals in each segment of the time series is thus modelled as
following a Poisson distribution with its own rate parameter.
The software used is R *changepoint* package, cpt.meanvar function, section 6.1
of package vignette at
http://www.lancs.ac.uk/~killick/Pub/KillickEckley2011.pdf.
For each day, a number of couples (lambda, changepoint) are estimated.

Next, the clustering algorithm DBSCAN
(http://scikit-learn.org/stable/modules/clustering.html#dbscan)
is run to identify common regimes and exclude outliers.
For each identified clusted, the centroid of the cluster is computed.
This gives the average arrival speed (lambda) and the average time of the day
when this regime starts.
"""

import numpy as np
from rpy2.robjects.packages import importr
import rpy2.robjects as robjects

import atddm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from constants import AIRPORTS, COLORS, TIMEZONES, TZONES, CODES
# import datetime

from sklearn.cluster import DBSCAN
# from sklearn import metrics
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler

# import pdb


from rpy2.robjects import pandas2ri
pandas2ri.activate()

sns.set(style="whitegrid", context='paper')

###################################
nairp = len(CODES)
f, axes = plt.subplots(nairp//2, 2, sharex=True, sharey=True)
II = np.repeat(list(range(nairp//2)), 2)
JJ = [0, 1]*(nairp//2)

HPARMS = pd.DataFrame.from_dict({
    'EDDF': [0.2, 15],
    'EGKK': [0.25, 21],
    'EGLL': [0.2, 15],
    'EHAM': [0.17, 15],
    'LEMD': [0.16, 17],
    'LFPG': [0.15, 15],
    'LGAV': [0.21, 21],
    'LIRF': [0.185, 17],
    }, orient='index')
HPARMS.columns = ['eps', 'msample']


BEGDT = pd.Timestamp(atddm.constants.BEGDT)
ENDDT = pd.Timestamp(atddm.constants.ENDDT)
INTERVAL = 10
NPERIODS = 24*60/INTERVAL
dd = atddm.load(subset=CODES)
r = robjects.r  # allows access to r object with r.
changepoint = importr('changepoint')

detected_lambdas = pd.DataFrame(columns=['icao', 'time', 'lambda'])
icount = 0

for code, i, j in zip(CODES, II, JJ):
    indx = pd.date_range(start=BEGDT, end=ENDDT,
                         freq=str(INTERVAL)+'min', tz=TZONES[code])
    binned = atddm.binarrivals(dd[code].M3_FL240,
                               interval=INTERVAL, tz=TZONES[code])[indx]
    daily = atddm.daily_avg(binned, tz=TZONES[code])

    freq = binned.index.freq.delta.components.minutes +\
        binned.index.freq.delta.components.hours * 60
    periods = int(TIMEZONES[code]/pd.Timedelta(freq, 'm'))

    values = changepoint.cpt_meanvar(binned.fillna(0).astype(int),
                                     test_stat='Poisson',
                                     method='PELT',
                                     penalty='AIC')  # pen_value='1.5*log(n)')
    pars = np.array(values.do_slot('param.est')).flatten()
    chng = (np.array(changepoint.cpts(values)).flatten()) % NPERIODS

    ax = axes[i, j]
    ax.plot(
        range(len(daily)),
        daily.mu,
        color=list(map(lambda x: x/255, COLORS[code])),
        alpha=.75
    )

    if len(chng) == 0:
        ax.set_title('{:s}: No changepoint detected'.format(code))
        continue

    X = np.vstack((chng[:-1], pars[1:-1])).T
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)

    ###########################################################################
    # Compute DBSCAN
    db = DBSCAN(eps=HPARMS.loc[code, 'eps'],
                min_samples=HPARMS.loc[code, 'msample'],
                n_jobs=2)
    db = db.fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_

    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    # Black removed and is used for noise instead.
    unique_labels = set(labels)

    XX = scaler.inverse_transform(X)
    # kolors = sns.color_palette("hls", n_clusters_)
    kolors = plt.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    for k, col in zip(unique_labels, kolors):
        alpha = 0.5
        if k == -1:
            # Black used for noise.
            col = 'k'
            alpha = 0.25
            # DO NOT PLOT NOISE
            continue

        class_member_mask = (labels == k)

        if k >= 0:
            centroid = XX[class_member_mask].mean(axis=0)
            ax.axvline(centroid[0], color=col, lw=0.25, ls='--')
            ax.axhline(centroid[1], color=col, lw=0.25, ls='--')
            detctime = '{h:02d}:{m:02d} UTC+{tz:02d}'.format(
                    h=int(centroid[0]*INTERVAL // 60),
                    m=int(centroid[0]*INTERVAL % 60),
                    tz=int(TIMEZONES[code]/pd.Timedelta(1, 'h')))
            t_hat = '{:.04f} aircraft/min'.format(round(centroid[1]/INTERVAL,
                                                  ndigits=4))
            detected_lambdas.loc[icount] = [code, detctime, t_hat]
            icount += 1

        xy = XX[class_member_mask & core_samples_mask]
        # plt.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
        #          markeredgecolor='k', markersize=6)
        ax.scatter(xy[:, 0], xy[:, 1], marker='o', c=col, s=25, alpha=alpha)

        xy = XX[class_member_mask & ~core_samples_mask]

        ax.scatter(xy[:, 0], xy[:, 1], marker='o', c=col, s=10, alpha=alpha)

        ax.set_title('{:s} (ICAO: {:s})'.format(AIRPORTS[code], code))

ax.set_xticks([int(i*NPERIODS/8) for i in range(9)])
ax.set_xticklabels(['{:02d}:{:02d}'.format(t*INTERVAL//60, t*INTERVAL % 60)
                    for t in ax.get_xticks()])
ax.set_ylim(bottom=0)

# f, ax = sns.plt.subplots()
# ax.scatter(chng[:-1], pars[1:-1])
# ax.set_xticks([int(i* NPERIODS/8) for i in range(9)])
# ax.set_xticklabels(['{:02d}:{:02d}'.format(t*INTERVAL//60, t*INTERVAL%60)
#                     for t in ax.get_xticks()])
# # ax.set_xticklabels([str(datetime.timedelta(minutes=t * INTERVAL))
#                       for t in ax.get_xticks()])
# sns.plt.show()

# f.suptitle('Data-driven Poisson process :: PELT with AIC penalty')
f.set_size_inches(10, 7.5)
f.savefig('./plots/DDPoisson.png', dpi=250, bbox_inches='tight')

detected_lambdas = detected_lambdas.set_index(['icao', 'time'])
detected_lambdas = detected_lambdas.sort_index()
ofile = open('./tables/poisson_rates.tex', 'w')
ofile.write(detected_lambdas.to_latex(column_format='ccr', escape=False))
ofile.close()

detected_lambdas.to_csv('poisson_parameters.csv')
# sns.plt.show()
