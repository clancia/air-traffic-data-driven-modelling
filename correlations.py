#!/usr/bin/env python3

import atddm
import pandas as pd
import numpy as np
import numpy.random as npr
from datetime import time
# from math import ceil
import seaborn as sns
# import matplotlib.pyplot as plt
from constants import AIRPORTS, COLORS, TZONES, CODES
from math import sqrt
from scipy import stats

# import pdb

sns.set(style='whitegrid', context='paper')


def timetofloat(t, freq):
    return t.time().hour*60//freq + t.time().minute/freq


def rho2z(r):
    return np.log((1+r)/(1-r))


def z2rho(z):
    return (np.exp(2*z)-1)/(np.exp(2*z)+1)


BEGDT = pd.Timestamp(atddm.constants.BEGDT)
ENDDT = pd.Timestamp(atddm.constants.ENDDT)
INTERVAL = 10
# NREPS = 300
ALPHA = 0.05
zval = stats.norm.ppf(1-ALPHA/2)
dd = atddm.load(subset=CODES)
m3_bin = {}
m1_bin = {}
psra_bin = {}
nairp = len(CODES)

PRFX = './plots/'

npr.seed()

midnight = pd.Timestamp('00:00:00')
sta_times = [midnight + pd.Timedelta(t*60, unit='s')
             for t in range(0, 24*60, INTERVAL)]
end_times = [midnight + pd.Timedelta((INTERVAL+t)*60-1, unit='s')
             for t in range(0, 24*60, INTERVAL)]
slices = [(a.time(), b.time()) for a, b in zip(sta_times, end_times)]

for code in CODES:
    df = dd[code]
    tz = TZONES[code]
    indx = pd.date_range(start=BEGDT, end=ENDDT,
                         freq=str(INTERVAL)+'min',
                         tz=tz)
    m3_bin[code] = atddm.binarrivals(df.M3_FL240,
                                     interval=INTERVAL,
                                     tz=tz)[indx].fillna(0)
    m3_bin[code].index = m3_bin[code].index.tz_localize(None)
    m1_bin[code] = atddm.binarrivals(df.M1_FL240,
                                     interval=INTERVAL,
                                     tz=tz)[indx].fillna(0)
    m1_bin[code].index = m1_bin[code].index.tz_localize(None)
    tmp = df.M1_FL240 + np.array(df.delay.sample(n=len(df), replace=True))
    psra_bin[code] = atddm.binarrivals(tmp,
                                       interval=INTERVAL,
                                       tz=tz)[indx].fillna(0)
    psra_bin[code].index = psra_bin[code].index.tz_localize(None)

daily = {}
daily_psra = {}
for code in CODES:
    tz = TZONES[code]
    daily[code] = atddm.daily_avg(m3_bin[code], tz=tz)
    daily[code].index = daily[code].index.tz_localize(None)
    daily_psra[code] = atddm.daily_avg(psra_bin[code], tz=tz)
    daily_psra[code].index = daily_psra[code].index.tz_localize(None)

airp_parm = pd.read_csv('poisson_parameters.csv')
airp_parm['time'] = airp_parm['time'].apply(lambda x:
                                            pd.Timestamp(x.split(' ')[0]))
airp_parm['lambda'] = airp_parm['lambda'].apply(lambda x:
                                                float(x.split(' ')[0]))
airp_parm = {key: list(zip(df['lambda'], df['time']))
             for key, df in airp_parm.groupby('icao')}
# airp_parm = {
#     'EDDF': [(0.6427, pd.Timestamp('02:44:00')),
#              (0.0892, pd.Timestamp('19:18:00'))],
#     'EGKK': [(0.3507, pd.Timestamp('04:58:00')),
#              (0.0658, pd.Timestamp('22:07:00'))],
#     'EGLL': [(0.6675, pd.Timestamp('04:16:00')),
#              (0.0741, pd.Timestamp('20:09:00'))],
#     'EGSS': [(0.2192, pd.Timestamp('04:33:00')),
#              (0.2322, pd.Timestamp('07:19:00')),
#              (0.0333, pd.Timestamp('22:13:00'))],
#     'EHAM': [(0.5721, pd.Timestamp('03:30:00')),
#              (1.1266, pd.Timestamp('04:21:00')),
#              (0.5371, pd.Timestamp('05:41:00')),
#              (0.1280, pd.Timestamp('19:17:00'))],
#     'LIRF': [(0.5013, pd.Timestamp('04:08:00')),
#              (0.0763, pd.Timestamp('19:06:00'))]
# }

# simul_arrivals = pd.DataFrame(index=sta_times)
# for k in airp_parm.keys():
#     simul_arrivals[k] = 0

# simul_df = []
# for i in range(NREPS):
#     tmp = pd.DataFrame(index=sta_times)
#     for k, v in airp_parm.items():
#         tmp[k] = 0
#         landas = [INTERVAL * x[0] for x in v]
#         landas = [landas[-1]] + landas
#         times = [pd.Timestamp('00:00:00')] + [x[1] for x in v] +\
#             [pd.Timestamp('23:59:59')]
#         times = list(zip(times[:-1], times[1:]))
#         for l, t in zip(landas, times):
#             start = tmp.index.searchsorted(t[0])
#             stop = tmp.index.searchsorted(t[1])
#             tmp.ix[start:stop, k] = npr.poisson(lam=l, size=stop-start)
#     simul_df.append(tmp)
#     simul_arrivals = simul_arrivals + tmp
# simul_arrivals = simul_arrivals/NREPS

# simul_arrivals = simul_arrivals.sort_index(axis=1)
# simul_arrivals.columns = [c + '_simul' for c in simul_arrivals.columns]

# for c in CODES:
#     simul_arrivals[c] = daily[c]['mu']
#     simul_arrivals[c + '_psra'] = daily_psra[c]['mu']
# colors = sns.color_palette('colorblind')
# freq = (simul_arrivals.index[1] - simul_arrivals.index[0]).components.minutes
freq = INTERVAL
times = [time(i, j).strftime('%H:%M') for i in range(24)
         for j in range(0, 60, freq)]
xticks = [(2+3*i)*60//freq for i in range(8)]

# nr = ceil(len(CODES)/2)
# f, axes = plt.subplots(nr, 2, sharex=True, sharey=True, figsize=(14, 2+4*nr))
f, axes = sns.plt.subplots(nairp//2, 2,
                           sharex=True,
                           sharey=True,
                           figsize=(10, 7.5))

for ax, code in zip(axes.flatten(), CODES):
    kolor = list(map(lambda x: x/255, COLORS[code]))
    ax.plot(range(24*60//freq),
            daily[code]['mu'],
            color=kolor,
            linestyle='None',
            marker='o',
            markersize=1,
            label='data',
            markerfacecolor=kolor,
            markeredgewidth=1,
            markeredgecolor=kolor)
    xval = [0] + [timetofloat(tpl[1], freq) for tpl in airp_parm[code]] +\
        [24*60//freq]
    yval = [tpl[0]*freq for tpl in airp_parm[code]]
    yval = [yval[-1], yval[-1]] + yval
    # ax.plot(range(len(simul_arrivals)),
    #         simul_arrivals[code + '_simul'],
    #         color=kolor,
    #         linestyle='-',
    #         linewidth=1,
    #         label='Poisson')
    # ax.plot(range(len(simul_arrivals)),
    #         simul_arrivals[code + '_psra'],
    #         color=kolor,
    #         linestyle='-',
    #         linewidth=1,
    #         label='PSRA')
    ax.plot(range(24*60//freq),
            daily_psra[code]['mu'],
            color=kolor,
            linestyle='None',
            marker='d',
            markersize=2,
            label='PSRA',
            markerfacecolor='None',
            markeredgewidth=1,
            markeredgecolor=kolor)
    ax.step(xval,
            yval,
            color=kolor,
            linestyle='-',
            linewidth=1,
            label='Poisson')
    legend = ax.legend(loc='upper left')
    ax.set_title('{:s} (ICAO: {:s})'.format(AIRPORTS[code], code))
    ax.set_xlim(0, 24*60//freq)
    ax.set_xticks(xticks)
    ax.set_xticklabels([times[i] for i in xticks])
# f.suptitle('Average daily arrivals per intervals ' +\
#   'of {:d} mins'.format(INTERVAL))
f.savefig(PRFX + 'mean_simul_arrivals.png', dpi=300, bbox_inches='tight')

# f, axes = plt.subplots(1,3, sharex=True, sharey=True, figsize=(16,9))
# for ax, code, kolor in zip(axes.flatten(),
#                            ['EDDF', 'EGLL', 'EHAM'],
#                            colors[:3]):
#     ax.plot(range(len(simul_arrivals)),
#             simul_arrivals[code],
#             color=kolor,
#             linestyle='None',
#             marker='o',
#             markersize=3,
#             label='data',
#             markerfacecolor='None',
#             markeredgewidth=1,
#             markeredgecolor=kolor)
#     ax.plot(range(len(simul_arrivals)),
#             simul_arrivals[code + '_simul'],
#             color=kolor,
#             linestyle='-',
#             linewidth=1,
#             label='Poisson')
#     ax.plot(range(len(simul_arrivals)),
#             simul_arrivals[code + '_psra'],
#             color=kolor,
#             linestyle=':',
#             linewidth=1,
#             label='PSRA')
#     legend = ax.legend(loc='upper right')
#     ax.set_title('{:s}'.format(code))
#     ax.set_xlim(0,24*60//freq)
#     ax.set_xticks(xticks)
#     ax.set_xticklabels([times[i] for i in xticks])
# # f.suptitle('Average daily arrivals per intervals ' +\
#   'of {:d} mins'.format(INTERVAL))
# f.savefig(PRFX + 'mean_simul_arrivals_3.png', dpi=300, bbox_inches='tight')

# simul_corr = pd.DataFrame(index=sta_times)
# for code in CODES:
#     tmp = np.array([df[code] for df in simul_df])
#     tmp = tmp.T
#     tmp = np.vstack((tmp, tmp[0,:]))
#     nrows = tmp.shape[0]
#     corr = []
#     for a, b in zip(range(nrows-1), range(1, nrows)):
#         corr.append(((tmp[a]*tmp[b]).mean() - tmp[a].mean()*tmp[b].mean())
#                     /(tmp[a].std()*tmp[b].std()))
#     simul_corr[code] = corr

correlations = pd.DataFrame(index=sta_times)
for code, ts in m3_bin.items():
    tmp = []
    for sa, sb in zip(slices, slices[1:]+[slices[0]]):
        tsa = ts.between_time(sa[0], sa[1])
        tsb = ts.between_time(sb[0], sb[1])
        tmp.append(np.corrcoef(tsa, tsb)[0, 1])
        # tmp.append(((np.array(tsa)*np.array(tsb)).mean() - tsa.mean()*
        #             tsb.mean())/(tsa.std()*tsb.std()))
    correlations[code] = tmp
    tmp = np.tanh(tmp)
    n = len(tsa)
    correlations[code+'lowerCI'] = np.arctanh(tmp + zval/sqrt(n-3))
    correlations[code+'upperCI'] = np.arctanh(tmp - zval/sqrt(n-3))

correlations_m1 = pd.DataFrame(index=sta_times)
for code, ts in m1_bin.items():
    tmp = []
    for sa, sb in zip(slices, slices[1:]+[slices[0]]):
        tsa = ts.between_time(sa[0], sa[1])
        tsb = ts.between_time(sb[0], sb[1])
        tmp.append(np.corrcoef(tsa, tsb)[0, 1])
    correlations_m1[code] = tmp
    tmp = np.tanh(tmp)
    n = len(tsa)
    correlations_m1[code+'lowerCI'] = np.arctanh(tmp + zval/sqrt(n-3))
    correlations_m1[code+'upperCI'] = np.arctanh(tmp - zval/sqrt(n-3))

correlations_psra = pd.DataFrame(index=sta_times)
for code, ts in psra_bin.items():
    tmp = []
    for sa, sb in zip(slices, slices[1:]+[slices[0]]):
        tsa = ts.between_time(sa[0], sa[1])
        tsb = ts.between_time(sb[0], sb[1])
        tmp.append(np.corrcoef(tsa, tsb)[0, 1])
    correlations_psra[code] = tmp
    tmp = np.tanh(tmp)
    n = len(tsa)
    correlations_psra[code+'lowerCI'] = np.arctanh(tmp + zval/sqrt(n-3))
    correlations_psra[code+'upperCI'] = np.arctanh(tmp - zval/sqrt(n-3))

# f, axes = plt.subplots(6,2, sharex=True, sharey=True, figsize=(9,12))
# kolors = sns.color_palette(n_colors=6)
# for code, ax, k in zip(CODES, axes, kolors):
#     correlations[code].plot(ax=ax[0],
#                             color=k,
#                             linestyle='None',
#                             marker='o',
#                             markersize=3,
#                             markerfacecolor='None',
#                             markeredgewidth=1,
#                             markeredgecolor=k)
#     ax[0].axhline(color=k)
#     ax[0].set_title('{:s}: correlations from data'.format(code))
#     simul_corr[code].plot(ax=ax[1],
#                             color=k,
#                             linestyle='None',
#                             marker='o',
#                             markersize=3,
#                             markerfacecolor='None',
#                             markeredgewidth=1,
#                             markeredgecolor=k)
#     ax[1].axhline(color=k)
#     ax[1].set_title('{:s}: correlations from simulations'.format(code))
# ax[1].set_ylim(-1,1)
# f.suptitle('Correlation of the arrivals in two consecutive intervals ' +\
#   'of {:d} mins'.format(INTERVAL))
# f.savefig(PRFX + 'correlations_true-pois.png', dpi=300)

# f, axes = plt.subplots(6,2, sharex=True, sharey=True, figsize=(9,12))
# kolors = sns.color_palette(n_colors=6)
# for code, ax, k in zip(CODES, axes, kolors):
#     correlations_m1[code].plot(ax=ax[0],
#                                color=k,
#                                linestyle='None',
#                                marker='o',
#                                markersize=3,
#                                markerfacecolor='None',
#                                markeredgewidth=1,
#                                markeredgecolor=k)
#     ax[0].axhline(color=k)
#     ax[0].set_title('{:s}: correlations from M1-data'.format(code))
#     correlations_psra[code].plot(ax=ax[1],
#                                color=k,
#                                linestyle='None',
#                                marker='o',
#                                markersize=3,
#                                markerfacecolor='None',
#                                markeredgewidth=1,
#                                markeredgecolor=k)
#     ax[1].axhline(color=k)
#     ax[1].set_title('{:s}: correlations from PSRA-like'.format(code))
# ax[1].set_ylim(-1,
#                                )
# f.suptitle('Correlation of the arrivals in two consecutive intervals ' +\
#   'of {:d} mins'.format(INTERVAL))
# f.savefig(PRFX + 'correlations_m1-psra.png', dpi=300)

# f, axes = plt.subplots(nr, 2, sharex=True, sharey=True, figsize=(14, 2+4*nr))
f, axes = sns.plt.subplots(nairp//2, 2,
                           sharex=True,
                           sharey=True,
                           figsize=(10, 7.5))
for code, ax in zip(CODES, axes.flatten()):
    k = list(map(lambda x: x/255, COLORS[code]))
    yerr = [correlations[code]-correlations[code+'lowerCI'],
            correlations[code+'upperCI']-correlations[code]]
    # correlations[code].plot(ax=ax,
    #                         color=k,
    #                         linestyle='None',
    #                         marker='o',
    #                         markersize=3,
    #                         markerfacecolor='None',
    #                         markeredgewidth=1,
    #                         markeredgecolor=k)
    ax.errorbar(range(24*60//freq),
                correlations[code],
                yerr=yerr,
                color=k,
                fmt='o',
                elinewidth=0.5,
                markersize=1.5)
    ax.axhline(color=k, lw=0.5)
    ax.set_xlim(-1, 24*60//freq+1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([times[i] for i in xticks])
    ax.set_title('{:s}: correlations from data'.format(code))
ax.set_ylim(-1, 1)
# f.suptitle('Correlation of the arrivals in two consecutive intervals ' +\
#   'of {:d} mins'.format(INTERVAL))
f.savefig(PRFX + 'correlations_true.png', dpi=300, bbox_inches='tight')

# f, axes = plt.subplots(nr, 2, sharex=True, sharey=True, figsize=(14, 2+4*nr))
f, axes = sns.plt.subplots(nairp//2, 2,
                           sharex=True,
                           sharey=True,
                           figsize=(10, 7.5))
for code, ax in zip(CODES, axes.flatten()):
    k = list(map(lambda x: x/255, COLORS[code]))
    yerr = [correlations_m1[code]-correlations_m1[code+'lowerCI'],
            correlations_m1[code+'upperCI']-correlations_m1[code]]
    # correlations_m1[code].plot(ax=ax,
    #                            color=k,
    #                            linestyle='None',
    #                            marker='o',
    #                            markersize=3,
    #                            markerfacecolor='None',
    #                            markeredgewidth=1,
    #                            markeredgecolor=k,
    #                            yerr=yerr)
    ax.errorbar(range(24*60//freq),
                correlations_m1[code],
                yerr=yerr,
                color=k,
                fmt='o',
                elinewidth=0.5,
                markersize=1.5)
    ax.axhline(color=k, lw=0.5)
    ax.set_xlim(-1, 24*60//freq+1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([times[i] for i in xticks])
    ax.set_title('{:s}: correlations from M1-data'.format(code))
ax.set_ylim(-1, 1)
# f.suptitle('Correlation of the arrivals in two consecutive intervals ' +\
#   'of {:d} mins'.format(INTERVAL))
f.savefig(PRFX + 'correlations_m1.png', dpi=300, bbox_inches='tight')

# f, axes = plt.subplots(nr, 2, sharex=True, sharey=True, figsize=(14, 2+4*nr))
f, axes = sns.plt.subplots(nairp//2, 2,
                           sharex=True,
                           sharey=True,
                           figsize=(10, 7.5))
for code, ax in zip(CODES, axes.flatten()):
    k = list(map(lambda x: x/255, COLORS[code]))
    yerr = [correlations_psra[code]-correlations_psra[code+'lowerCI'],
            correlations_psra[code+'upperCI']-correlations_psra[code]]
    # correlations_psra[code].plot(ax=ax,
    #                              color=k,
    #                              linestyle='None',
    #                              marker='o',
    #                              markersize=3,
    #                              markerfacecolor='None',
    #                              markeredgewidth=1,
    #                              markeredgecolor=k,
    #                              yerr=yerr)
    ax.errorbar(range(24*60//freq),
                correlations_psra[code],
                yerr=yerr,
                color=k,
                fmt='o',
                elinewidth=0.5,
                markersize=1.5)
    ax.axhline(color=k, lw=0.5)
    ax.set_xlim(-1, 24*60//freq+1)
    ax.set_xticks(xticks)
    ax.set_xticklabels([times[i] for i in xticks])
    ax.set_title('{:s}: correlations from PSRA'.format(code))
ax.set_ylim(-1, 1)
# f.suptitle('Correlation of the arrivals in two consecutive intervals ' +\
#   'of {:d} mins'.format(INTERVAL))
f.savefig(PRFX + 'correlations_psra.png', dpi=300, bbox_inches='tight')
