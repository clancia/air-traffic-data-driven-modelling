#!/usr/bin/env python3

import atddm
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import pytz
# from datetime import time
from constants import COLORS, TZONES, CODES, BEGDT, ENDDT

import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import pandas2ri

r = robjects.r
TRUE = robjects.BoolVector([True])
FALSE = robjects.BoolVector([False])
pandas2ri.activate()
dgof = importr('dgof')
dweib = importr('DiscreteWeibull')


def format_time_interval(t1, t2):
    return '{h1:02d}:{m1:02d}--{h2:02d}:{m2:02d}'.format(h1=t1.hour,
                                                         m1=t1.minute,
                                                         h2=t2.hour,
                                                         m2=t2.minute)


def formatter_float_n_digits(x, n):
    return '{x:.{n}f}'.format(x=x, n=n)


def ff3(x):
    return formatter_float_n_digits(x, 3)


def ifelse_formatter(x):
    return formatter_float_n_digits(x, 2) if x >= 0.01 else '<0.01'


sns.set(style="whitegrid", context='paper')

BEGDT = pd.Timestamp(BEGDT)
ENDDT = pd.Timestamp(ENDDT)
INTERVAL = 10
NREPS = 300
dd = atddm.load(subset=CODES)
interarrivals = {}

# TIMES_LOC = [pd.Timestamp('07:00:00'),
#              pd.Timestamp('10:00:00'),
#              pd.Timestamp('13:00:00'),
#              pd.Timestamp('16:00:00')]
# BEGTM_LOC = TIMES_LOC[:-1]
# ENDTM_LOC = TIMES_LOC[1:]

BEGTM_LOC = [pd.Timestamp('08:00:00'),
             pd.Timestamp('12:00:00'),
             pd.Timestamp('18:00:00')]
ENDTM_LOC = [pd.Timestamp('09:30:00'),
             pd.Timestamp('13:30:00'),
             pd.Timestamp('19:30:00')]

PRFX = './../plots/'

for code in CODES:
    tmp = dd[code]['M3_FL240'].sort_values()
    indx = pd.DatetimeIndex(tmp[1:])
    tz = pytz.timezone(TZONES[code])
    indx = indx.tz_localize(pytz.utc).tz_convert(tz)
    interarrivals[code] = pd.DataFrame(data=(tmp.diff().dropna() /
                                             pd.Timedelta(1, 's')).tolist(),
                                       index=indx,
                                       columns=[code])
    interarrivals[code].index = interarrivals[code].index.tz_localize(None)

tmp = list(map(format_time_interval, BEGTM_LOC, ENDTM_LOC))
tmp = list(zip(sorted(tmp*len(CODES)), list(CODES)*len(BEGTM_LOC)))
indx = pd.MultiIndex.from_tuples(tmp, names=['time', 'airport'])
fitted_values = pd.DataFrame(index=indx,
                             columns=['q', 'beta', 'D', 'p', 'mean'])
fitted_values = fitted_values.sort_index()
for t1, t2 in zip(BEGTM_LOC, ENDTM_LOC):
    timeinterval = format_time_interval(t1, t2)
    ia = {}
    f1, axes1 = plt.subplots(2, 4,
                             sharex=False,
                             sharey=False,
                             figsize=(22, 10))
    f2, axes2 = plt.subplots(2, 4,
                             sharex=False,
                             sharey=False,
                             figsize=(16, 9))
    for code, ax1, ax2 in zip(CODES, axes1.flatten(), axes2.flatten()):
        k = list(map(lambda x: x/255, COLORS[code]))
        tmp = interarrivals[code].ix[BEGDT:ENDDT]
        tmp = tmp.between_time(t1.time(), t2.time())
        ia[code] = np.array(tmp).flatten()
        yy = ia[code]
        # fit = exponweib.fit(yy,fa=1)
        # fitted_values.ix[(timeinterval, code),:4] = np.array(fit).round(4)
        # fitted_values.ix[(timeinterval, code),4:] = kstest(yy,
        #                                                    'exponweib',
        #                                                    args=fit)
        fit = dweib.estdweibull(yy, 'ML', zero=TRUE)
        fitted_values.ix[(timeinterval, code), :2] = np.array(fit).round(4)
        qqx = r.seq(0, max(yy))
        ffx = r.stepfun(qqx, r.c(dweib.pdweibull(qqx,
                                                 fit[0],
                                                 fit[1],
                                                 zero=TRUE), 1))
        ks = dgof.ks_test(yy, ffx)
        tmp = [ks.rx2('statistic'), ks.rx2('p.value')]
        fitted_values.ix[(timeinterval, code), 2:4] = np.array(tmp).flatten()
        xx = r.seq(0, 2500)
        probs = dweib.ddweibull(xx, fit[0], fit[1], zero=TRUE)
        fitted_values.ix[(timeinterval, code), 4] = sum(np.array(xx) *
                                                        np.array(probs))
        # n, bins, patches = ax2.hist(yy,
        #                             50,
        #                             normed=1,
        #                             facecolor=k,
        #                             alpha=0.75)
        n, bins, patches = ax2.hist(yy,
                                    bins=range(int(min(yy)),
                                               int(max(yy)+1),
                                               30),
                                    normed=1,
                                    facecolor=k,
                                    alpha=0.5)
        qq = np.arange(len(yy))/(len(yy)+1.)
        yy = np.percentile(yy, 100*qq)
        # xx = exponweib.ppf(qq, *fit)
        xx = np.array(dweib.qdweibull(qq, fit[0], fit[1], zero=TRUE))
        ax1.scatter(xx, yy, color=k, s=5, alpha=0.5)
        ax1.plot(xx, xx, c='k', lw=.75, ls='--')
        ax1.axis('square')
        ax1.set_xlabel('Theoretical quantiles')
        ax1.set_ylabel('Empirical quantiles')
        ax1.set_title('{:s}: q = {:.3f}, beta = {:.3f}'.format(code,
                                                               fit[0],
                                                               fit[1]))

        # add a 'best fit' line for the normal PDF
        # bincenters = 0.5*(bins[1:]+bins[:-1])
        # y = exponweib.pdf(bincenters, *fit)
        # l = ax2.plot(bincenters, y, 'k--', linewidth=1)
        y = np.array(dweib.ddweibull(bins, fit[0], fit[1], zero=TRUE))
        l = ax2.plot(bins, y, 'k--', linewidth=1)
        ax2.set_xlabel('Time (sec)')
        ax2.set_ylabel('Probability')
        ax2.set_title('{:s}: q = {:.3f}, beta = {:.3f}'.format(code,
                                                               fit[0],
                                                               fit[1]))
    plt.subplots_adjust(top=0.9)
    tmp = r'Fitted Weibull: $P_W(X=x;q,\beta) = q^{x^\beta} - q^{(x+1)^\beta}$'
    f1.suptitle(tmp, fontsize=18)
    tmp = r'Fitted Weibull: $P_W(X=x;q,\beta) = q^{x^\beta} - q^{(x+1)^\beta}$'
    f2.suptitle(tmp, fontsize=18)
    tmp = '{:s}IA_qqplot{:02d}{:02d}-{:02d}{:02d}.png'.format(PRFX,
                                                              t1.hour,
                                                              t1.minute,
                                                              t2.hour,
                                                              t2.minute)
    f1.savefig(tmp, dpi=300, bbox_inches='tight')
    tmp = '{:s}IA_hist{:02d}{:02d}-{:02d}{:02d}.png'.format(PRFX,
                                                            t1.hour,
                                                            t1.minute,
                                                            t2.hour,
                                                            t2.minute)
    f2.savefig(tmp, dpi=300, bbox_inches='tight')

ofile = open('./tables/fit_table.tex', 'w')
fitted_values['pp'] = fitted_values['p'].apply(ifelse_formatter)
newcols = ['$q$', '$\\beta$', 'mean', '$D$-stat.', '$p$-value']
rename_dict = dict(zip(['q', 'beta', 'mean', 'D', 'pp'], newcols))
df = fitted_values.rename(columns=rename_dict)
df = df.reset_index()
df['airport'] = df['airport'].apply(lambda s: '\\airp{' + s.lower() + '}')
df = df.set_index(['time', 'airport'])
ofile.write(df.to_latex(column_format='llrrrrr',
                        escape=False,
                        columns=newcols,
                        float_format=ff3))
ofile.close()
plt.close('all')
