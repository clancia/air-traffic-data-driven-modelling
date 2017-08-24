import glob
import pandas as pd
from pytz import timezone, utc
from scipy.stats import sem

from constants import PREFIX, POSTFIX, CLMNS, TIMERESAMPLE


def load(subset=None, pathtocsv=PREFIX, **kwargs):
    """
    Read data from csv files in pathtocsv
    Datasets must be named as XXXX.csv, where XXXX is the ICAO code of the
    inbound airport
    A list of ICAO codes can be passed to subset to filter the datasets loaded
    Remaining arguments are passed to pd.read_csv
    """
    dataframes = []
    _ = kwargs.setdefault('parse_dates', [3, 4])
    # in the default dataset columns 4 and 5 are dates
    _ = kwargs.setdefault('infer_datetime_format', True)
    if subset is None:
        files = glob.glob(pathtocsv + '[A-Z]'*4 + '.csv')
        # filters all csv files named XXXX.csv
    else:
        files = [pathtocsv+code+POSTFIX for code in subset]
    failed = []
    for f in files:
        try:
            df = pd.read_csv(f, **kwargs)
        except OSError as e:
            print('ERROR :: {}'.format(e))
            print('This file will be skipped\n')
            failed.append(f)
            pass
        else:
            df.columns = CLMNS
            df['delay'] = df.M3_FL240 - df.M1_FL240
            dataframes.append(df)
    notfailed = [code[-8:-4] for code in files if code not in failed]
    if not len(notfailed):
        print('WARNING :: No dataset loaded')
    return dict(zip(notfailed, dataframes))


def binarrivals(ss, interval=TIMERESAMPLE, tz=None):
    ts = pd.Series(index=ss, data=1, name='arrivals')
    # ts = ts.resample(str(interval)+'Min', how='sum')
    ts = ts.resample(str(interval)+'Min').sum()
    ts = ts.sort_index()
    if tz is not None:
        tz = timezone(tz)
        ts.index = ts.index.tz_localize(utc).tz_convert(tz)
    return ts


def daily_avg(ts, tz=None):

    freq = ts.index.freq.delta.components.hours*60 +\
        ts.index.freq.delta.components.minutes
    slices = list(map(pd.Timestamp,
                      ['{:02d}:{:02d}'.format(i, j) for i in range(24)
                       for j in range(0, 60, freq)]))
    if tz is not None:
        slices = [timezone(tz).localize(sl) for sl in slices]
    means = []
    stdvs = []
    # upper = []
    # lower = []
    sems = []
    for i, j in zip(slices, slices[1:]+[slices[0]]):
        ss = ts.between_time(i.time(), j.time()).fillna(value=0)
        means.append(ss.mean())
        stdvs.append(ss.std())
        sems.append(sem(ss))

    daily = list(map(lambda x: x.isoformat(), slices))
    daily = pd.DataFrame(data={'mu': means, 'stermn': sems, 'sigma': stdvs},
                         index=pd.DatetimeIndex(daily))
    daily = daily.asfreq(str(freq)+'Min')
    if tz is not None:
        tz = timezone(tz)
        daily.index = daily.index.tz_localize(utc).tz_convert(tz)
    return daily
