import atddm
import numpy as np
import pandas as pd
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.model_selection import cross_validate, TimeSeriesSplit
from constants import CODES, TZONES, INTERVAL

nairp = len(CODES)
CODES.sort()
# sort airport ICAO codes alphabetically
FEATURES = ['national', 'continental', 'intercontinental', 'time_cos',
            'time_sin', 'weekday_cos', 'weekday_sin']
# features create by load


def to_seconds(x):
    return x.hour*3600 + x.minute*60 + x.second


def load(**kwargs):
    """
    Load data and compute some features

    **kwargs can be used to pass arguments to atddm.load
    """
    dd = atddm.load(**kwargs)
    for df in dd.values():
        df['national'] = (df['START'].apply(lambda x: x[:2]) ==
                          df['END'].apply(lambda x: x[:2])).astype(int)
        df['continental'] = df['START'].apply(lambda x: x[0]).isin(['E', 'L'])\
            * abs(1-df['national'])
        df['intercontinental'] = 1 - df['national'] - df['continental']
    onedayinsecs = 24 * 3600
    for df in dd.values():
        # df['day_part'] = df['M1_FL240'].dt.time.apply(categorize_time)
        m1_time = to_seconds(df['M1_FL240'].dt)
        df['time_cos'] = np.cos(m1_time*2*np.pi/onedayinsecs)
        df['time_sin'] = np.sin(m1_time*2*np.pi/onedayinsecs)
        weekday = df['M1_FL240'].dt.weekday
        df['weekday_cos'] = np.cos(weekday*2*np.pi/7)
        df['weekday_sin'] = np.sin(weekday*2*np.pi/7)
        df['week'] = df['M1_FL240'].dt.week
        df['dayno'] = df['M1_FL240'].dt.dayofyear
        df['delay_sec'] = df['delay']/pd.Timedelta(1, unit='s')
    return dd


def predict_demand(regr, X, y):
    """
    Predict tM3 according to the model t^M3 = tM1 + delta
    and aggregate the predicted tM3 to obtain the predicted demand

    regr is a regression model that can predict delays delta = tM3 - tM1
    X is a matrix of features with X[:, 0] being tM1
    y is a vector with tM3
    """
    params = regr.get_params()
    interval = params['interval']
    tz = params['tz']
    tm3 = pd.to_datetime(y)
    # transform y in data format
    y_true = atddm.binarrivals(tm3, interval=interval, tz=tz).fillna(0)
    # bin observed tM3 to obtain observed demand
    y_pred = atddm.binarrivals(regr.predict(X), interval=interval,
                               tz=tz).fillna(0)
    # bin predicted tM3 to obtain predicted demand
    combined_indx = y_true.index.union(y_pred.index)
    y_true = y_true.reindex(index=combined_indx).fillna(0)
    y_pred = y_pred.reindex(index=combined_indx).fillna(0)
    # reindex predicted demand as observed one to avoid length mismatch
    return y_true, y_pred


def demand_r2_score(regr, X, y, sample_weight=None):
    """
    Return r2 score of the predicted demand

    X is a matrix of features with X[:, 0] being the M1 time
    y is the observed M3 time
    """
    y_true, y_pred = predict_demand(regr, X, y)
    return r2_score(y_true, y_pred, sample_weight)


def demand_mse_score(regr, X, y, sample_weight=None):
    """
    Return mean squared error between observed and predicted demand

    X is a matrix of features with X[:, 0] being the M1 time
    y is the observed M3 time
    """
    y_true, y_pred = predict_demand(regr, X, y)
    return mean_squared_error(y_true, y_pred, sample_weight)


def demand_mae_score(regr, X, y, sample_weight=None):
    """
    Return median absolute error between observed and predicted demand

    X is a matrix of features with X[:, 0] being the M1 time
    y is the observed M3 time
    """
    y_true, y_pred = predict_demand(regr, X, y)
    return mean_absolute_error(y_true, y_pred, sample_weight)


SCORING = dict(zip(['r2', 'mse', 'mae'],
                   [demand_r2_score, demand_mse_score, demand_mae_score]))


class psraSVR(SVR):
    """
    Custom support vector machine to predict delays between arrivals according
    to regulated (M1) and current (M3) flight plans
    It exposes customized fit and predict methods to deal with time data
    """
    def __init__(self, kernel='rbf', degree=3, gamma='auto', coef0=0.0,
                 tol=0.001, C=1.0, epsilon=0.1, shrinking=True, cache_size=200,
                 verbose=False, max_iter=-1, interval=INTERVAL, tz='UTC'):
        """
        Constructor of the class

        interval is the binning interval
        tz is the time zone of the arrival airport for which the demand is to
           be predicted
        """
        super().__init__()
        self.interval = interval
        self.tz = tz

    def fit(self, X, y=None, sample_weight=None):
        """
        fit model

        X    is an array of features with the understanding that the first
             column X[:, 0] is the M1 arrival time
        y    is the M3 arrival time time
        """
        # transform y and X[:,0] via pd.to_datetime
        tm3 = pd.to_datetime(y)
        tm1 = pd.to_datetime(X[:, 0])
        target = np.array((tm3 - tm1)/pd.Timedelta(1, unit='s'))
        features = X[:, 1:]
        return super().fit(features, target, sample_weight)

    def predict(self, X):
        """
        return predicted M3 time according to formula M1 + predicted_delay

        X    is an array of features with the understanding that the first
             column X[:, 0] is the M1 arrival time
        y    is the M3 arrival time time
        """
        # transform X[:,0] via pd.to_datetime then transform the prediction in
        # TimeDelta before adding it
        tm1 = pd.to_datetime(X[:, 0])
        features = X[:, 1:]
        y_pred = super().predict(features)
        tm3_hat = tm1 + pd.to_timedelta(y_pred, unit='s')
        return tm3_hat


def test():
    print('Cross-validation test routine')
    dd = load()
    code = 'LIRF'
    features = ['national', 'continental', 'intercontinental', 'time_cos',
                'time_sin', 'weekday_cos', 'weekday_sin']

    df = dd[code].sort_values(by='M1_FL240')
    y = np.array(df['M3_FL240'])
    cols = ['M1_FL240'] + features
    X = df.as_matrix(columns=cols)

    psvr = psraSVR(INTERVAL, TZONES[code])
    tss = TimeSeriesSplit(n_splits=3)
    scores = cross_validate(psvr, X, y, scoring=SCORING, cv=tss, n_jobs=3,
                            return_train_score=False)
    print('Results of cross-validation')
    print(scores)


if __name__ == '__main__':
    test()
