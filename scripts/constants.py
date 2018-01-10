import os
import pandas as pd

pathtohere = os.path.abspath(__file__)
HERE = os.path.dirname(pathtohere)

PREFIX = HERE + '/../DataByEndAirport/'
POSTFIX = '.csv'

CODES = ['EDDF', 'EGKK', 'EGLL', 'EHAM', 'LEMD', 'LFPG', 'LGAV', 'LIRF']

AIRPORTS = {
    'EDDF': 'Frankfurt am Main International Airport',
    'EGKK': 'London Gatwick Airport',
    'EGLL': 'London Heathrow Airport',
    'EGSS': 'London Stansted Airport',
    'EGGW': 'London Luton Airport',
    'EHAM': 'Amsterdam Airport Schiphol',
    'EIDW': 'Dublin Airport',
    'LEMD': 'Madrid Barajas International Airport',
    'LFPG': 'Charles de Gaulle International Airport',
    'LGAV': 'Athens International Airport',
    'LIRF': 'Rome Fiumicino International Airport'
}

TIMEZONES = {
    'EDDF': pd.Timedelta(2, 'h'),
    'EGKK': pd.Timedelta(1, 'h'),
    'EGLL': pd.Timedelta(1, 'h'),
    'EGSS': pd.Timedelta(1, 'h'),
    'EGGW': pd.Timedelta(1, 'h'),
    'EHAM': pd.Timedelta(2, 'h'),
    # 'EIDW': pd.Timedelta(1, 'h'),
    'LEMD': pd.Timedelta(2, 'h'),
    'LFPG': pd.Timedelta(2, 'h'),
    'LGAV': pd.Timedelta(3, 'h'),
    'LIRF': pd.Timedelta(2, 'h')
}

TZONES = {
    'EDDF': 'Europe/Berlin',
    'EGKK': 'Europe/London',
    'EGLL': 'Europe/London',
    'EGSS': 'Europe/London',
    'EGGW': 'Europe/London',
    'EHAM': 'Europe/Amsterdam',
    # 'EIDW': 'Europe/Dublin',
    'LEMD': 'Europe/Madrid',
    'LFPG': 'Europe/Paris',
    'LGAV': 'Europe/Athens',
    'LIRF': 'Europe/Rome'
}

# http://www.somersault1824.com/tips-for-designing-scientific-figures-for-color-blind-readers/
COLORS = {
    'EDDF': [73, 0, 146],
    'EGKK': [0, 73, 73],
    'EGLL': [109, 182, 255],
    'EGSS': [219, 209, 0],
    'EGGW': [182, 219, 255],
    'EHAM': [255, 182, 119],
    'LEMD': [146, 0, 0],
    'LFPG': [146, 73, 0],
    'LGAV': [0, 109, 219],
    'LIRF': [36, 255, 36],
}

ALLICAO = AIRPORTS.keys()

BEGDT = '2016-06-15 00:00:00'
ENDDT = '2016-09-14 23:59:59'

CLMNS = ['START', 'END', 'ID', 'M1_FL240', 'M3_FL240']
TIMERESAMPLE = 5
