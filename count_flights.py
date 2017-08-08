#!/usr/bin/env python3

import atddm
import pandas as pd
from constants import AIRPORTS, CODES

dd = atddm.load(subset=CODES)

indx = [AIRPORTS[c] for c in CODES]
indx = pd.Index(indx, name='airport name')
df = pd.DataFrame(index=indx, data={
    '\\acs{ICAO} code': list(map(lambda s: '\\airp{' + s.lower() + '}',
                             CODES)), 'sample size': [0]*len(CODES)
    })

BEGDT = pd.Timestamp(atddm.constants.BEGDT)
ENDDT = pd.Timestamp(atddm.constants.ENDDT)

for code in CODES:
    name = AIRPORTS[code]
    tmp = (dd[code]['M3_FL240'] >= BEGDT) & (dd[code]['M3_FL240'] <= ENDDT)
    df.loc[name, 'sample size'] = sum(tmp)
    # tmp = dd[code]['M3_FL240'].sort_values()
    # indx = pd.DatetimeIndex(tmp[1:])
    # df.loc[name, 'sample size'] = len(indx[indx.slice_indexer(BEGDT,ENDDT)])

ofile = open('./tables/count_table.tex', 'w')
ofile.write(df.to_latex(column_format='lcr', escape=False))
ofile.close()
