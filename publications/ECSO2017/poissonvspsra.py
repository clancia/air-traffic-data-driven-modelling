import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(context='talk')
palette = sns.color_palette()

beta = 1
landa = 1./beta
reps = 25

pois = np.random.exponential(beta, reps)
pois = pois.cumsum()

psra = np.arange(reps)*beta + np.random.exponential(beta, reps) - landa
psra.sort()

f, ax = plt.subplots(1, 2, sharex=True, figsize=(24, 10))

yy = np.arange(reps) + 1

for x, y in zip(pois, yy):
    ax[0].plot([x, x], [0, y], c=palette[0], ls='--', lw=2)
ax[0].step(pois, yy, lw=5)
ax[0].scatter(pois, np.zeros(reps))
ax[0].set_title(r'Poisson arrivals, $\lambda$ = {:.1f}'.format(landa))
ax[0].set_xlabel('time')
ax[0].set_ylabel('count')

for x, y in zip(psra, yy):
    ax[1].plot([x, x], [0, y], c=palette[0], ls='--', lw=2)
ax[1].step(psra, yy, lw=5)
ax[1].scatter(psra, np.zeros(reps))
title = r'Pre-scheduled random arrivals, $\sigma$ = {:.1f}'.format(landa)
ax[1].set_title(title)
ax[1].set_xlabel('time')

plt.savefig('pois_psra.png')
