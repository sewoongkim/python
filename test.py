import random
random.randint(1, 125)


import random
sum(random.choices([0, 1], [0.992, 0.008], k = 125))

import random

import numpy as np
import scipy.stats

K = 1.0
DP = 0.01
DC = 0.999
GR = 100000  # 답을 못 찾으면 이 값을 더 크게

p =  0.000882
# input(f'Probability [{DP:.2f}]: ')
p = float(p) if len(p) > 0 else DP

c = 0.999
# input(f'Confidence [{DC:.3f}]: ')
c = float(c) if len(c) > 0 else DC
n = np.arange(K, GR*K)

dist = scipy.stats.binom.sf(K, n, p)
trial = int(n[dist >= c][0])

print(trial)

import random 
from collections import defaultdict

import pandas as pd

EPOCHS = 5

LEVELS = [0, 1, 2, 3]
WEIGHTS = [0.791492, 0.202117, 0.005509, 0.000882]
# MAXLENS = [7, 42, 1672, 10465]  # 신뢰도 0.999
# MAXLENS = [9, 53, 2129, 13324]  # 신뢰도 0.9999
MAXLENS = [10, 64, 2578, 16135]   # 신뢰도 0.99999
# MAXLENS = [12, 75, 3022, 18914]   # 신뢰도 0.999999
TOTAL = 1655579

results = defaultdict(list)
dfs = []
for e in range(EPOCHS):
    random.seed(e)
    r = random.choices(LEVELS, WEIGHTS, k=TOTAL)
    sr = pd.Series(r)
    # print(sr)
    for l in LEVELS:
        kk = sr[sr == l]
        # print(kk)
        dd = kk.reset_index()['index'].to_frame()
        # print(dd)
        dd = dd.assign(lag=dd['index'].shift(1))
        # print("QQQQQQQQQQQ")
        # print(dd)
        dd = dd.assign(dif=dd['index'] - dd['lag'])
        ml = dd.dif.max()
        # print("MAX")
        # print(ml)
        dif = int(ml - MAXLENS[l])
        rate = dif / MAXLENS[l]
        res = 'PASS' if dif <= 0 else f'FAIL (+{dif}, {rate:.2f})'
        print(f"epoch: {e}, level: {l}, maxlen: {ml}, result: {res}")

