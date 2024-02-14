import pandas as pd
import time
from datetime import datetime
import math
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data_fill_main.csv")

t0 = 0
t = 0
point_num = df.iloc[t + t0, 5] #前项
mij = [0]
mji = [0]
alpha1 = 10
alpha2 = 1
Kij0 = 1
Kji0 = 1
Pij = []
Pji = []
avg_dist = 90
avg_time = 30
distance1 = 0
distance2 = 0
cache = []
percentage = 0
total = 0
t2 = 0
p = 0
sji_same = 0.8
sji_diff = 1.2

match_num = 0
set_num = 0
game_num = 0
while p < 300:#game
    if point_num != df.iloc[t + t0, 5]:
        p += 1
        t2 = 0
        distance1 = 0
        distance2 = 0
        mij[t] *= 0.5
        mji[t] *= 0.5
    point_num = df.iloc[t + t0, 5]
    distance1 += float(df.iloc[t + t0, 39])#game开始到现在跑动距离
    distance2 += float(df.iloc[t + t0, 40])

    Time = datetime.strptime(df.iloc[t + t0, 3],"%H:%M:%S").timestamp() + 2209017943#game开始到现在秒
    deltaTime = datetime.strptime(df.iloc[t + t0, 3],"%H:%M:%S").timestamp() - (datetime.strptime(df.iloc[t + t0 - 1, 3],"%H:%M:%S").timestamp() if t + t0 != 0 else -2209017943)
    deltaTime = max(0, deltaTime)
    #上一球秒数
    Kij = ((1 + alpha1) * Kij0 * (1 + alpha2)) / ((1 + alpha1 * math.exp(-deltaTime / avg_time)) * (1 + alpha2 * math.exp(-distance1 / avg_dist)))
    Kji = ((1 + alpha1) * Kji0 * (1 + alpha2)) / ((1 + alpha1 * math.exp(-deltaTime / avg_time)) * (1 + alpha2 * math.exp(-distance2 / avg_dist)))
    Wij = (df.iloc[t + t0, 16] - (df.iloc[t + t0 - 1, 16] if t + t0 != 0 else 0))
    Wji = (df.iloc[t + t0, 17] - (df.iloc[t + t0 - 1, 17] if t + t0 != 0 else 0))

    Pij.append(1 / (1 + math.pow(10, (mji[t] - mij[t]) / 20)))
    Pji.append(1 / (1 + math.pow(10, (mij[t] - mji[t]) / 20)))
    
    server = df.iloc[t + t0, 13]
    sij = sji_same if (server == 1) == Wij else sji_diff #发球且得分则小值
    sji = sji_same if (server == 2) == Wji else sji_diff

    mij.append(mij[t] + Kij * sij * (Wij - Pij[t])) #mij[t + 1]
    mji.append(mji[t] + Kji * sji * (Wji - Pji[t]))
    #print((mji[t] - mij[t]) / 400)
    #cache.append(0.5 if Wij == (Pij[t] > 0.5) else 0.4)
    cache.append(1 if Wij else 0)
    if t2 >= 0:
        percentage += (Wij == (Pij[t] > 0.5))
        total += 1
    
    #print(" Kij:", Kij, " Kji:", Kji, " Pij:", Pij[t], " Pji:", Pji[t], " mij:", mij[t], " mji:", mji[t], "\n")
    
    t += 1
    t2 += 1

#print(t, len(Pij))
print(percentage / total)

mij.pop()

ax = plt.axes()
ax.set_xlabel('points')
ax.set_ylabel('Pij')
plt.plot(range(0,t), np.array(Pij))#蓝
plt.plot(cache, 'o')
plt.legend(['Correct predictions'])

"""
ax.set_xlabel('points')
ax.set_ylabel('mij or Pij (scaled)')
plt.plot(range(0,t), (np.array(Pij) * 10 - 5) * 1.8)#蓝
plt.plot(range(0,t), mij)#蓝
plt.legend(["Pij (scaled)", "mij"])
"""
plt.show()





