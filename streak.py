import pandas as pd
import math
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv("data.csv")
streak = 1
streaks = []
winlose = np.zeros((df.shape[0]))

winlose = df.loc[:,["point_victor"]].values

for t in range(1,df.shape[0]):
    if winlose[t] == winlose[t - 1]:
        streak += 1
    else:
        if streak > 15:
            print(t)
        streaks.append(streak)
        streak = 1


np.random.seed(1)
streak2 = 1
streaks2 = []
rand_set = np.random.randint(0, 2, size=(df.shape[0]))
for i in range(1, df.shape[0]):
    if rand_set[i] == rand_set[i-1]:
        streak2 += 1
    else:
        streaks2.append(streak2)
        streak2 = 1

O = np.bincount(streaks)
T = np.bincount(np.pad(streaks2,(14)))
chi_sq = np.sum(((O - T) ** 2 / T))

print("Data:", np.average(streaks), "Random:", np.average(streaks2), "Chi_sq:", chi_sq)
streaks2 = streaks2[0:len(streaks)]
out = np.column_stack((streaks,streaks2))

plt.hist(out, 10, histtype='bar', density=True, alpha=0.6, color=['g','b'], rwidth = 0.5)
plt.xlabel("streak length")
plt.ylabel("frequency")
plt.legend(["data","random"])
#plt.hist(streaks2, histtype='bar', density=True, bins=10, alpha=0.6, color='b', rwidth = 0.5)