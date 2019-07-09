#!coding-utf-8

from __future__ import print_function

import pandas as pd

from scipy.interpolate import interp1d
from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

df = pd.read_csv("snap.summary.csv")


gpu_hours = df['global_step'] / 600.0 / 60.0
y_list = df["Energy/mae/atom"] * 1000.0
f_list = df["Forces/mae"]


y_spline = interp1d(gpu_hours, y_list)

yx = None
yy = None
step0 = int(min(gpu_hours) / 0.01) + 1
stept = int(max(gpu_hours) / 0.01)

for step in range(step0, stept):
    t = step * 0.01
    if yx is None:
        yt = y_spline(t)
        if yt <= 22.5:
            yx = t
            yy = yt
            break


fig, ax1 = plt.subplots(1, 1, figsize=[6, 4])

ax1.plot(gpu_hours, y_list, "r-")

ax1.set_xlabel("GPU Hour", fontsize=16)
ax1.set_ylabel("Energy MAE (meV/atom)", fontsize=14, color='r')
ax1.yaxis.set_minor_locator(MultipleLocator(1))

assert isinstance(ax1, plt.Axes)
ax1.set_yscale('log')
# ax1.plot((yx, yx), (0.0, yy), linestyle="--", color='r', linewidth=0.8)


ax2 = ax1.twinx()

ax2.plot(gpu_hours, f_list, "b-")
ax2.set_ylabel(r"Force MAE (eV/$\AA$)", fontsize=14, color='b')
ax2.yaxis.set_major_locator(MultipleLocator(0.05))
ax2.yaxis.set_minor_locator(MultipleLocator(0.025))

plt.tight_layout()
plt.savefig("snap_train_curve.pdf")
plt.show()
