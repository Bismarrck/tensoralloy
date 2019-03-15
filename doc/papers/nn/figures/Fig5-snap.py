from __future__ import print_function

import numpy as np
import re

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

def read_traj(filename):
    """ 
    Read the trajectory and return the MAE/time(GPU) curve.
    """
    speed = 1600
    batch_size = 50
    factor = 1.0 / (speed / batch_size * 3600)
    traj_patt = re.compile(
        r"Step\s=\s+(\d+)\sy_mae/atom\s=\s+([0-9.]+)\sf_mae\s=\s([0-9.]+)")
    with open(filename) as fp:
        data = []
        for line in fp:
            line = line.strip()
            m = traj_patt.search(line)
            if m:
                step = float(m.group(1))
                t = step * factor
                y_mae = float(m.group(2)) * 1000.0
                f_mae = float(m.group(3))
                data.append([t, y_mae, f_mae])
        data = np.asarray(data).transpose()
        return data

gpu_hours, y_list, f_list = read_traj("fig5-traj/g2.50.gpu.traj")

fig, ax1 = plt.subplots(1, 1, figsize=[6, 4])

ax1.plot(gpu_hours, y_list, "r-")

ax1.set_xlabel("GPU Hour", fontsize=16)
ax1.set_ylabel("Energy MAE (meV/atom)", fontsize=14, color='r')
ax1.set_ylim([0, 80])
ax1.yaxis.set_minor_locator(MultipleLocator(20))

xmin, xmax = ax1.get_xlim()[:]
ax1.axhline(22, linestyle='--', color='k', linewidth=1.25,
            label="SNAP/Ni-Mo")
ax1.legend(frameon=False)

ax2 = ax1.twinx()

ax2.plot(gpu_hours, f_list, "b-")
ax2.set_ylabel(r"Force MAE (eV/$\AA$)", fontsize=14, color='b')
ax2.set_ylim([0.14, 0.38])
ax2.yaxis.set_major_locator(MultipleLocator(0.05))
ax2.yaxis.set_minor_locator(MultipleLocator(0.025))

plt.tight_layout()
plt.savefig("Fig5-snap.pdf")
plt.show()
