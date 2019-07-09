#!coding=utf-8
from __future__ import print_function

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

orders = [
    ('CPU', 'Radial'), ('CPU', 'Angular'), 
    ('GPU', 'Radial'), ('GPU', 'Angular'),
]

fig = plt.figure(figsize=[12, 10])
grid_spec = GridSpec(2, 24)

# ----------
# Fig.A
# ----------

ax = plt.subplot(grid_spec[0, 0: 11])
assert isinstance(ax, plt.Axes)

df1 = pd.read_csv("qm7/qm7.speed.csv", index_col=0)
speed_flatten = np.zeros((4, 4))
batch_size_idx_map = {1: 0, 25: 1, 50: 2, 100: 3}

for i, (device, sf) in enumerate(orders):
    if device == "CPU":
        use_cpu = True
    else:
        use_cpu = False
    if sf == "Angular":
        angular = True
    else:
        angular = False
    rows = df1.loc[(df1["cpu"] == use_cpu) &
                   (df1["angular"] == angular) &
                   (df1["arch"] == "AtomicResNN")]
    for _, row in rows.iterrows():
        j = batch_size_idx_map[row.batch_size]
        speed_flatten[i, j] = row.examples_per_sec

n_groups = 5
bar_width = 0.2
index = np.arange(4) * 1.5

ax.bar(
    index + bar_width * 0,
    np.array(speed_flatten[0]), 
    bar_width, 
    alpha=0.4, color='r', label=f"{orders[0][0]}/{orders[0][1]}")
ax.bar(
    index + bar_width * 1,
    np.array(speed_flatten[2]),
    bar_width,
    alpha=0.4, color='g', label=f"{orders[2][0]}/{orders[2][1]}")
ax.bar(
    index + bar_width * 2,
    np.array(speed_flatten[1]),
    bar_width,
    alpha=0.4, color='b', label=f"{orders[1][0]}/{orders[1][1]}")
ax.bar(
    index + bar_width * 3,
    np.array(speed_flatten[3]),
    bar_width,
    alpha=0.4, color='k', label=f"{orders[3][0]}/{orders[3][1]}")

ax.set_xticks(index + bar_width * 1.5)
ax.set_xticklabels(('1', '25', '50', '100'))
ax.set_xlabel(r"Batch Size", fontsize=16)
ax.set_ylabel(r"Structures per Second", fontsize=16)
ax.legend(frameon=False, fontsize=12)
ax.text(-0.1, 1.05, "a)", transform=ax.transAxes, size=16, weight='bold')

# ----------
# Fig.B
# ----------

new_traj = []
t_factors = []
labels = []
idx = 0

for batch_size in [1, 50, 100]:
    for angular in [False, True]:
        row = df1[(df1["cpu"] == False) &
                  (df1["angular"] == angular) &
                  (df1["arch"] == "AtomicResNN") &
                  (df1["batch_size"] == batch_size)]
        t_factor = 1.0 / (row.examples_per_sec.values[0] / batch_size * 3600)
        t_factors.append(t_factor)
        if angular:
            tag = "Angular"
            gval = 4
        else:
            tag = "Radial"
            gval = 2
        labels.append(f"{tag} / Batch Size {batch_size}")
        new_traj.append(pd.read_csv(f"qm7/g{gval}.{batch_size}.res.csv"))


ax = plt.subplot(grid_spec[0, 13: 24])
assert isinstance(ax, plt.Axes)

idx = 0
for color in "rgb":
    for style in ["-", "--"]:
        ax.plot(new_traj[idx]['global_step'] * t_factors[idx],
                new_traj[idx]['Energy/mae/atom'] * 1000.0,
                f'{color}{style}',
                label=labels[idx])
        idx += 1

ax.set_xlabel(r"GPU Hour", fontsize=16)
ax.set_ylabel(r"MAE (meV/atom)", fontsize=16)
ax.set_xlim(0.0, 1.01)

ax.legend(frameon=False, fontsize=12)
ax.text(-0.1, 1.05, "b)", transform=ax.transAxes, size=16, weight='bold')

# ----------
# Fig.C
# ----------

new_traj.append(pd.read_csv("qm7/g4.100.ann.csv"))
t_factors.append(t_factors[-1])

ax = plt.subplot(grid_spec[1, 7: 17])
assert isinstance(ax, plt.Axes)

ax.plot(new_traj[5]['global_step'] * t_factors[5],
        new_traj[5]['Energy/mae/atom'] * 1000.0,
        'b--',
        label="AtomicResNN")

ax.plot(new_traj[6]['global_step'] * t_factors[6],
        new_traj[6]['Energy/mae/atom'] * 1000.0,
        '--',
        color='orange',
        label="AtomicNN")

ax.set_xlabel(r"GPU Hour (Angular, Batch Size 100)", fontsize=16)
ax.set_ylabel(r"MAE (meV/atom)", fontsize=16)
ax.set_ylim(0.0, 50.0)
ax.text(-0.2, 0.95, "c)", transform=ax.transAxes, size=16, weight='bold')
ax.legend(frameon=False, fontsize=12, loc='upper center')

grid_spec.tight_layout(fig, pad=0)

plt.savefig("qm7_speed.pdf")
plt.show()
