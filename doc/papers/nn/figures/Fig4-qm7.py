from __future__ import print_function

import numpy as np
import re
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

examples_per_second = {
    'GPU': {
        'Angular': {
            1: 120,
            10: 780,
            20: 1600,
            50: 2400,
            100: 2800,
        },
        'Radial': {
            1: 90,
            10: 1000,
            20: 2400,
            50: 6000,
            100: 10800,
        }
    },
    'CPU': {
        'Angular': {
            1: 270,
            10: 690,
            20: 750,
            50: 610,
            100: 660,
        },
        'Radial': {
            1: 400,
            10: 2800,
            20: 3800,
            50: 5100,
            100: 5800,
        }
    }
}

orders = [
    ('CPU', 'Radial'), ('CPU', 'Angular'), 
    ('GPU', 'Radial'), ('GPU', 'Angular'),
]

traj_files = [
    (1, 'Radial', 'Radial / Batch Size 1', 'fig4-traj/g2.1.gpu.traj'),
    (50, 'Radial', 'Radial / Batch Size 50', 'fig4-traj/g2.50.gpu.traj'),
    (100, 'Radial', 'Radial / Batch Size 100', 'fig4-traj/g2.100.gpu.traj'),
    (1, 'Angular', 'Angular / Batch Size 1', 'fig4-traj/g4.1.gpu.traj'),
    (50, 'Angular', 'Angular / Batch Size 50', 'fig4-traj/g4.50.gpu.traj'),
    (100, 'Angular', 'Angular / Batch Size 100', 'fig4-traj/g4.100.gpu.traj'),
]

traj_patt = re.compile(r"Step\s=\s+(\d+)\sy_mae/atom\s=\s+([0-9.]+)")


def read_traj(filename, batch_size: int, sf: str):
    """ 
    Read the trajectory and return the MAE/time(GPU) curve.
    """
    speed = examples_per_second['GPU'][sf][batch_size]
    factor = 1.0 / (speed / batch_size * 3600)
    with open(filename) as fp:
        data = []
        tail = []
        for line in fp:
            line = line.strip()
            m = traj_patt.search(line)
            if m:
                step = float(m.group(1))
                x = step * factor
                y = float(m.group(2)) * 1000.0
                data.append([x, y])
        data = np.asarray(data)
        return data


speed_flatten = []
for (device, sf) in orders:
    speed_flatten.append([
        examples_per_second[device][sf][batch_size] 
        for batch_size in (1, 10, 20, 50, 100)])

traj = []
for (batch_size, sf, tag, filename) in traj_files:
    traj.append((tag, read_traj(filename, batch_size, sf)))


fig = plt.figure(figsize=[12, 10])
grid_spec = GridSpec(2, 24)

ax = plt.subplot(grid_spec[0, 0: 11])

n_groups = 5
bar_width = 0.2
index = np.arange(n_groups) * 1.5

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
ax.set_xticklabels(('1', '10', '20', '50', '100'))
ax.set_xlabel(r"Batch Size", fontsize=16)
ax.set_ylabel(r"Structures per Second", fontsize=16)
ax.legend(frameon=False, fontsize=12)
ax.text(-0.1, 1.05, "a)", transform=ax.transAxes, size=16, weight='bold')

ax = plt.subplot(grid_spec[0, 13: 24])

ax.plot(traj[0][1][:, 0], traj[0][1][:, 1], 'r-', label=traj_files[0][2])
ax.plot(traj[3][1][:, 0], traj[3][1][:, 1], 'r--', label=traj_files[3][2])
ax.plot(traj[1][1][:, 0], traj[1][1][:, 1], 'g-', label=traj_files[1][2])
ax.plot(traj[4][1][:, 0], traj[4][1][:, 1], 'g--', label=traj_files[4][2])
ax.plot(traj[2][1][:, 0], traj[2][1][:, 1], 'b-', label=traj_files[2][2])
ax.plot(traj[5][1][:, 0], traj[5][1][:, 1], 'b--', label=traj_files[5][2])
ax.set_xlabel(r"GPU Hour", fontsize=16)
ax.set_ylabel(r"MAE (meV/atom)", fontsize=16)
ax.set_xlim([0.0, 1.2])
ax.set_ylim([0, 50])
ax.legend(frameon=False, fontsize=12)
ax.text(-0.1, 1.05, "b)", transform=ax.transAxes, size=16, weight='bold')


ax = plt.subplot(grid_spec[1, 7: 17])

ax.plot(traj[5][1][:, 0], traj[5][1][:, 1], 'b--',
        label="AtomicResNN")

ann_traj = [
    (read_traj("fig4-traj/g4.100.gpu.ann.traj", 100, "Angular"), 
     "AtomicNN")
]
ax.plot(ann_traj[0][0][:, 0], ann_traj[0][0][:, 1], '--', color="orange", 
        label=ann_traj[0][1])

ax.set_xlabel(r"GPU Hour (Angular, Batch Size 100)", fontsize=16)
ax.set_ylabel(r"MAE (meV/atom)", fontsize=16)
ax.set_xlim([0.0, 1.2])
ax.set_ylim([0, 100])
ax.text(-0.2, 0.95, "c)", transform=ax.transAxes, size=16, weight='bold')
ax.legend(frameon=False, fontsize=12, loc='upper center')

grid_spec.tight_layout(fig, pad=0)

plt.savefig("Fig4-qm7.pdf")
plt.show()
