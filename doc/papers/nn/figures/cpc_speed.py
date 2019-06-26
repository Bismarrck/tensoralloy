from __future__ import print_function, absolute_import

import numpy as np

from matplotlib import pyplot as plt
from matplotlib.ticker import MultipleLocator

__author__ = 'Xin Chen'
__email__ = 'Bismarrck@me.com'


workstation = """
Number of atoms: 2000
* ASE neighbor time: 0.5930616855621338
* Python feed dict time: 1.5961778163909912
* TensorFlow execution time: 0.04986286163330078
Number of atoms: 6750
* ASE neighbor time: 1.2973895072937012
* Python feed dict time: 3.7827587127685547
* TensorFlow execution time: 0.12984418869018555
Number of atoms: 16000
* ASE neighbor time: 3.7363853454589844
* Python feed dict time: 9.718251943588257
* TensorFlow execution time: 0.17889857292175293
Number of atoms: 31250
* ASE neighbor time: 5.931849002838135
* Python feed dict time: 16.419432401657104
* TensorFlow execution time: 0.3449971675872803
Number of atoms: 54000
* ASE neighbor time: 10.83665919303894
* Python feed dict time: 28.86913824081421
* TensorFlow execution time: 0.6041886806488037
Number of atoms: 85750
* ASE neighbor time: 19.667125940322876
* Python feed dict time: 51.148345708847046
* TensorFlow execution time: 0.38422513008117676
Number of atoms: 128000
* ASE neighbor time: 26.636351346969604
* Python feed dict time: 70.64423513412476
* TensorFlow execution time: 1.0043096542358398"""


cpu_only = """Number of atoms: 2000
* ASE neighbor time: 0.623563289642334
* Python feed dict time: 1.633838415145874
* TensorFlow execution time: 0.10250282287597656
Number of atoms: 6750
* ASE neighbor time: 1.3154628276824951
* Python feed dict time: 3.6260945796966553
* TensorFlow execution time: 0.32638072967529297
Number of atoms: 16000
* ASE neighbor time: 3.5952367782592773
* Python feed dict time: 9.219270467758179
* TensorFlow execution time: 0.6607975959777832
Number of atoms: 31250
* ASE neighbor time: 5.899898529052734
* Python feed dict time: 16.461201190948486
* TensorFlow execution time: 1.1427969932556152
Number of atoms: 54000
* ASE neighbor time: 10.898675680160522
* Python feed dict time: 29.321943759918213
* TensorFlow execution time: 1.8838279247283936
Number of atoms: 85750
* ASE neighbor time: 21.225797653198242
* Python feed dict time: 54.72197508811951
* TensorFlow execution time: 2.7469184398651123
Number of atoms: 128000
* ASE neighbor time: 30.465934991836548
* Python feed dict time: 78.67807269096375
* TensorFlow execution time: 4.12495493888855"""


def get_seconds(output: str):
    _t_nl = []
    _t_feed = []
    _t_core = []
    _tick_labels = []

    for line in output.split("\n"):
        line = line.strip()
        if line.startswith("Number"):
            n = int(line.split()[-1])
            nroot = int(np.round(np.power(n // 2, 1.0 / 3.0)))
            _tick_labels.append(r"$({%d})^3$" % nroot)
        elif line.startswith("* ASE"):
            _t_nl.append(float(line.split()[-1]))
        elif line.startswith("* Python"):
            _t_feed.append(float(line.split()[-1]) - _t_nl[-1])
        elif line.startswith("* TensorFlow"):
            _t_core.append(float(line.split()[-1]))
    return _t_nl, _t_feed, _t_core, _tick_labels

width = 0.3

fig, axes = plt.subplots(1, 2, figsize=[10, 5])

ax = axes[0]
assert isinstance(ax, plt.Axes)

t_nl, t_feed, t_core, tick_labels = get_seconds(workstation)

ax.set_yscale('linear')

x = np.arange(len(t_nl))
t_nl = np.array(t_nl)
t_feed = np.array(t_feed)
t_core = np.array(t_core)
ax.bar(x, t_nl, width=width, bottom=0.0, label="Neighbor List")
ax.bar(x, t_feed, width=width, bottom=t_nl, label="VAP")
handle = ax.bar(x, t_core, width=0.3, bottom=t_nl + t_feed, label="Graph")

for i, patch in enumerate(handle.patches):
    bx, by = patch.get_xy()
    ax.text(bx + 0.15, by + 1.0 + t_core[i], "%.2f" % (t_core[i]),
            verticalalignment="center", horizontalalignment="center",
            fontdict={"fontsize": 8.0}, style='italic')

ax.set_xticklabels(x)
ax.set_xticklabels([""] + tick_labels)
ax.set_xlabel(r"$(\mathrm{MoNi})_{x}$", fontsize=15)

ax.set_ylabel(f"Overall w/ GPU (s)", fontsize=15)
ax.yaxis.set_minor_locator(MultipleLocator(2.0))
ax.legend()
ax.text(-0.1, 1.05, "a)", transform=ax.transAxes, size=14, weight='bold')

ax = axes[1]
assert isinstance(ax, plt.Axes)

_, _, c_core, _ = get_seconds(cpu_only)

ax.bar(x - width * 0.5, t_core, width=width, color='green', label="GPU")
ax.bar(x + width * 0.5, c_core, width=width, color='green', alpha=0.2,
       label="CPU")

ax.set_ylabel("Graph Time w/o GPU (s)", fontsize=15)
ax.set_xticklabels(x)
ax.set_xticklabels([""] + tick_labels)
ax.set_xlabel(r"$(\mathrm{MoNi})_{x}$", fontsize=15)

ax.legend()
ax.text(-0.1, 1.05, "b)", transform=ax.transAxes, size=14, weight='bold')

plt.tight_layout()
plt.savefig("Prediction-speed.pdf", dpi=150)
plt.show()
