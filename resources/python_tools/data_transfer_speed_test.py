
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import cm

bytes = np.array([5000, 10000, 25000, 50000, 100000, 250000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000, 500000000, 1000000000, ])

pageable_times = np.array([823.915710, 467.116516, 272.008392, 206.431534, 176.525452, 154.102173, 146.772583, 144.678619, 146.861282, 169.380722, 185.248611, 182.535004, 189.000336, 191.082413, ])
pinned_times = np.array([955.946777, 547.597717, 241.850601, 160.495712, 120.001266, 94.218506, 85.848717, 81.935150, 78.687759, 78.222366, 77.494759, 77.548729, 77.619942, 77.320145, ])


offset = 1
bytes = bytes[offset:]
pageable_times = pageable_times[offset:]
pinned_times = pinned_times[offset:]

mpl.rcParams['font.size'] = 16
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

fig = plt.figure()

ax = fig.add_subplot(1, 1, 1)
# ax.set_yscale('log')
ax.set_xscale('log')

def byte2frame(x):
    return x / (512 * 512)


def frame2byte(x):
    return x * (512 * 512)


frame_ax = ax.secondary_xaxis('top', functions=(byte2frame, frame2byte))
frame_ax.set_xlabel(r"Batch size [512 $\times$ 512 Frames]")
ax.set_xlabel("Batch size [Bytes]")

ax.set_ylabel("Host-to-device bandwidth $[GBs^{-1}]$")

ax.plot(bytes, np.reciprocal(pinned_times)*1000,  label="Pinned Data Transfer",  ls="--", marker="o")
# ax.plot(bytes, np.reciprocal(pageable_times), label="Pageable Data Transfer", ls="--", marker="o")
# ax.legend(loc="upper left", frameon=False)
plt.show()