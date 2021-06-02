from functools import reduce
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt 
from scipy import fftpack
import argparse 
import string


def read_file(file_name):
    """
    Takes in a file path and reads / returns the corresponding I(lambda, tau) vector,
    Must be formatted as follows:

    file-start
    [0] <vector of lambda values>
    [1] <vector of t values>
    [2] <I(lambda, t) for 1st lambda value>
    [3] <I(lambda, t) for 2nd lambda value>
    ...
    [.] <I(lambda, t) for last lambda value>
    file-end
    """

    try:
        with open(file_name, "r") as f:
            # First 2 
            lambda_vector = [float(n) for n in f.readline().split()]
            tau_vector = [float(n) for n in f.readline().split()]
            
            lambda_count = len(lambda_vector)

            ISF = []
            for _ in range(lambda_count):
                ISF.append(np.array([float(n) for n in f.readline().split()]))

    except FileNotFoundError:
        print(f"Error reading {file_name}, check file exsists / formatting")
        lambda_vector, tau_vector, ISF = [], [], [[]]

    return (np.array(lambda_vector), np.array(tau_vector), np.array(ISF))

def lorentzian(f, f0, a, gam):
    return a * gam**2 / ( gam**2 + (f - f0 )**2)

def ft_lorentzian(x, f0, a, gam):
    return a * abs(gam) * np.exp(-x * abs(gam)) * np.cos(-x * f0)


CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'


def find_freq(ISF, tau):
    maxes = []
    for i in range(1, len(ISF)-1):
        if ISF[i-1] < ISF[i] > ISF[i+1]:
            maxes.append(tau[i])

    Ts = []
    for x in range(len(maxes) - 1):
        Ts.append(maxes[x+1] - maxes[x])

    print(maxes)
    print(Ts)
    print(np.average(Ts))
    



if __name__ == "__main__":
    plt.rcParams['font.size'] = 16
    plt.rcParams['mathtext.fontset'] = 'stix'
    plt.rcParams['font.family'] = 'STIXGeneral'

    parser = argparse.ArgumentParser(description="Plot multi-DDM ISF for Brownian motion.")

    parser.add_argument("--root",  metavar="FILES", nargs=1,    help="Root file-path for the ISF input files.")
    parser.add_argument("--tiles", metavar="T",     nargs="+",  help="Number of tiles per full frame.")
    parser.add_argument("--scales",metavar="N",     nargs="+",  help="Length scale for each input.")

    args = parser.parse_args()

    file_root = args.root[0]
    tiles = args.tiles
    scales = args.scales

    for scale_idx, scale in enumerate(scales):

        print(f"[{scale} x {scale}]")

        tile_count = int(tiles[scale_idx])
        side_len = int(np.ceil(np.sqrt(tile_count)))

        data = [read_file(file_root + str(scale) +  "-" + str(x)) for x in range(tile_count)]

        for tile_idx, (lamda_vector, tau_vector, full_ISF) in enumerate(data):
            capture_rate = 7
            
            ISF = np.average(full_ISF, axis=0)
            ISF -= np.average(ISF)
            ISF = ISF / max(ISF)

            ft_ISF = fftpack.fft(ISF)
            abs_ft_ISF = np.abs(ft_ISF)
            freqs = fftpack.fftfreq(len(ISF)) * capture_rate

            if tile_idx < 2:
                off = 0
            else:
                off = 2

            real_ax = plt.subplot(side_len, 2*side_len, (tile_idx+1)+off)
            ft_ax = plt.subplot(side_len, 2*side_len, (tile_idx+1)+off+2)

            real_ax.plot(tau_vector, ISF, color="black", alpha=0.6)
            real_ax.scatter(tau_vector, ISF, color="black", marker="+")


            offset = 10
            reached_peak = False
            reduced_freqs = [f for f in freqs[:offset]]
            reduced_ISF = [f for f in abs_ft_ISF[:offset]]

            for i in range(offset+1, len(abs_ft_ISF)-1):
                if not reached_peak:
                    if abs_ft_ISF[i+1] < abs_ft_ISF[i]:
                        reached_peak = True
                else:
                    if abs_ft_ISF[i+1] > abs_ft_ISF[i]:
                        # reached first min
                        break
                reduced_freqs.append(freqs[i])
                reduced_ISF.append(abs_ft_ISF[i])

            popt, pcov = opt.curve_fit(lorentzian, reduced_freqs, reduced_ISF)
            perr = np.sqrt(np.diag(pcov)) # find 1 standard deviation from covariances

            BPM = (1.0 / popt[0]) * 60
            err_BMP = (perr[0] / popt[0]) * BPM

            fit_tau = np.linspace(min(freqs), max(freqs), num=750)
            fit_lor = [lorentzian(t, *popt) for t in fit_tau]

            print(f"{scale}:{tile_idx}, freq = {popt[0]} +- {perr[0]}, BPM = {BPM} +- {err_BMP}")

            freqs, abs_ft_ISF = zip(*sorted(zip(freqs, abs_ft_ISF)))

            ft_ax.plot(fit_tau, fit_lor, "--r")
            ft_ax.plot(freqs, abs_ft_ISF, "-k", alpha=0.6)
            ft_ax.set_xlim(0, capture_rate / 2)

            # fit_tau = np.linspace(min(tau_vector), max(tau_vector), num=750)
            # fit_lor = [ft_lorentzian(7*x, *popt) for x in fit_tau]   

            ft_ax.annotate(string.ascii_uppercase[tile_idx],
              xy=(0.9, 0.90), xycoords='axes fraction', fontweight="bold")
            real_ax.annotate(string.ascii_uppercase[tile_idx],
              xy=(0.9, 0.05), xycoords='axes fraction', fontweight="bold")

            ft_ax.set_yticklabels([])
            real_ax.set_yticklabels([])
            real_ax.set(xlabel=r"Lag time, $\tau$ [s]", ylabel=r"I($\tau$) [a. u.]")
            ft_ax.set(xlabel=r"Frequency, $\omega$ [Hz]", ylabel=r"| I($\omega$) | [a. u.]")
            find_freq(ISF, tau_vector)
        plt.show()