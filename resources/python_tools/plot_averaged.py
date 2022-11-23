# Function to plot ISF values and extract multi-scale diffusion coefficent
# Gives example of how to read and analyse ISF

# Example usage:
# python plot_averaged.py --root ./out/out --scales 1024 512 256 --tiles 1 4 16 --umpx 0.09700
# ./out/out - the file path of the ISF files
# 1024, 512, 256 - the pixel scales to be analysed
# 1, 4, 16 - the number of tiles for each scale

import argparse 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import cm
import scipy.optimize as opt 
import time
import string
import matplotlib.ticker as mticker

CB91_Blue = '#2CBDFE'
CB91_Green = '#47DBCD'
CB91_Pink = '#F3A0F2'
CB91_Purple = '#9D2EC5'
CB91_Violet = '#661D98'
CB91_Amber = '#F5B14C'

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


def fit_func(x, F, A, B):
    return A * (1 - np.exp( - x  * F, where=(- x  * F < 700))) + B


# Fits Brownian approximation to the ISF of the form:
# A (1 - exp(- tau * F)) + B 
# A, B are fitting parameters and not of interest, t is the
# characteristic decay time and will be used to find the 
# diffusion coefficient.
def fit_ISF(ISF, q_vector, tau_vector):
    out_params = []
    out_errs   = []

    for q_idx in range(len(q_vector)):

        ISF_q = ISF[q_idx]

        # try:
        popt, pcov = opt.curve_fit(fit_func, tau_vector, ISF_q, p0=[6, 3e-2, 4e-3])
        # except:
            # print(f"Fitting ISF failed for q_idx: {q_idx} [using default values of 1].")
            # popt = [1, 1, 1]
            # pcov = [(1,1,1), (1,1,1), (1,1,1)]

        perr = np.sqrt(np.diag(pcov)) # find 1 standard deviation from covariances

        out_params.append(popt)
        out_errs.append(perr)

    return np.array(out_params), np.array(out_errs)

def plot_diff_coeff(ax, q_vector, params, errs):

    params = np.array(params)
    
    print(params)
    print(errs)

    Fs = params[:, 0]
    Fs_err = errs[:, 0]
    # Fs = params[:]
    # Fs_err = errs[:]

    # F must be greator than zero
    Fs_tmp = []
    q_vector_tmp = []
    err_tmp = []

    for i, F in enumerate(Fs):
        if F >= 0:
            q_vector_tmp.append(q_vector[i])
            err_tmp.append(Fs_err[i])
            Fs_tmp.append(F)
        else:
            print("Unexpected positive F value (removed).")
    
    Fs = Fs_tmp
    Fs_err = err_tmp
    q_vector = q_vector_tmp
    print(q_vector)
    # From theory we have lnF = ln(D q^2) = ln(D) + 2lnq
    grad2line = lambda x, c: 2 * x + c


    lnFs = np.log(Fs)
    lnqs  = np.log(q_vector)

    popt, pcov = opt.curve_fit(grad2line, xdata=lnqs, ydata=lnFs)
    perr = np.sqrt(np.diag(pcov))

    lnD = popt[0]
    lnD_err = perr[0]

    D = np.exp(lnD)
    D_err = lnD_err * D

    ax.scatter(q_vector, np.reciprocal(Fs), marker='+', color="black")
    ax.plot(q_vector, np.power(q_vector, -2) * np.exp(-lnD), "-k", alpha=0.2,)
    
    txt = "Stokes-Einstein diffusion coefficient, D = {:.6f} +- {:.6f}".format(D, D_err) + "\tum^2 / s"
    print(txt)

    ax.set_yscale('log')
    ax.set_xscale('log')

    ax.xaxis.set_minor_formatter(mticker.FormatStrFormatter('%.0f'))
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%.0f'))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter('%.1f'))

    #ax.set_title(r"Charateristic decay time $\tau_c$ versus the wave-vector q.")
    #ax.legend(loc="upper right")

    ax.set(xlabel=r"Wavevector, q [$\mu m^{-1}]$")
    ax.set(ylabel=r"Decay time, $\tau_c$ [s]")

    ax.xaxis.set_tick_params(which='major', direction='in')
    ax.xaxis.set_tick_params(which='minor',  direction='in')
    ax.yaxis.set_tick_params(which='major',  direction='in')
    ax.yaxis.set_tick_params(which='minor', direction='in')



def plot_ISF(ax, ISF, tau_vector, q_vector, params, plot_arrow=False, max_qi=12):
    ax.xaxis.set_major_locator(plt.MaxNLocator(5))

    q_count = len(q_vector)
    colors = cm.get_cmap("tab10", q_count)

    # ax.set_aspect("equal")
    ax.xaxis.set_tick_params(which='major', direction='in')
    ax.xaxis.set_tick_params(which='minor',  direction='in')
    ax.yaxis.set_tick_params(which='major',  direction='in')
    ax.yaxis.set_tick_params(which='minor', direction='in')
    ax.set_yticklabels([])

    for qidx, q_val in enumerate(q_vector):
        if (qidx >= max_qi):
            break
        
        tmp_label = f"q = {q_val} " + "$\mu m^{-1}$"

        ax.scatter(tau_vector, ISF[qidx], label=tmp_label, color=colors(qidx), marker="o", s=20, alpha=.95)
        
        # tau_fit = np.linspace(min(tau_vector), max(tau_vector))
        # fits = fit_func(tau_fit, params[qidx][0], params[qidx][1], params[qidx][2])
        # ax.plot(tau_fit, fits)

        ax.set(xlabel=r"Lag time $\tau$ [s]", ylabel=r"I(q, $\tau$) [a. u.]")

    if plot_arrow:
        print("Plot arrow")
        ax.annotate("q",
            xy=(0.5, 0.001), xycoords='data',
            xytext=(0.1, 0.006), textcoords='data',
            arrowprops=dict(arrowstyle="->",
                            connectionstyle="arc3"),
            )

            # plot_all_axes = True
            # if(plot_all_axes):
            #     ax_isf.set(xlabel=r"Lag time $\tau$ [s]", ylabel=r"I(q, $\tau$) [a. u.]")
            # else:
            #     ax_isf.get_xaxis().set_visible(False)
            #     ax_isf.get_yaxis().set_visible(False)


if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Plot multi-DDM ISF for Brownian motion.")

    parser.add_argument("--root",  metavar="FILES", nargs=1,    help="Root file-path for the ISF input files.")
    parser.add_argument("--tiles", metavar="T",     nargs="+",  help="Number of tiles per full frame.")
    parser.add_argument("--scales",metavar="N",     nargs="+",  help="Length scale for each input.")
    parser.add_argument("--umpx",  help="Micrometer per pixel conversion factor.")

    args = parser.parse_args()

    file_root = args.root[0]
    tiles = args.tiles
    scales = args.scales
    um_per_px = float(args.umpx)

    # we must have the same number of tiles as scales
    if len(tiles) != len(scales):
        print("Must be a tile count for each scale.\n")
        raise argparse.ArgumentError

    # Plotting Info
    plt.rcParams['font.size'] = 14
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'

    # separate plot for each scale
    for scale_idx, scale in enumerate(scales):

        print(f"[{scale} x {scale}]")


        tile_count = int(tiles[scale_idx])
        side_len = int(np.ceil(np.sqrt(tile_count)))

        data = [read_file(file_root + str(scale) +  "-" + str(x)) for x in range(tile_count)]

        lamda_vector = data[0][0]
        lambda_count = len(lamda_vector)

        tau_vector = data[0][1]
        tau_count = len(tau_vector)

        ISF_scale = np.zeros((lambda_count, tau_count))
        for (_, _, ISF) in data:
            ISF_scale += ISF

        ISF_scale /= tile_count

        # Convert lamda values to correct units
        lamda_vector *= um_per_px 

        # Must convert the lambda values to q-values
        q_vector = np.reciprocal(lamda_vector) * (2 * np.pi) 

        # Plot axes
        ax_ISF = plt.subplot(len(scales),2 , 2*(scale_idx+1) - 1)
        ax_D   = plt.subplot(len(scales),2 , 2*(scale_idx+1))

        length = float(scale) * um_per_px

        # ax_ISF.set_title("{:.0f} $\mu m$ [{} px]".format(length, scale))
        #ax_ISF.set_title(string.ascii_uppercase[scale_idx], fontweight="bold")
        ax_ISF.annotate(string.ascii_uppercase[scale_idx],
            xy=(0.05, 0.85), xycoords='axes fraction', fontweight="bold")

        params, params_err = fit_ISF(ISF, q_vector, tau_vector)

        # Plot ISF, no fitting
        plot_ISF(ax_ISF, ISF, tau_vector, q_vector, params, plot_arrow=(scale_idx==0))

        # Fit ISF with exponentials [Params (F, A, B)]

        # Find diff coeff and plot graph
        # plot_diff_coeff(ax_D, q_vector, params, params_err)

    print(f"Time elapsed: {time.time() - start_time} seconds")

    plt.subplots_adjust(hspace=0.3, wspace=0.3)
    plt.savefig('/u/homes/yz655/net/cicutagroup/yz655/cuda_run_plot/0050/')
    plt.show()
    
