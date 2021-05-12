#!/usr/bin/python3

import argparse 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import cm
import scipy.optimize as opt 

def read_file(file_name):
    """
    Takes in a file path and reads / returns the corresponding I(q, tau) vector,
    Must be formatted as follows:

    file-start
    [0] <vector of q values>
    [1] <vector of t values>
    [2] <I(q, t) for 1st q value>
    [3] <I(q, t) for 2nd q value>
    ...
    [.] <I(q, t) for last q value>
    file-end
    """

    try:
        with open(file_name, "r") as f:
            # First 2 
            q_vector = [float(n) for n in f.readline().split()]
            tau_vector = [float(n) for n in f.readline().split()]
            
            q_count = len(q_vector)

            iqt = []
            for q in range(q_count):
                iqt.append(np.array([float(n) for n in f.readline().split()]))

    except FileNotFoundError:
        print(f"Error reading {file_name}, check file exsists / formatting")
        q_vector, tau_vector, iqt = [], [], [[]]

    return (np.array(q_vector), np.array(tau_vector), np.array(iqt))


def dampend_osc(x, ampl, freq, tau, offset):
    return ampl * (1 - np.exp(np.cos(freq * x))) * np.exp(- x / tau) + offset


def fit_iqtau(iqtau, q_vector, tau_vector):
    # Fits function of form 
    #    A * (1 - exp(- tau / T)) + B
    # Note only realy T is of interest as this can be used to calculate 
    # the diffusion coefficient

    # Returns list of parsed q

    # def func(x, A, B, T):
    #     if (- (x  * T) >)

    func = lambda x, A, B, T: A * (1 - np.exp( - x  * T, where=(- x  * T < 700))) + B

    params = []
    params_stds = []

    taus = tau_vector
    for q_idx in range(len(q_vector)):
        I = iqtau[q_idx]

        try:
            popt, pcov = opt.curve_fit(func, taus, I, p0=[0.01, 1e-3, 1], maxfev = 1500)
        except:
            print("failed")
            popt = [1,1,1]
            pcov = [(1,1,1),(1,1,1),(1,1,1)]
        
        stds = np.sqrt(np.diag(pcov))
        stds[2] = stds[2] / (popt[2]**2)
        popt[2] = 1.0 / popt[2]

        stds[np.isinf(stds)] = 0
        
        params.append(popt)
        params_stds.append(stds)


    return np.array(params), np.array(params_stds)


def parseIqtau(iqtau, q_vector, q_factor=1):
    # I(q)
    # |    _ q-max
    #  \  / \
    #   \/   '.__ q->
   
    # Data quality can be improved if we only consider values of q, larger
    # than the first maximum

    mean_iqtau = np.mean(iqtau, axis=1)

    for idx in range(len(q_vector) // 2, 0, -1):
        if mean_iqtau[idx - 1] < mean_iqtau[idx]:
            max_q_idx = idx
            break
    else:
        max_q_idx = 0

    max_q_idx = 0

    # Parsed I(q, tau) matrix
    q_vector = q_vector[max_q_idx:] * q_factor
    iqtau    = iqtau[:][max_q_idx:]

    return iqtau, q_vector


def plot_params(ax, q_vector, params, stds):
    params = np.array(params)

    Ts = params[:, 2]
    Ts_err = stds[:, 2]

    ax.set_yscale('log')
    ax.set_xscale('log')

    # Best fit line    

    line_2 = lambda x, c: -2 * x + c

    popt, pcov = opt.curve_fit(line_2, np.log(q_vector), np.log(Ts))
    stds = np.sqrt(np.diag(pcov))
    lnD = -popt[0]



    label = r"$\tau$ =" + " {:.2f}".format(np.exp(popt[0]))  + " $q^{-2}$"  

    
    D = np.exp(lnD)
    D_err = stds[0] * D

    txt = "Stokes-Einstein diffusion coeffcient, D = {:.6f}".format(D) \
        + " +- " +  "{:.6f}".format(D_err) + "\tum^2 / s"

    print(txt)

    # with open("./test_output/D_estimate", "a") as f:
    #     f.write(f"{D}, {D_err}\n")

    #fig.text(.5, .01, txt, ha='center')

    #ax.set_title(r"Charateristic decay time $\tau_c$ versus the wave-vector q.")
    ax.set(xlabel=r"Wavevector, q [$\mu m^{-1}]$")
    ax.set(ylabel=r"Decay time, $\tau_c$ [s]")
    #ax.legend(loc="upper right")
    ax.yaxis.labelpad = -10

    ax.xaxis.set_tick_params(which='major', direction='in')
    ax.xaxis.set_tick_params(which='minor',  direction='in')
    ax.yaxis.set_tick_params(which='major',  direction='in')
    ax.yaxis.set_tick_params(which='minor', direction='in')

    ax.plot(q_vector, np.power(q_vector, -2) * np.exp(-lnD), "-k", alpha=0.2, label=label)
    ax.errorbar(q_vector, Ts, yerr=Ts_err, marker='+', linestyle="None", color="black")
    #ax.set_xticks(ax.get_xticks()[::2])
    #ax.set_xlim(1e)

    # line_m = lambda x, m, c: m * x + c
    # stds_m = np.sqrt(np.diag(pcov_m))
    # label_m = r"$\tau$ =" + " {:.2f}".format(np.exp(popt_m[1]))  + " $q^{" + "{:.2f}".format(popt_m[0]) + "}$"
    # popt_m, pcov_m = opt.curve_fit(line_m, np.log(q_vector), np.log(Ts))
    # ax.plot(q_vector, np.power(q_vector, popt_m[0]) * np.exp(popt_m[1]), "-r", label=label_m)




if __name__ == "__main__":
    MAX_Q = 10

    parser = argparse.ArgumentParser(description="Plot tiled Inensitiy graphs")

    parser.add_argument("--root", metavar="FILES", nargs=1, help="file root path for the I(q ,t) input files")
    parser.add_argument("--tiles", metavar="T", nargs="+", help="number of tiles in a full frame")
    parser.add_argument("--scales", metavar="N", nargs="+", help="scale size for each input I(q, t), i.e. frame size = N * N" )
    parser.add_argument("--umpx", help="um per pixel")

    args = parser.parse_args()
    file_root = args.root[0]

    if not (len(args.tiles) == len(args.scales)):
        print("roots, scales and tiles must have same length.")
        raise argparse.ArgumentError

    plt.rcParams['font.size'] = 12
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'

    # Separate plot for each scale
    for scale_idx, scale in enumerate(args.scales):
        print(f"{scale} x {scale}")

        fig = plt.figure(scale_idx)
        tile_count = int(args.tiles[scale_idx])
        side_len = int(np.ceil(np.sqrt(tile_count)))

        data = [read_file(file_root + str(scale) +  "-" + str(x)) for x in range(tile_count)]

        for tidx, (lamda_vector, tau_vector, iqtau) in enumerate(data):
            q_vector  = np.power(lamda_vector, -1)
            q_vector *= 2 * np.pi / float(args.umpx)

            ax_isf = plt.subplot(side_len, 2 * side_len, 2 * (tidx+1) - 1)
            ax_D   = plt.subplot(side_len, 2 * side_len, 2 * (tidx+1))

            # Plot ISF vs tau
            
            q_count = min(MAX_Q, len(q_vector))

            # Gen colurs from "tab10"
            colors = cm.get_cmap("tab10", q_count)

            ax_isf.xaxis.set_tick_params(which='major', direction='in')
            ax_isf.xaxis.set_tick_params(which='minor',  direction='in')
            ax_isf.yaxis.set_tick_params(which='major',  direction='in',)
            ax_isf.yaxis.set_tick_params(which='minor', direction='in')

            for qidx, q_val in enumerate(q_vector):
                if (qidx > MAX_Q):
                    break



                ax_isf.plot(tau_vector, iqtau[qidx], label=f"q = {q_val}"+ " $\mu m^{-1}$", color=colors(qidx), marker="o", markersize=3, linestyle=None)
            
            plot_all_axes = True
            if(plot_all_axes):
                ax_isf.set(xlabel=r"Lag time $\tau$ [s]", ylabel=r"I(q, $\tau$) [a. u.]")
            else:
                ax_isf.get_xaxis().set_visible(False)
                ax_isf.get_yaxis().set_visible(False)

            # Plot Diffusion coeff

            params, params_stds = fit_iqtau(iqtau, q_vector, tau_vector)

            plot_params(ax_D, q_vector, params, params_stds)
            #except:
             #   print(f"Plotting failed for scale {scale}, tile idx {tidx}")


        # max_I = 0
        # for _, _, iqt in data:
        #     max_I = max(max_I, np.max(iqt))

        #fig.suptitle(r"I (q, $\tau$) vs Lagtime, Spatial frequency vs $\tau_c$"+ f"[frame size {args.scales[idx]} px. X {args.scales[idx]} px.]")
        # ax.legend(loc="upper left")
    
    
    
    plt.autoscale()

    plt.show()
