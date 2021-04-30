#!/usr/bin/python3

import argparse 
import numpy as np
import matplotlib.pyplot as plt
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

    func = lambda x, A, B, T: A * (1 - np.exp( - x  * T)) + B

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


def plot_params(ax, q_vector, params, stds, plot_labels=True):
    params = np.array(params)
    Ts = params[:, 2]
    Ts_err = stds[:, 2]

    ax.set_yscale('log')
    ax.set_xscale('log')

    # Best fit line    

    line_2 = lambda x, c: -2 * x + c
    line_m = lambda x, m, c: m * x + c

    popt_2, pcov_2 = opt.curve_fit(line_2, np.log(q_vector), np.log(Ts))
    #popt_m, pcov_m = opt.curve_fit(line_m, np.log(q_vector), np.log(Ts))

    stds_2 = np.sqrt(np.diag(pcov_2))
    #stds_m = np.sqrt(np.diag(pcov_m))

    lnD = -popt_2[0]

    #ax.set_title(r"Charateristic decay time $\tau$ versus the wave-vector q.")

    label_2 = r"$\tau$ =" + " {:.2f}".format(np.exp(popt_2[0]))  + " $q^{-2}$"
    #label_m = r"$\tau$ =" + " {:.2f}".format(np.exp(popt_m[1]))  + " $q^{" + "{:.2f}".format(popt_m[0]) + "}$"

    ax.plot(q_vector, np.power(q_vector, -2) * np.exp(-lnD), "-k", alpha=0.2, label=label_2)
    #ax.plot(q_vector, np.power(q_vector, popt_m[0]) * np.exp(popt_m[1]), "-r", label=label_m)

    # Plot main data

    ax.errorbar(q_vector, Ts, yerr=Ts_err, marker='+', linestyle="None", color="black")

    D = np.exp(lnD)
    D_err = stds_2[0] * D


    txt = "Fitting of the data gives a value of the Stokes-Einstein diffusion coeffcient, $D_m$ $\equal$ {:.6f}".format(D) \
        + " $\pm$ " +  "{:.6f}".format(D_err) + " $\mu m s^{-1}$"

    print(txt)

    # with open("./test_output/D_estimate", "a") as f:
    #     f.write(f"{D}, {D_err}\n")

    #fig.text(.5, .01, txt, ha='center')

    if (plot_labels):
        ax.set(xlabel=r"Spatial frequency q [$px^{-1}]$", ylabel=r"$\tau_c(q)$ [s]")
        #ax.legend(loc="upper right")
        ax.yaxis.labelpad = -10

    else:
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot tiled Inensitiy graphs")

    parser.add_argument("--root", metavar="FILES", nargs="+", help="file root path for the I(q ,t) input files")
    parser.add_argument("--tiles", metavar="T", nargs="+", help="number of tiles in a full frame")
    parser.add_argument("--scales", metavar="N", nargs="+", help="scale size for each input I(q, t), i.e. frame size = N * N" )

    args = parser.parse_args()

    root = args.root[0]

    if not (len(args.tiles) == len(args.scales)):
        print("roots, scales and tiles must have same length.")
        raise argparse.ArgumentError

    for idx, scale in enumerate(args.scales):
        fig = plt.figure(idx)

        print(f"Scale {scale} x {scale}")

        tile_count = int(args.tiles[idx])
        side_len = int(np.ceil(np.sqrt(tile_count)))
        
        data = [read_file(root + str(scale) +  "-" + str(tidx)) for tidx in range(tile_count)]

        # max_I = 0
        # for _, _, iqt in data:
        #     max_I = max(max_I, np.max(iqt))

        #fig.suptitle(r"I (q, $\tau$) vs Lagtime, Spatial frequency vs $\tau_c$"+ f"[frame size {args.scales[idx]} px. X {args.scales[idx]} px.]")
        for tidx, (q_vector, tau_vector, iqtau) in enumerate(data):

            ax = plt.subplot(side_len, 2*side_len, 2*(tidx+1) - 1)
            ax2 = plt.subplot(side_len, 2*side_len, 2*(tidx+1))
            
            params, params_stds = fit_iqtau(iqtau, q_vector, tau_vector)

            plot_axis = tidx == side_len * (side_len - 1)

            try:
                plot_params(ax2, q_vector, params, params_stds, plot_axis)
            except:
                print("fail")

            for qidx, q_val in enumerate(q_vector):
                ax.plot(tau_vector, iqtau[qidx], label=f"q = {q_val}"+ " $px^{-1}$")
                # ax.set_ylim(0, max_I)
                
            if(plot_axis):
                ax.set(xlabel=r"Lag time $\tau$ [s]", ylabel=r"I(q, $\tau$) [a. u.]")
            else:
                ax.get_xaxis().set_visible(False)
                ax.get_yaxis().set_visible(False)

            #ax.legend(loc="upper left")
    plt.autoscale()

    plt.show()
