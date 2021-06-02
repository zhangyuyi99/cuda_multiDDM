#!/usr/bin/python3

import argparse 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import cm
import scipy.optimize as opt 
from scipy import fftpack
import time

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




if __name__ == "__main__":
    start_time = time.time()

    parser = argparse.ArgumentParser(description="Plot multi-DDM ISF for Brownian motion.")

    parser.add_argument("--root",  metavar="FILES", nargs=1,    help="Root file-path for the ISF input files.")
    parser.add_argument("--tiles", metavar="T",     nargs="+",  help="Number of tiles per full frame.")
    parser.add_argument("--scales",metavar="N",     nargs="+",  help="Length scale for each input.")

    args = parser.parse_args()

    file_root = args.root[0]
    tiles = args.tiles
    scales = args.scales

    # we must have the same number of tiles as scales

    if len(tiles) != len(scales):
        print("Must be a tile count for each scale.\n")
        raise argparse.ArgumentError


    # separate plot for each scale
    for scale_idx, scale in enumerate(scales):

        print(f"[{scale} x {scale}]")


        tile_count = int(tiles[scale_idx])
        side_len = int(np.ceil(np.sqrt(tile_count)))

        if tile_count != 4:
            continue

        data = [read_file(file_root + str(scale) +  "-" + str(x)) for x in range(tile_count)]

        # for tile_idx, (lamda_vector, tau_vector, ISF) in enumerate(data):
        #     # Plot axes
        #     ax = plt.subplot(side_len, side_len, tile_idx+1)
            
        #     ISF_avg = np.average(ISF, axis=0, weights=[(1 < i < 8) for i in range(ISF.shape[0])])
        #     ISF_avg -= np.mean(ISF_avg)



        #     ax.xaxis.set_tick_params(which='major', direction='in')
        #     ax.xaxis.set_tick_params(which='minor',  direction='in')
        #     ax.yaxis.set_tick_params(which='major',  direction='in')
        #     ax.yaxis.set_tick_params(which='minor', direction='in')

        #     f_s = 7
        #     X = fftpack.fft(ISF_avg)
        #     freqs = fftpack.fftfreq(len(ISF_avg)) * f_s
        #     ax.stem(freqs, np.abs(X))
        #     ax.set_xlim(0, f_s / 2)

        for tile_idx, (lamda_vector, tau_vector, ISF) in enumerate(data):

            for l in range(len(lamda_vector)):
                print(l)
                fig, ax = plt.subplots()
                ax.set(title=f'l = {l}')
                ax.plot(tau_vector, ISF[l], color="black")
                fig.show()


        # fig = plt.figure(len(scales) + scale_idx)
        # for tile_idx, (lamda_vector, tau_vector, ISF) in enumerate(data):
        #     # Plot axes
        #     ax = plt.subplot(side_len, side_len, tile_idx+1)
            

        #     ISF_avg = np.average(ISF, axis=0, weights=[(0 < i < 10) for i in range(ISF.shape[0])])
    

        #     ax.xaxis.set_tick_params(which='major', direction='in')
        #     ax.xaxis.set_tick_params(which='minor',  direction='in')
        #     ax.yaxis.set_tick_params(which='major',  direction='in')
        #     ax.yaxis.set_tick_params(which='minor', direction='in')

        #     # from scipy.signal import argrelextrema
        #     # local_maxima = argrelextrema(ISF_avg, np.less)[0]
            
        #     # left = local_maxima[2] + 1
        #     # right = local_maxima[-1] - 1

        #     # ISF_avg = ISF_avg[left:right]
        #     # tau_vector = tau_vector[left:right]

        #     ax.plot(tau_vector, ISF_avg, color="black")



    plt.show()

    
    print(f"Time elapsed: {time.time() - start_time} seconds")
    
