#!/usr/bin/python3

import argparse 
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pylab import cm
import scipy.optimize as opt 
import time
import string




# matlab_T = "5.112359	5.046277	5.00491	4.994659	4.988849	4.970708	4.902392	4.941209	4.895422	4.927494	4.890111	4.881868	4.987734	4.950065	4.952131	5.058859	5.043625	5.095918	5.188156	5.220777	5.437645	5.526039	5.63582	5.85387	5.961353	6.034391	6.256725	6.427493	6.856493	6.992009	7.259905	7.634123	7.919676	8.466765	8.677512	9.098578	9.575456	10.0978	10.51138	11.04412	11.58328	11.99178	12.50606	13.15163	13.88196	14.33743	14.79036	15.33816	15.56812	16.58193	16.94026	17.57422	18.22788	19.2239	19.39189	20.24987	21.15813	21.0629	22.12676	22.78872	23.03205	24.02374	24.7748	25.17718	25.63604	26.74215	26.5963	27.46483	28.55121	29.10872	30.42636	30.92617	31.53899	32.42013	33.18295	33.72711	35.15404	35.75322	35.81491	37.49465	38.96638	40.42306	39.96606	40.59656	41.1779	42.25175	43.61397	43.99789	45.06893	46.33086	46.1581	48.26007	49.33541	50.7658	50.14602	52.32167	51.3773	52.20216	53.50521	53.95377	56.1175	58.51063	58.42798	59.49721	62.40104	60.52884	63.92709	64.75387	67.38456	65.00835	65.88411	66.41159	67.98379	63.42581	67.40973	68.28379	70.28844	72.19962	73.97222	72.64084	73.11863	80.17574	75.09723	75.00852	76.31732	75.32914	73.83848	77.26621	80.23615	77.07639	81.11362	78.34454	79.60375	75.9218	73.63928	78.6562	81.44227".split()
# matlab_q = "0.184078	0.190214	0.19635	0.202485	0.208621	0.214757	0.220893	0.227029	0.233165	0.239301	0.245437	0.251573	0.257709	0.263845	0.269981	0.276117	0.282252	0.288388	0.294524	0.30066	0.306796	0.312932	0.319068	0.325204	0.33134	0.337476	0.343612	0.349748	0.355884	0.362019	0.368155	0.374291	0.380427	0.386563	0.392699	0.398835	0.404971	0.411107	0.417243	0.423379	0.429515	0.435651	0.441786	0.447922	0.454058	0.460194	0.46633	0.472466	0.478602	0.484738	0.490874	0.49701	0.503146	0.509282	0.515418	0.521553	0.527689	0.533825	0.539961	0.546097	0.552233	0.558369	0.564505	0.570641	0.576777	0.582913	0.589049	0.595185	0.60132	0.607456	0.613592	0.619728	0.625864	0.632	0.638136	0.644272	0.650408	0.656544	0.66268	0.668816	0.674952	0.681087	0.687223	0.693359	0.699495	0.705631	0.711767	0.717903	0.724039	0.730175	0.736311	0.742447	0.748583	0.754719	0.760854	0.76699	0.773126	0.779262	0.785398	0.791534	0.79767	0.803806	0.809942	0.816078	0.822214	0.82835	0.834486	0.840621	0.846757	0.852893	0.859029	0.865165	0.871301	0.877437	0.883573	0.889709	0.895845	0.901981	0.908117	0.914253	0.920388	0.926524	0.93266	0.938796	0.944932	0.951068	0.957204	0.96334	0.969476	0.975612	0.981748	0.987884	0.99402	1.000155	1.006291	1.012427	1.018563".split()

# tmpq = []
# tmpt = []

# for i in range(len(matlab_q)):
#     if matlab_T[i].lower() != "nan":
#         tmpq.append(float(matlab_q[i]))
#         tmpt.append(float(matlab_T[i]))

# matlab_q = np.array(tmpq) / 0.097
# matlab_T = np.array(tmpt) 



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
    # Attempt to prevent overflow in exp
    # if (-(x * F)).any() > 700:
    #     if A < 1e-7:
    #         return B
    #     else:
    #         print("Function has blown up - returning 1")
    #         return 1
    
    # else:
    #     return A * (1 - np.exp( - x  * F)) + B



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

    Fs = params[:, 0]
    Fs_err = errs[:, 0]

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



    q1024 = np.array([float (x) for x in "0.00613592315154257	0.0122718463030851	0.0184077694546277	0.0245436926061703	0.0306796157577128	0.0368155389092554	0.0429514620607980	0.0490873852123405	0.0552233083638831	0.0613592315154257	0.0674951546669682	0.0736310778185108	0.0797670009700534	0.0859029241215959	0.0920388472731385	0.0981747704246810	0.104310693576224	0.110446616727766	0.116582539879309	0.122718463030851	0.128854386182394	0.134990309333936	0.141126232485479	0.147262155637022	0.153398078788564	0.159534001940107	0.165669925091649	0.171805848243192	0.177941771394734	0.184077694546277	0.190213617697820	0.196349540849362	0.202485464000905	0.208621387152447	0.214757310303990	0.220893233455532	0.227029156607075	0.233165079758617	0.239301002910160	0.245436926061703	0.251572849213245	0.257708772364788	0.263844695516330	0.269980618667873	0.276116541819415	0.282252464970958	0.288388388122501	0.294524311274043	0.300660234425586	0.306796157577128	0.312932080728671	0.319068003880213	0.325203927031756	0.331339850183299	0.337475773334841	0.343611696486384	0.349747619637926	0.355883542789469	0.362019465941011	0.368155389092554	0.374291312244096	0.380427235395639	0.386563158547182	0.392699081698724	0.398835004850267	0.404970928001809	0.411106851153352	0.417242774304894	0.423378697456437	0.429514620607980	0.435650543759522	0.441786466911065	0.447922390062607	0.454058313214150	0.460194236365692	0.466330159517235	0.472466082668778	0.478602005820320	0.484737928971863	0.490873852123405	0.497009775274948	0.503145698426490	0.509281621578033	0.515417544729576	0.521553467881118	0.527689391032661	0.533825314184203	0.539961237335746	0.546097160487288	0.552233083638831	0.558369006790373	0.564504929941916	0.570640853093459	0.576776776245001	0.582912699396544	0.589048622548086	0.595184545699629	0.601320468851171	0.607456392002714	0.613592315154257	0.619728238305799	0.625864161457342	0.632000084608884	0.638136007760427	0.644271930911969	0.650407854063512	0.656543777215054	0.662679700366597	0.668815623518140	0.674951546669682	0.681087469821225	0.687223392972767	0.693359316124310	0.699495239275852	0.705631162427395	0.711767085578938	0.717903008730480	0.724038931882023	0.730174855033565	0.736310778185108	0.742446701336650	0.748582624488193	0.754718547639735	0.760854470791278	0.766990393942821	0.773126317094363	0.779262240245906	0.785398163397448	0.791534086548991	0.797670009700533	0.803805932852076	0.809941856003619	0.816077779155161	0.822213702306704	0.828349625458246	0.834485548609789	0.840621471761331	0.846757394912874	0.852893318064417	0.859029241215959	0.865165164367502	0.871301087519044	0.877437010670587	0.883572933822129	0.889708856973672	0.895844780125214	0.901980703276757	0.908116626428300	0.914252549579842	0.920388472731385	0.926524395882927	0.932660319034470	0.938796242186012	0.944932165337555	0.951068088489098	0.957204011640640	0.963339934792183	0.969475857943725	0.975611781095268	0.981747704246810	0.987883627398353	0.994019550549895	1.00015547370144	1.00629139685298	1.01242732000452	1.01856324315607	1.02469916630761	1.03083508945915	1.03697101261069	1.04310693576224	1.04924285891378	1.05537878206532	1.06151470521686	1.06765062836841	1.07378655151995	1.07992247467149	1.08605839782303	1.09219432097458	1.09833024412612	1.10446616727766	1.11060209042920	1.11673801358075	1.12287393673229	1.12900985988383	1.13514578303537	1.14128170618692	1.14741762933846	1.15355355249000	1.15968947564154	1.16582539879309	1.17196132194463	1.17809724509617	1.18423316824772	1.19036909139926	1.19650501455080	1.20264093770234	1.20877686085389	1.21491278400543	1.22104870715697	1.22718463030851	1.23332055346006	1.23945647661160	1.24559239976314	1.25172832291468	1.25786424606623	1.26400016921777	1.27013609236931	1.27627201552085	1.28240793867240	1.28854386182394	1.29467978497548	1.30081570812702	1.30695163127857	1.31308755443011	1.31922347758165	1.32535940073319	1.33149532388474	1.33763124703628	1.34376717018782	1.34990309333936	1.35603901649091	1.36217493964245	1.36831086279399	1.37444678594553	1.38058270909708	1.38671863224862	1.39285455540016	1.39899047855170	1.40512640170325	1.41126232485479	1.41739824800633	1.42353417115788	1.42967009430942	1.43580601746096	1.44194194061250	1.44807786376405	1.45421378691559	1.46034971006713	1.46648563321867	1.47262155637022	1.47875747952176	1.48489340267330	1.49102932582484	1.49716524897639	1.50330117212793	1.50943709527947	1.51557301843101	1.52170894158256	1.52784486473410	1.53398078788564	1.54011671103718	1.54625263418873	1.55238855734027	1.55852448049181	1.56466040364335	1.57079632679490".split()])
    t1024 = np.array([float(x) for x in "NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	90.6518862342037	NaN	NaN	NaN	94.5824508286086	105.744691451026	NaN	104.433084042508	92.0841422747520	NaN	102.500258159248	NaN	NaN	110.232133621471	104.621543460958	92.0208915499080	NaN	NaN	91.4947972719239	107.191649491181	107.595580857088	103.072916332043	100.784070726940	101.814922614414	103.706819877144	101.002598127986	NaN	105.439115191395	113.623298849540	107.960615355876	111.664159032150	99.8906165405284	NaN	NaN	NaN	NaN	NaN	113.058287703717	97.2455792907771	NaN	NaN	114.245336532496	96.2335686853943	NaN	NaN	99.8801282953125	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	103.430572430562	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN	NaN".split()])
    ax.scatter(q1024/0.97, np.reciprocal(t1024))

    # PLOT MATLAB 
    # ml_lnFs = np.log(matlab_T)
    # ml_lnqs = np.log(matlab_q)
    # ml_popt, ml_pcov = opt.curve_fit(grad2line, xdata=ml_lnqs, ydata=ml_lnFs)
    # perr = np.sqrt(np.diag(pcov))
    # ml_lnD = ml_popt[0]
    # ml_D =  np.exp(ml_lnD) / 2
    # ml_D_err = perr[0] * ml_D
    # ax.scatter(matlab_q, np.reciprocal(matlab_T), marker='+', color="black")
    # ax.plot(matlab_q, np.power(matlab_q, -2) * np.exp(-ml_lnD), "-k", alpha=0.2,)


    ax.scatter(q_vector, np.reciprocal(Fs), marker='+', color="black")
    ax.plot(q_vector, np.power(q_vector, -2) * np.exp(-lnD), "-k", alpha=0.2,)

    
    txt = "Stokes-Einstein diffusion coeffcient, D = {:.6f} +- {:.6f}".format(D, D_err) + "\tum^2 / s"
    print(txt)

    ax.set_yscale('log')
    ax.set_xscale('log')

    #ax.set_title(r"Charateristic decay time $\tau_c$ versus the wave-vector q.")
    #ax.legend(loc="upper right")

    ax.set(xlabel=r"Wavevector, q [$\mu m^{-1}]$")
    ax.set(ylabel=r"Decay time, $\tau_c$ [s]")

    # ax.set_aspect("equal")
    ax.xaxis.set_tick_params(which='major', direction='in')
    ax.xaxis.set_tick_params(which='minor',  direction='in')
    ax.yaxis.set_tick_params(which='major',  direction='in')
    ax.yaxis.set_tick_params(which='minor', direction='in')
    ax.yaxis.labelpad = 4



def plot_ISF(ax, ISF, tau_vector, q_vector, params, plot_arrow=False, max_qi=12):
    
    q_count = len(q_vector)
    colors = cm.get_cmap("tab10", q_count)

    # ax.set_aspect("equal")
    ax.xaxis.set_tick_params(which='major', direction='in')
    ax.xaxis.set_tick_params(which='minor',  direction='in')
    ax.yaxis.set_tick_params(which='major',  direction='in')
    ax.yaxis.set_tick_params(which='minor', direction='in')

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
                xy=(1.0, 0.1), xycoords='data',
                xytext=(0.1, 0.6), textcoords='data',
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
    plt.rcParams['font.size'] = 12
    mpl.rcParams['mathtext.fontset'] = 'stix'
    mpl.rcParams['font.family'] = 'STIXGeneral'

    # separate plot for each scale
    for scale_idx, scale in enumerate(scales):

        print(f"[{scale} x {scale}]")

        fig = plt.figure(scale_idx)

        tile_count = int(tiles[scale_idx])
        side_len = int(np.ceil(np.sqrt(tile_count)))

        data = [read_file(file_root + str(scale) +  "-" + str(x)) for x in range(tile_count)]

        for tile_idx, (lamda_vector, tau_vector, ISF) in enumerate(data):
            # Convert lamda values to correct units
            lamda_vector *= um_per_px 

            # Must convert the lambda values to q-values
            q_vector = np.reciprocal(lamda_vector) * (2 * np.pi) 

            # Plot axes
            ax_ISF = plt.subplot(side_len, 2 * side_len, 2 * (tile_idx+1) - 1)
            ax_D   = plt.subplot(side_len, 2 * side_len, 2 * (tile_idx+1))

            ax_ISF.set_title(string.ascii_uppercase[tile_idx], fontweight="bold")

            params, params_err = fit_ISF(ISF, q_vector, tau_vector)

            # Plot ISF, no fitting
            plot_ISF(ax_ISF, ISF, tau_vector, q_vector, params, plot_arrow=(tile_idx==0))

            # Fit ISF with exponentials [Params (F, A, B)]

            # Find diff coeff and plot graph
            plot_diff_coeff(ax_D, q_vector, params, params_err)

    print(f"Time elapsed: {time.time() - start_time} seconds")


    plt.show()
    
