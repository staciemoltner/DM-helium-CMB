import os,sys
import numpy as np
import matplotlib
import matplotlib.pyplot as plt


####################


# model = ['p1n0_H', 'p0n1_He', 'p1n0_H_He', 'p1n1_H_He']
def chains_results(model):
    """
    Imports chain files, creates triangle plots and .txt files of cross 
    section limits for all models and all n.
    """
    pathy = '../DMeff-neutron/{}_chains'.format(model)
    trifile = '../DMeff-neutron/triangle-plots/'

    ##########
    
    # reduced mass dictionary
    mu_H = {1e-5: 1e-5, 1e-3: 9.989e-4, 1e-1: 9.037e-2, 1: 0.4842, 1e1: 0.8582, 1e2: 0.9301, 1e3: 0.9379}
    mu_He = {1e-5: 1e-5, 1e-3: 9.997e-4, 1e-1: 9.739e-2, 1: 0.7885, 1e1: 2.716, 1e2: 3.594, 1e3: 3.714}
    
    # headings dictionary
    modelheadings = {'p1n0_H':['n', 'DM Mass (GeV)', 'sigma_H'], 
                     'p0n1_He':['n', 'DM Mass (GeV)', 'sigma_He'], 
                     'p1n0_H_He':['n', 'DM Mass (GeV)', 'sigma_H', 'sigma_He'], 
                     'p1n1_H_He':['n', 'DM Mass (GeV)', 'sigma_H', 'sigma_He']}

    ##########
    
    # imported stuff
    from matplotlib.backends.backend_pdf import PdfPages

    from os import listdir
    from os.path import isfile, join
    import glob
    import getdist
    import numpy as np

    from getdist import loadMCSamples

    ##########

    # create results files:
    pp = PdfPages(trifile + '{}_allnpow_weighted.pdf'.format(model))
    title = '../DMeff-neutron/{}_weighted.txt'.format(model)

    file = open(title, "w")
    file.write("#")  # comment out column headings
    
    headings = modelheadings[model]
    for k in range(len(list(headings))):
        file.write(list(headings)[k] + "  ")
    file.write("\n")
    
    ##########

    # triangle plot analysis settings, 
    # https://getdist.readthedocs.io/en/latest/analysis_settings.html?highlight=smooth_scale_1D#analysis-settings
    burnin = 0.3
    smooth1D = -1  # if -1: set optimized smoothing bandwidth automatically for each parameter
    smooth2D = -1  # if -1: automatic optimized bandwidth matrix selection

    ##########
    
    a = (glob.glob(pathy+'/*')) 
    for i in range(len(a)):
        b = (glob.glob(a[i]+'/*'))

        for j in range(len(b)):
            sample = str(b[j] + '/' + b[j].split('/')[-1])

            # load chains
            gd_AP = getdist.loadMCSamples(sample, settings={'ignore_rows': burnin, 'smooth_scale_1D': smooth1D, 'smooth_scale_2D': smooth2D})

            if model=='p1n0_H' or model=='p0n1_He':
                
                n_a = (a[i].split('/')[-1]).split('_')[1]  # gives "npow*" where * = n
                # weight by cross section
                σ = gd_AP.getParams().log_sigma_dmeff
                m = float((b[j].split('/')[-1]).split('_')[2].split('G')[0])
                n = n_a.split('npow')[-1]

                file.write(str(n) + "  " 
                           + '{0:1.0e}'.format(m) + "  " 
                           + str(10**weighted_quantile(σ, 0.95, sample_weight=σ)) 
                           + "\n")

                # progress plots, titled by chain filenames
                from cobaya.samplers.mcmc import plot_progress
                plot_progress(sample)
                plt.title(b[j].split('/')[-1])
                plt.tight_layout()
                pp.savefig()

                # triangle plots
                from getdist import plots
                g = plots.get_subplot_plotter()
                g.triangle_plot([gd_AP], ['omega_b', 'omega_dmeff', 'H0', 'logA', 'n_s', 'tau_reio', 'log_sigma_dmeff'],
                                filled=True, legend_loc='upper right', 
                                param_limits={'log_sigma_dmeff':(gd_AP.getLower('log_sigma_dmeff'), gd_AP.getUpper('log_sigma_dmeff'))}, # plots full sampled range of log_sigmaH_dmeff
                                title_limit=2, # first title limit (for 1D plots) is 68% by default
                               )
                pp.savefig()
                plt.savefig(trifile + '/' + model + '_triangle-plots/' + b[j].split('/')[-1] + '.pdf')
                print(trifile + '/' + model + '_triangle-plots/' + b[j].split('/')[-1] + '.pdf')

            elif model=='p1n0_H_He':
                n_a = (a[i].split('/')[-1]).split('_')[-2]  # gives "npow*" where * = n
                # weight by cross section
                σH = gd_AP.getParams().log_sigmaH_dmeff
                m = float((b[j].split('/')[-1]).split('_')[4].split('G')[0])
                n = n_a.split('npow')[-1]

                if n == '-4':
                    σHe = np.log10(4) + weighted_quantile(σH, 0.95, sample_weight=σH) + 2*np.log10(mu_H[m]/mu_He[m])

                elif n == '-2':
                    σHe = np.log10(4) + weighted_quantile(σH, 0.95, sample_weight=σH)

                elif n == '0':
                    σHe = np.log10(4) + weighted_quantile(σH, 0.95, sample_weight=σH) + 2*np.log10(mu_He[m]/mu_H[m])

                elif n == '2':
                    σHe = np.log10(4) + weighted_quantile(σH, 0.95, sample_weight=σH) + 4*np.log10(mu_He[m]/mu_H[m])

                file.write(n_a.split('npow')[-1] + "  " 
                           + '{0:1.0e}'.format(m) + "  " 
                           + str(10**weighted_quantile(σH, 0.95, sample_weight=σH)) + "  "
                           + str(10**σHe) 
                           + "\n")

                # progress plots, titled by chain filenames
                from cobaya.samplers.mcmc import plot_progress
                plot_progress(sample)
                plt.title(b[j].split('/')[-1])
                plt.tight_layout()
                pp.savefig()

                # triangle plots
                from getdist import plots
                g = plots.get_subplot_plotter()
                g.triangle_plot([gd_AP], ['omega_b', 'omega_dmeff', 'H0', 'logA', 'n_s', 'tau_reio', 'log_sigmaH_dmeff'],
                                filled=True, legend_loc='upper right', 
                                param_limits={'log_sigmaH_dmeff':(gd_AP.getLower('log_sigmaH_dmeff'), gd_AP.getUpper('log_sigmaH_dmeff'))}, # plots full sampled range of log_sigmaH_dmeff
                                title_limit=2, # first title limit (for 1D plots) is 68% by default
                               )
                pp.savefig()
                plt.savefig(trifile + '/' + model + '_triangle-plots/' + b[j].split('/')[-1] + '.pdf')
                print(trifile + '/' + model + '_triangle-plots/' + b[j].split('/')[-1] + '.pdf')

            elif model=='p1n1_H_He':
                n_a = (a[i].split('/')[-1]).split('_')[-2]  # gives "npow*" where * = n
                # weight by cross section
                σH = gd_AP.getParams().log_sigmaH_dmeff
                m = float((b[j].split('/')[-1]).split('_')[4].split('G')[0])
                n = n_a.split('npow')[-1]

                if n == '-4':
                    σHe = np.log10(16) + weighted_quantile(σH, 0.95, sample_weight=σH) + 2*np.log10(mu_H[m]/mu_He[m])

                elif n == '-2':
                    σHe = np.log10(16) + weighted_quantile(σH, 0.95, sample_weight=σH)

                elif n == '0':
                    σHe = np.log10(16) + weighted_quantile(σH, 0.95, sample_weight=σH) + 2*np.log10(mu_He[m]/mu_H[m])

                elif n == '2':
                    σHe = np.log10(16) + weighted_quantile(σH, 0.95, sample_weight=σH) + 4*np.log10(mu_He[m]/mu_H[m])

                file.write(n_a.split('npow')[-1] + "  " 
                           + '{0:1.0e}'.format(m) + "  " 
                           + str(10**weighted_quantile(σH, 0.95, sample_weight=σH)) + "  "
                           + str(10**σHe) 
                           + "\n")

                # progress plots, titled by chain filenames
                from cobaya.samplers.mcmc import plot_progress
                plot_progress(sample)
                plt.title(b[j].split('/')[-1])
                plt.tight_layout()
                pp.savefig()

                # triangle plots
                from getdist import plots
                g = plots.get_subplot_plotter()
                g.triangle_plot([gd_AP], ['omega_b', 'omega_dmeff', 'H0', 'logA', 'n_s', 'tau_reio', 'log_sigmaH_dmeff'],
                                filled=True, legend_loc='upper right', 
                                param_limits={'log_sigmaH_dmeff':(gd_AP.getLower('log_sigmaH_dmeff'), gd_AP.getUpper('log_sigmaH_dmeff'))}, # plots full sampled range of log_sigmaH_dmeff
                                title_limit=2, # first title limit (for 1D plots) is 68% by default
                               )
                pp.savefig()
                plt.savefig(trifile + '/' + model + '_triangle-plots/' + b[j].split('/')[-1] + '.pdf')
                print(trifile + '/' + model + '_triangle-plots/' + b[j].split('/')[-1] + '.pdf')

    pp.close() 
    plt.close()
    file.close()


####################


# from https://stackoverflow.com/questions/21844024/weighted-percentile-using-numpy
def weighted_quantile(values, quantiles, sample_weight=None, 
                      values_sorted=False, old_style=False):
    """ Very close to numpy.percentile, but supports weights.
    NOTE: quantiles should be in [0, 1]!
    :param values: numpy.array with data
    :param quantiles: array-like with many quantiles needed
    :param sample_weight: array-like of the same length as `array`
    :param values_sorted: bool, if True, then will avoid sorting of
        initial array
    :param old_style: if True, will correct output to be consistent
        with numpy.percentile.
    :return: numpy.array with computed quantiles.
    """
    values = np.array(values)
    quantiles = np.array(quantiles)
    if sample_weight is None:
        sample_weight = np.ones(len(values))
    sample_weight = np.array(sample_weight)
    assert np.all(quantiles >= 0) and np.all(quantiles <= 1), \
        'quantiles should be in [0, 1]'
    if not values_sorted:
        sorter = np.argsort(values)
        values = values[sorter]
        sample_weight = sample_weight[sorter]
    weighted_quantiles = np.cumsum(sample_weight) - 0.5 * sample_weight
    if old_style:
        # To be convenient with numpy.percentile
        weighted_quantiles -= weighted_quantiles[0]
        weighted_quantiles /= weighted_quantiles[-1]
    else:
        weighted_quantiles /= np.sum(sample_weight)
    return np.interp(quantiles, weighted_quantiles, values)


####################


def plots(n):
    """
    Creates plot for given value of n and its relevant models.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    from matplotlib.ticker import (FormatStrFormatter)
    model = ['p0n1_He', 'p1n1_H_He', 'p1n0_H_He', 'p1n0_H']
    
    model_labels = {'p0n1_He': r'Scenario A', # neutron only
                    'p1n1_H_He': r'Scenario C-SI', # coupling to both neutrons, protons
                    'p1n0_H_He': r'Scenario B-SI', # proton only, SI
                    'p1n0_H': r'Scenario B/C-SD' # proton only, SD
                    }
    colours = {'p0n1_He': '#d7191c', 'p1n1_H_He': '#fdae61', 'p1n0_H_He': '#8db5c2', 'p1n0_H': '#1f5782'}
    styles = {'H': '-', 'He': '--'}
    labels = {'H': 'Hydrogen', 'He': 'Helium'}

    fig, ax = plt.subplots()
    ax.set_xscale('log'), ax.set_xlim(10**-5, 10**3), ax.set_yscale('log')
    ax.grid(b=True, which='major', axis='both', linestyle=':')

    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    ax.yaxis.get_ticklocs(minor=True)
    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='minor', bottom=False)
    
    if n==2:
        ax.set_yticks([1e-24, 1e-23, 1e-22, 1e-21, 1e-20, 1e-19, 1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13])
        ax.set_yticklabels(['','','$10^{-22}$','','$10^{-20}$','','$10^{-18}$','','$10^{-16}$','','$10^{-14}$',''])

    if n==4:
        ax.set_yticks([1e-21, 1e-20, 1e-19, 1e-18, 1e-17, 1e-16, 1e-15, 1e-14, 1e-13, 1e-12, 1e-11, 1e-10, 1e-9, 1e-8, 1e-7, 1e-6, 1e-5])
        ax.set_yticklabels(['','$10^{-20}$','','$10^{-18}$','','$10^{-16}$','','$10^{-14}$','','$10^{-12}$','','$10^{-10}$','','$10^{-8}$','','$10^{-6}$',''])

    ax.set_xlabel(r'Dark matter mass $m_\chi ~ \mathrm{[GeV]}$', fontsize=14), 
    ax.set_ylabel(r'H, He cross section $\mathrm{[cm^2]}$', fontsize=14)
    
    for i in model:
        datafile = '../DMeff-neutron/{}_weighted.txt'.format(i)

        if i=='p1n0_H' or i=='p0n1_He':
            n_list, mass_list, sigma_list = np.loadtxt(datafile,unpack=True)
            mass = mass_list[n_list==n]
            if mass.size==0:
                continue
            sigma = sigma_list[n_list==n]
            idx_sort = np.argsort(mass)

            ax.plot(mass[idx_sort], sigma[idx_sort], linestyle=styles[i.split("_")[-1]], label=model_labels[i], c=colours[i])

        else:
            n_list, mass_list, sigma_H_list, sigma_He_list = np.loadtxt(datafile,unpack=True)
            mass = mass_list[n_list==n]
            if mass.size==0:
                continue
            sigma_H = sigma_H_list[n_list==n]
            sigma_He = sigma_He_list[n_list==n]

            idx_sort = np.argsort(mass)

            ax.plot(mass[idx_sort], sigma_H[idx_sort], linestyle=styles['H'], label=model_labels[i], c=colours[i])
            ax.plot(mass[idx_sort], sigma_He[idx_sort], linestyle=styles['He'], c=colours[i])

    p0n1_He = mpatches.Patch(color=colours['p0n1_He'], label=model_labels['p0n1_He'])
    He_n4 = mpatches.Patch(color=colours['p0n1_He'], label='Scenario A generic')
    p1n1_H_He = mpatches.Patch(color=colours['p1n1_H_He'], label=model_labels['p1n1_H_He'])
    p1n0_H_He = mpatches.Patch(color=colours['p1n0_H_He'], label=model_labels['p1n0_H_He'])
    p1n0_H = mpatches.Patch(color=colours['p1n0_H'], label=model_labels['p1n0_H'])
    H_nm4 = mpatches.Patch(color=colours['p1n0_H'], label='Scenario B/C-SD generic') # new
    
    # sigma legend
    ax2 = ax.twinx()
    for j, k in (styles.items()):
        ax2.plot(np.NaN, np.NaN, ls=styles[j], label=labels[j], c='black')
    ax2.get_yaxis().set_visible(False)
    ax2.legend(loc = 4, fontsize=11)

    # model legend            
    if n == -4:
        ax.legend(loc = 2, handles=[p0n1_He, p1n0_H_He, H_nm4, p1n1_H_He], fontsize=11) # modified
        ax.get_legend().set_title(r"$\mathrm{{n = {{{}}}}}$".format(n), prop={'size':11})
    
    elif n == 4:
        ax.legend(loc = 2, handles=[He_n4, p1n0_H], fontsize=11)
        ax.get_legend().set_title(r"$\mathrm{{n = {{{}}}}}$".format(n), prop={'size':11})
        
    else:
        ax.legend(loc = 2, handles=[p0n1_He, p1n0_H_He, p1n0_H, p1n1_H_He], fontsize=11)
        ax.get_legend().set_title(r"$\mathrm{{n = {{{}}}}}$".format(n), prop={'size':11})

    plt.savefig('../DMeff-neutron/npow{}.pdf'.format(n), bbox_inches="tight")


####################


def allplots():
    n = [-4, -2, 0, 2, 4]
	
    for i in n:
        plots(i)


####################


def get_results(n, model):
    """
    Returns results for a given velocity power $n$ and interaction model.
    """
    datafile = '../DMeff-neutron/{}_weighted.txt'.format(model)

    if model=='p1n0_H' or model=='p0n1_He':
        n_list, mass_list, sigma_list = np.loadtxt(datafile,unpack=True)
        n_current = n_list[n_list==n]
        mass = mass_list[n_list==n]
        sigma = sigma_list[n_list==n]
        idx_sort = np.argsort(mass)
        m, s = mass[idx_sort], sigma[idx_sort]
        return(m, s)

    else:
        n_list,mass_list,sigma_H_list,sigma_He_list = np.loadtxt(datafile,unpack=True)
        mass = mass_list[n_list==n]
        sigma_H = sigma_H_list[n_list==n]
        sigma_He = sigma_He_list[n_list==n]
        idx_sort = np.argsort(mass)
        m, s_H, s_He = mass[idx_sort], sigma_H[idx_sort], sigma_He[idx_sort]
        return(m, s_H, s_He)


####################


def latex_table():
    """
    Creates latex table of results.
    """
    n = [-4,-2,0,2,4]
    m = ['10 & keV', '1 & MeV', '100 & MeV', '1 & GeV', '10 & GeV', '100 & GeV', '1 & TeV']
    model = ['p0n1_He', 'p1n0_H_He', 'p1n0_H', 'p1n1_H_He']

    print(r'\begin{table*}[ht]')
    print(r'\centering')
    print(r'\begin{tabular}{|c|>{\centering}r@{\ }l|>{\centering}m{2.2cm}|>{\centering}m{2.2cm}|>{\centering}m{2.2cm}|>{\centering}m{2.2cm}|>{\centering}m{2.2cm}|>{\centering\arraybackslash}m{2.2cm}|}')
    print(r'\hline')
    print(r'\multirow{2}{1cm}{\centering $n$}')
    print(r'& \multicolumn{2}{c|}{\multirow{2}{1.4cm}{\centering DM Mass}}')
    print(r'& Scenario A')
    print(r'& \multicolumn{2}{c|}{Scenario B-SI}')
    print(r'& Scenario B/C-SD')
    print(r'& \multicolumn{2}{c|}{Scenario C-SI}\\')
    print(r'\cline{4-9}')
    print(r'& & & He scattering')
    print(r'& H scattering & He scattering')
    print(r'& H scattering')
    print(r'& H scattering & He scattering \\')
    print(r'\hline')

    for i in n:
        print(i)
        
        for j in range(len(m)):
            print("& " + m[j], end=" ")
            for k in model:

                if k=='p1n0_H' or k=='p0n1_He':
                    mass, sigma = get_results(i, k)
                    if sigma.size==0:
                        print("& ", end=" ")
                        continue
                    print("& " + '{0:1.1e}'.format(sigma[j]), end=" ")

                else:
                    mass, sigma_H, sigma_He = get_results(i, k)
                    if sigma_H.size==0:
                        print("&  & ", end=" ")
                        continue
                    print("& " + '{0:1.1e}'.format(sigma_H[j]) + " & " + '{0:1.1e}'.format(sigma_He[j]), end=" ")

            print(r"\\")
        print(r"\hline")
    print(r'\end{tabular}')
    print(r'\caption{}')
    print(r'\label{tab:limits}')
    print(r'\end{table*}')


################################################################################


if __name__=="__main__":

	# set to True if you want to run all chains files (takes a while)
	allchains = False
	"""
	Creates all triangle plots and .txt files of cross section limits 
	for all models and all n.
	"""
	if allchains:
		model = ['p0n1_He', 'p1n1_H_He', 'p1n0_H_He', 'p1n0_H']
		for i in model:
			chains_results(i)
			
	allplots()
	latex_table()
