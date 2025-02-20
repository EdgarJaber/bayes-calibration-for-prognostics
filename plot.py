import matplotlib.pyplot as plt


def plot_tpd(axs, input_dic, y, simulation_time, label):
    """
    Plot the clogging trajectories of the PCE metamodel

    Parameters
    ----------
    input_dic: dict
        dictionary containing the input parameters of the PCE model
    y: numpy.ndarray
        clogging trajectories
    simulation_time: numpy.ndarray
        simulation time
    label: str
        label to be used in the legend of the plot

    Returns
    -------
    fig: matplotlib.figure.Figure
        figure object of the plot
    axs: matplotlib.axes._subplots.AxesSubplot
        axes object of the plot
    """
    nrow, ncol = 1, 1
    #fig, axs = plt.subplots(nrow, ncol, figsize=(10,6), layout='tight')

    colors_chem_cond = {'C1': {'name': r'$\chi_{1}$', 'low': 'gainsboro', 'high': 'silver'},
                      'C2': {'name': r'$\chi_{2}$', 'high': 'darkgray'}}

    #Chemical conditioning
    cond = input_dic['chemistry']['conditioning']
    for k in range(len(cond)):
        if k == len(cond)-1:
            t_i, t_f = cond[k]['time']/24, max(simulation_time)
        else:
            t_i, t_f = cond[k]['time']/24, cond[k+1]['time']/24
        c = cond[k]
        typ = colors_chem_cond[c['type']]
        col = typ[str(c['ph'])]
        name = typ['name']
        axs.axvspan(t_i, t_f, facecolor=col, alpha=0.8, label='{0}, {1} pH'.format(name, c['ph']))

    #Clogging trajectories
    axs.grid()
    n = y.shape[0]
    d = y.shape[1]
    
    if d > 1:
        for i in range(n):   
            axs.plot(simulation_time, y[i], c='g', alpha=0.1)
            if i == n-1:
                axs.plot(simulation_time, y[i], c='g', alpha=0.5, label=label)
    else:
        axs.scatter(simulation_time, y, c='g', s=5, label=label, zorder=10)

    axs.set_xlim(250,17500)
    axs.set_ylim(0,100)

    axs.set_xlabel(r'$t\;(d)$', fontsize=25)
    axs.set_ylabel(r'$\tau_c$', fontsize=25)

    axs.axvline(x=[10996.875], color='darkorange', linewidth=1.5, label=r'$t_{P}$')

    #Cleaning dates
    nett = input_dic['chemistry']['cleaning']
    for k in range(len(nett)):
        if nett[k]['type'] == 'curative':
            axs.axvline(x=nett[k]['time']/24, color='k', linestyle='-', linewidth=1.5, label='Curative cleaning')

        elif nett[k]['type'] == 'preventive':
            axs.axvline(x=nett[k]['time']/24, color='k', linestyle='--', linewidth=1.5, label='Preventive cleaning')

    #Legend
    #lines_labels = [ax.get_legend_handles_labels() for ax in fig.axes]
    #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #fig.legend(lines, labels, bbox_to_anchor=(-0.01,0.9), fontsize=20)

    return axs
