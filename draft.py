import os.path
import matplotlib.pyplot as plt
import bibarch

TAG = ('RESIDUS', 'MOYENS', 'RhoEtot')
# TAG = ('SCHEMA NUM', 'DTLOC', 'Reduc. max')

if __name__ == '__main__':
    def cread(names):
        root = "/scratchm/pseize/SPHERE_LOBB"
        # root = "/scratchm/pseize/CYLINDRE_LOBB"

        names = names if isinstance(names, (list, tuple)) else [names]
        return bibarch.read_histo([os.path.join(root, name) for name in names], *TAG)


    # fig = plt.figure(figsize=[12, 9.41 * 2 / 3])
    # ax = fig.add_subplot(111)
    # ax.grid(True)

    # ax.plot(*cread("BASE_NS/RUN_MTP"), label='NS MTP', lw=3)
    # ax.plot(*cread("BASE_NS/RUN_KEX"), label='NS KEX', lw=3)

    # ax.plot(*cread("BASE_NS_REFINED/RUN_MTP"), label='NS MTP fine', lw=3)
    # ax.plot(*cread("BASE_NS_REFINED/RUN_KEX"), label='NS KEX fine', lw=3)

    # ax.plot(*cread("JFNK_NS/RUN_MTP_j1"), label='NS MTP JFNK j1', lw=3)
    # ax.plot(*cread("JFNK_NS/RUN_MTP"), label='NS MTP JFNK j2', lw=3)
    # ax.plot(*cread("JFNK_NS/RUN_KEX_j1"), label='NS KEX JFNK j1', lw=1)
    # ax.plot(*cread("JFNK_NS/RUN_KEX"), label='NS KEX JFNK j2', lw=3)
    # ax.plot(*cread("JFNK_NS/RUN_MTP_j2_40"), label='NS MTP JFNK j2 40', lw=3)
    # ax.plot(*cread("JFNK_NS/RUN_KEX_40"), label='NS KEX JFNK j2 40', lw=3)
    # ax.plot(*cread("JFNK_NS/RUN_KEX_j2_noprec"), label='NS KEX JFNK j2 no prec', lw=3)
    # ax.plot(*cread("JFNK_NS/RUN_KEX_j2_noprec_fgmres"), label='NS KEX JFNK j2 no prec + FGMRES', lw=3)
    # ax.plot(*cread("JFNK_NS_REFINED/RUN_MTP"), label='NS MTP fine JFNK', lw=3)
    # ax.plot(*cread("JFNK_NS_REFINED/RUN_KEX"), label='NS KEX fine JFNK', lw=3)

    # ax.plot(*bibarch.read_histo(["/visu/pseize/SPHERE_LOBB/BASE/RUN_NS_1",
    #                              "/visu/pseize/SPHERE_LOBB/BASE/RUN_NS_2"], *tag), label='NS MTP', lw=3)
    # ax.plot(*bibarch.read_histo("/visu/pseize/SPHERE_LOBB/JFNK/RUN_NS_1", *tag), label='NS MTP JFNK j2', lw=3)
    # ax.plot(*bibarch.read_histo(["/visu/pseize/SPHERE_LOBB/JFNK/RUN_NS_j1_1",
    #                              "/visu/pseize/SPHERE_LOBB/JFNK/RUN_NS_j1_2"], *tag), label='NS MTP JFNK j1', lw=3)

    # ax.legend()
    # plt.show()

    plt.rcParams.update({
        'xtick.labelsize': 10,
        'ytick.labelsize': 10,
        'xtick.major.width': 1.5,
        'xtick.minor.width': 1,
        'ytick.major.width': 1.5,
        'ytick.minor.width': 1,
        #        'axes.grid': True,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        #        'grid.linewidth': 2,
        'legend.fontsize': 16,
        'legend.labelspacing': 1,
        # 'lines.markeredgewidth': 15,
        # 'lines.markersize': 3,
        'lines.linewidth': 3,
        'text.latex.preamble': r"\usepackage{amsmath}",
        'text.usetex': True,
        #        'figure.figsize': (13.25, 6.28),
        'figure.titlesize': 20,
    })

    fig = plt.figure(figsize=[12, 9.41 * 2 / 3])
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.grid(True)
    ax2.grid(True)
    ax1.set_xticklabels([])
    ax2.set_xticklabels([])
    fig.suptitle(r'$L_2$ norm of the residual')
    ax1.set_ylabel(r'$\rho E$', rotation=0, labelpad=20)
    ax2.set_ylabel(r'$\rho V_{y}$', rotation=0, labelpad=20)
    ax2.set_xlabel(r'$n_\textrm{iteration}$')
    ax1.plot(*bibarch.read_histo("/scratchm/pseize/SPHERE_LOBB/BASE_NS/RUN_KEX", 'RESIDUS', 'MOYENS', 'RhoEtot'),
             label='Classical method')
    ax1.plot(*bibarch.read_histo("/scratchm/pseize/SPHERE_LOBB/JFNK_NS/RUN_KEX_j1", 'RESIDUS', 'MOYENS', 'RhoEtot'),
             label='JFNK method (using first order function)')
    ax1.plot(*bibarch.read_histo("/scratchm/pseize/SPHERE_LOBB/JFNK_NS/RUN_KEX", 'RESIDUS', 'MOYENS', 'RhoEtot'),
             label='JFNK method')
    ax2.plot(*bibarch.read_histo("/scratchm/pseize/SPHERE_LOBB/BASE_NS/RUN_KEX", 'RESIDUS', 'MOYENS', 'RhoV_y'))
    ax2.plot(*bibarch.read_histo("/scratchm/pseize/SPHERE_LOBB/JFNK_NS/RUN_KEX_j1", 'RESIDUS', 'MOYENS', 'RhoV_y'))
    ax2.plot(*bibarch.read_histo("/scratchm/pseize/SPHERE_LOBB/JFNK_NS/RUN_KEX", 'RESIDUS', 'MOYENS', 'RhoV_y'))
    ax1.legend()
    fig.subplots_adjust(left=0.1, bottom=0.08, right=0.98, top=0.90, wspace=None, hspace=0.03)
    fig.savefig('/d/pseize/Figure_1.png')
    plt.show()
