import os.path

import matplotlib.pyplot as plt
import numpy as np

import bibarch
import utils

TREE = {
    'HORLOGE': {'DATES': ['Temps', ],
                'CYCLES': ['Ncycle', ]},
    'ETATS': {'MINIMA': ['P', 'T', 'V_x', 'V_y', 'V_z', 'Y_F1:P1:N2tr', 'Y_F1:P1:O2tr', 'Y_F1:P1:NO', 'Y_F1:P1:N',
                         'Y_F1:P1:O', 'Y_F1:P1:N2p', 'Y_F1:P1:O2p', 'Y_F1:P1:NOp', 'Y_F1:P1:Np', 'Y_F1:P1:Op',
                         'Y_F4:P1:e', 'Tv_F2:P1:N2v', 'Tv_F3:P1:O2v', 'Te', 'Pe'],
              'MAXIMA': ['P', 'T', 'V_x', 'V_y', 'V_z', 'Y_F1:P1:N2tr', 'Y_F1:P1:O2tr', 'Y_F1:P1:NO', 'Y_F1:P1:N',
                         'Y_F1:P1:O', 'Y_F1:P1:N2p', 'Y_F1:P1:O2p', 'Y_F1:P1:NOp', 'Y_F1:P1:Np', 'Y_F1:P1:Op',
                         'Y_F4:P1:e', 'Tv_F2:P1:N2v', 'Tv_F3:P1:O2v', 'Te', 'Pe'],
              'MOYENS': ['P', 'T', 'V_x', 'V_y', 'V_z', 'Y_F1:P1:N2tr', 'Y_F1:P1:O2tr', 'Y_F1:P1:NO', 'Y_F1:P1:N',
                         'Y_F1:P1:O', 'Y_F1:P1:N2p', 'Y_F1:P1:O2p', 'Y_F1:P1:NOp', 'Y_F1:P1:Np', 'Y_F1:P1:Op',
                         'Y_F4:P1:e', 'Tv_F2:P1:N2v', 'Tv_F3:P1:O2v', 'Te', 'Pe']},
    'RESIDUS': {'MAXIMA': ['RhoY_F1:P1:N2tr', 'RhoY_F1:P1:O2tr', 'RhoY_F1:P1:NO', 'RhoY_F1:P1:N', 'RhoY_F1:P1:O',
                           'RhoY_F1:P1:N2p', 'RhoY_F1:P1:O2p', 'RhoY_F1:P1:NOp', 'RhoY_F1:P1:Np', 'RhoY_F1:P1:Op',
                           'RhoY_F4:P1:e', 'RhoV_x', 'RhoV_y', 'RhoV_z', 'RhoF2:P1:N2v_EvF2:P1:N2v',
                           'RhoF3:P1:O2v_EvF3:P1:O2v', 'Rhoe_Ee', 'RhoEtot'],
                'MOYENS': ['RhoY_F1:P1:N2tr', 'RhoY_F1:P1:O2tr', 'RhoY_F1:P1:NO', 'RhoY_F1:P1:N', 'RhoY_F1:P1:O',
                           'RhoY_F1:P1:N2p', 'RhoY_F1:P1:O2p', 'RhoY_F1:P1:NOp', 'RhoY_F1:P1:Np', 'RhoY_F1:P1:Op',
                           'RhoY_F4:P1:e', 'RhoV_x', 'RhoV_y', 'RhoV_z', 'RhoF2:P1:N2v_EvF2:P1:N2v',
                           'RhoF3:P1:O2v_EvF3:P1:O2v', 'Rhoe_Ee', 'RhoEtot'],
                'GLOBAUX': ['RhoY_F1:P1:N2tr', 'RhoY_F1:P1:O2tr', 'RhoY_F1:P1:NO', 'RhoY_F1:P1:N', 'RhoY_F1:P1:O',
                            'RhoY_F1:P1:N2p', 'RhoY_F1:P1:O2p', 'RhoY_F1:P1:NOp', 'RhoY_F1:P1:Np', 'RhoY_F1:P1:Op',
                            'RhoY_F4:P1:e', 'RhoV_x', 'RhoV_y', 'RhoV_z', 'RhoF2:P1:N2v_EvF2:P1:N2v',
                            'RhoF3:P1:O2v_EvF3:P1:O2v', 'Rhoe_Ee', 'RhoEtot']},
    'SCHEMA NUM': {'DTLOC': ['Dtloc min', 'Dtloc max', 'Niter max', 'Niter min', 'Reduc. max', 'Reduc. min']}}
TAG = ('RESIDUS', 'MOYENS', 'RhoEtot')
ROOT = "/scratchm/pseize/SPHERE_LOBB"

MODE = ('ITER', 'TIME', 'WALL')[2]


class Creader:
    offset = 0

    @staticmethod
    def __call__(names, save_offset=False):
        names = names if isinstance(names, (list, tuple)) else [names]

        if MODE == 'WALL':
            x_data = np.array([])
            y_data = np.array([])
            offset = Creader.offset
            for name in names:
                t = utils.fetch_suivi_stats(*utils.fetch_file(os.path.join(os.path.join(ROOT, name), 'suivi.1')))
                if t is None:
                    t = utils.fetch_slurm_stats(*utils.fetch_file(os.path.join(os.path.join(ROOT, name), 'slurm.*.out')))[0]
                x, y = bibarch.read_histo(os.path.join(ROOT, name), *TAG)
                x = t * (x - x[0]) / (x[-1] - x[0]) + offset
                x += (x[1] - x[0])
                offset = x[-1]
                x_data = np.concatenate((x_data, x))
                y_data = np.concatenate((y_data, y))
            if save_offset:
                Creader.offset = offset
        else:
            x_data, y_data = bibarch.read_histo([os.path.join(ROOT, name) for name in names], *TAG,
                                                use_time=MODE == 'TIME')
        return x_data, y_data

    @staticmethod
    def reset_offset(names):
        names = names if isinstance(names, (list, tuple)) else [names]

        Creader.offset = 0
        for name in names:
            t = utils.fetch_slurm_stats(*utils.fetch_file(os.path.join(os.path.join(ROOT, name), 'slurm.*.out')))[0]
            x, y = bibarch.read_histo(os.path.join(ROOT, name), *TAG)
            x = t * (x - x[0]) / (x[-1] - x[0]) + Creader.offset
            x += (x[1] - x[0])
            Creader.offset = x[-1]


def cread(*args, **kwargs):
    return Creader()(*args, **kwargs)


if __name__ == '__main__':
    fig = plt.figure(figsize=[12, 9.41 * 2 / 3])
    ax = fig.add_subplot(111)
    ax.grid(True)

    # ax.plot(*cread("BASE_NS/RUN_MTP"), label='NS MTP', lw=3)
    # ax.plot(*cread("BASE_NS/RUN_KEX"), label='NS KEX', lw=3)

    ax.set_xlabel({'ITER': 'Iteration number', 'TIME': 'Simulation time', 'WALL': 'Wall time'}[MODE])
    ax.set_title(' '.join(TAG))
    ax.legend()
    plt.show()
