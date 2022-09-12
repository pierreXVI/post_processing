import os.path

import matplotlib.pyplot as plt
import numpy as np

import bibarch
import utils

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
