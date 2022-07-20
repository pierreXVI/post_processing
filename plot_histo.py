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


    fig = plt.figure(figsize=[12, 9.41 * 2 / 3])
    ax = fig.add_subplot(111)
    ax.grid(True)

    # ax.plot(*cread("BASE_NS/RUN_MTP"), label='NS MTP', lw=3)
    # ax.plot(*cread("BASE_NS/RUN_KEX"), label='NS KEX', lw=3)

    ax.legend()
    plt.show()
