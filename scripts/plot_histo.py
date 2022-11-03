import matplotlib.pyplot as plt

from postprocess import bibarch

if __name__ == '__main__':
    fig = plt.figure(figsize=[12, 9.41 * 2 / 3])
    ax = fig.add_subplot(111)
    ax.grid(True)

    tags = ('RESIDUS', 'MOYENS', 'RhoV_x')
    p = bibarch.HistoPlotter(ax, tags=tags, root="/scratchm/pseize/RAE_2822_FINE")

    p.plot("BASE/INIT", 'k', label="Init")
    p.reset_offset("BASE/INIT")
    p.plot(["BASE/RUN_1", "BASE/RUN_2", "BASE/RUN_3"], lw=3, label="Base")
    p.plot("MF/RUN_1", lw=3, label="MF")

    ax.legend()
    plt.show()
