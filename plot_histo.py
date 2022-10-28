import os.path

import matplotlib.pyplot as plt

import bibarch
import utils


class HPlotter:

    def __init__(self, axis, tags, mode='ITER', root=''):
        """

        :param matplotlib.axis._axis.Axis axis:
        :param str mode: 'ITER', 'TIME' or 'WALL'
        :param str root: path to prepend to case files
        :param tuple(str, str, str) tags:
        """

        self._tags = tags
        self._mode = mode
        self._root = root

        self._ax = axis
        self._ax.set_xlabel({'ITER': 'Iteration number', 'TIME': 'Simulation time', 'WALL': 'Wall time'}[self._mode])
        self._ax.set_title(' '.join(self._tags))

        self._offset = 0

    def get_wall_time(self, name):
        file, = utils.fetch_file(os.path.join(os.path.join(self._root, name), 'suivi.1'))
        t = utils.fetch_suivi_stats(file)
        if t is None:
            file, = utils.fetch_file(os.path.join(os.path.join(self._root, name), 'slurm.*.out'))
            t = utils.fetch_slurm_stats(file)[0]
        return t

    def get(self, names):
        names = names if isinstance(names, (list, tuple)) else [names]
        x_data, y_data, restarts = bibarch.read_histo([os.path.join(self._root, name) for name in names], *self._tags,
                                                      use_time=self._mode == 'TIME')
        if self._mode == 'WALL':
            lengths = [0, *restarts, len(x_data)]
            offset = self._offset
            for i in range(len(names)):
                t = self.get_wall_time(names[i])
                x = x_data[lengths[i]:lengths[i + 1]]
                x_data[lengths[i]:lengths[i + 1]] = t * (x + x[1] - 2 * x[0]) / (x[-1] + x[1] - 2 * x[0]) + offset
                offset = x_data[lengths[i + 1] - 1]
        return x_data, y_data, restarts

    def reset_offset(self, names):
        if self._mode != 'WALL':
            return
        names = names if isinstance(names, (list, tuple)) else [names]
        self._offset = sum(self.get_wall_time(name) for name in names)

    def plot(self, case, *args, **kwargs):
        x, y, restarts = self.get(case)
        self._ax.plot(x, y, *args, **kwargs)
        self._ax.plot(x[restarts], y[restarts], 'k|', ms=10, mew=3)


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
