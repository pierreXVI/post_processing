import glob
import os
import shutil
import subprocess

import dateutil.parser
import matplotlib.pyplot as plt
import numpy as np

import config


def fetch_file(filename):
    if glob.glob(filename):
        return glob.glob(filename)

    tmp_filename = os.path.normpath(config.TMP_DIR + os.path.sep + filename)
    if not glob.glob(tmp_filename):
        tmp_root = os.path.dirname(tmp_filename)
        os.makedirs(tmp_root, exist_ok=True)
        found = False
        for h in config.HOSTS:
            print("\rLoading {0} from {1}".format(filename, h), end='', flush=True)
            if not subprocess.run(['scp', '-r', '{0}:{1}'.format(h, filename), tmp_root],
                                  stderr=subprocess.DEVNULL).returncode:
                found = True
                break
        if not found:
            print('\r', end='', flush=True)
            raise FileNotFoundError("Cannot find {0} on {1} or {2}"
                                    .format(filename, ', '.join(config.HOSTS), 'localhost'))
        print()

    return glob.glob(tmp_filename)


def insert_repeat(collection_in, *args):
    """ Insert items in args after each item in collection_in """
    list_out = []
    for obj in collection_in:
        list_out.append(obj)
        for item in args:
            list_out.append(item)
    return list_out


def fetch_slurm_stats(slurm_out):
    """ Returns total run time in seconds and node list """
    with open(slurm_out) as file:
        for line in file:
            if "SLURM_JOB_NODELIST" in line:
                node_list = line.split()[2]
            elif 'NodeList' in line:
                node_list = line.split('=')[1]
            elif "Job started at" in line:
                start = dateutil.parser.parse(line.split()[3])
            elif "StartTime" in line:
                start = dateutil.parser.parse(line.split('=')[1])
            elif "Job ended at" in line:
                end = dateutil.parser.parse(line.split()[3])
            elif "EndTime" in line:
                end = dateutil.parser.parse(line.split('=')[1])
    return (end - start).total_seconds(), node_list


def fetch_suivi_stats(suivi):
    """ Returns total run time in seconds and node list """
    with open(suivi) as file:
        for line in file:
            if "cedre.f90 : CALCUL cedre (SORTIE)" in line:
                return float(line.split('=')[-1])


class Counter:
    """
    Class used to display a progress bar. Rewrites on the same line to get a moving progress bar.
    Prints only if there is a difference with the previous print, not to slow the bigger process.
    The counter can be called, with an integer ``i``. ``i == -1`` correspond to 0% and ``i == n - 1`` to 100%.
    :param str text: The text of the counter
    :param int n: The final value
    Example:
       >>> n_max = 400
       >>> c = Counter(n_max, 'Text')
       >>> for i in range(-1, n_max):
       ...     c(i)  # Will display a new message only when i = -1, 3, 7, ..., 399 in this example
       >>>
    """

    def __init__(self, n, text=''):
        self.text = text
        self.n = n
        self.i = -1
        if self.text:
            self.string = '\r{0} - |{{0:<{{bar_length}}}}| {{1:=4.0%}}'.format(self.text)
        else:
            self.string = '\r|{0:<{bar_length}}| {1:=4.0%}'

    def __call__(self, i):
        i += 1
        if self.n == 0 or i > self.n or 100 * self.i // self.n == 100 * i // self.n:
            return
        self.i = i
        try:
            bar_length = shutil.get_terminal_size().columns + 23 - len(self.string)
        except AttributeError:
            bar_length = 10
        print(self.string.format(int(bar_length * i / self.n) * '-', i / self.n, bar_length=bar_length), end='')
        if i == self.n:
            print()


def annotate_slope(axis, s, base=0.2, dx=0.0, dy=0.0, transpose=False, pad=None):
    """

    :param plt.Axis axis:
    :param float s: slope value
    :param float base:
    :param float dx:
    :param float dy:
    :param bool transpose: if True, flip the triangle
    :param float pad: text padding
    """
    xm, xp, ym, yp = np.inf, -np.inf, np.inf, -np.inf
    for line in axis.get_lines():
        if line.get_xdata().size and line.get_ydata().size:
            xm = min(xm, np.min(line.get_xdata()))
            xp = max(xp, np.max(line.get_xdata()))
            ym = min(ym, np.min(line.get_ydata()))
            yp = max(yp, np.max(line.get_ydata()))

    line_x = np.array([np.power(xm, base) * np.power(xp, 1 - base), xp])
    line_y = np.array([ym, ym * np.power(line_x[1] / line_x[0], s)])
    if dx:
        line_x *= np.power(xm / xp, dx * (1 - base))
    if dy:
        line_y *= np.power(yp / ym, dy * (1 - base * s * np.log(xp / xm) / np.log(yp / ym)))

    line, = axis.plot(line_x, line_y, 'k')
    if not pad:
        pad = line.get_linewidth() / 2 + min(max(2, line.get_linewidth() / 2), plt.rcParams['font.size'])
    if not transpose:
        axis.plot([line_x[0], line_x[1], line_x[1]], [line_y[0], line_y[0], line_y[1]], 'k-.')
        axis.annotate(1, xy=(np.sqrt(line_x[0] * line_x[1]), line_y[0]),
                      ha='center', va='top', xytext=(0, -pad), textcoords='offset points')
        axis.annotate(s, xy=(line_x[1], np.sqrt(line_y[0] * line_y[1])),
                      ha='left', va='center', xytext=(pad, 0), textcoords='offset points')
    else:
        axis.plot([line_x[0], line_x[0], line_x[1]], [line_y[0], line_y[1], line_y[1]], 'k-.')
        axis.annotate(1, xy=(np.sqrt(line_x[0] * line_x[1]), line_y[1]),
                      ha='center', va='bottom', xytext=(0, pad), textcoords='offset points')
        axis.annotate(s, xy=(line_x[0], np.sqrt(line_y[0] * line_y[1])),
                      ha='right', va='center', xytext=(-pad, 0), textcoords='offset points')
