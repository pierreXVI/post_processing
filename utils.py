import glob
import os
import shutil
import subprocess

import dateutil.parser

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
            if not subprocess.run(['scp', '{0}:{1}'.format(h, filename), tmp_root],
                                  stderr=subprocess.DEVNULL).returncode:
                found = True
                break
        if not found:
            print('\r', end='', flush=True)
            raise FileNotFoundError("Cannot find {0} on {1} or {2}"
                                    .format(filename, ', '.join(('localhost', *config.HOSTS[:-1])), config.HOSTS[-1]))
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
