import collections
import importlib.resources
import os

import numpy as np

from . import utils


def read_error(case_path):
    """
    Read the L2 error from the case slurm.*.out file

    :param str case_path:
    :return: the error at each period for each variable for each error type
    :rtype: dict
    """
    data = {key: collections.defaultdict(list) for key in [
        "Discrete L2 error :",
        "Discrete L2 error scaled by the jacobian :",
        "Integral L2 error :",
        "LInfinite error :"
    ]}

    with open(utils.fetch_file(os.path.join(case_path, 'slurm.*.out'))[0]) as file:
        target = None
        for line in file:
            if target is None:
                for d in data:
                    if d in line:
                        target = data[d]
                        break
            else:
                try:
                    tag, val = line[:-1].split(':')
                    target[tag.strip()].append(float(val))
                except ValueError:
                    target = None

    out = {}
    for d in data:
        out[d[:-2]] = {}
        for tag in data[d]:
            out[d[:-2]][tag] = np.array(data[d][tag])
    return out


def read_integral_quantities(case_path):
    """
    Read the `sol_IntegralQuantities*.dat` files from the case

    :param str case_path:
    :return: the values for each saved quantity
    :rtype: dict
    """
    data = collections.defaultdict(lambda: np.ndarray((0,)))
    files = sorted(utils.fetch_file(os.path.join(case_path, 'sol_IntegralQuantities*.dat')))
    for file in files:
        with open(file) as dat_file:
            labels = [label for label in dat_file.readline().split('"') if label.strip()]
        file_data = np.loadtxt(file, skiprows=1).T
        file_data = file_data[:, :-1]  # TODO: last saved value seems wrong

        # TODO: Jaguar saves empty entries with time=0 at the end
        if file == files[-1]:
            i, j = 0, file_data.shape[1] - 1
            while file_data[0, j] == 0:
                if i == j - 1:
                    break
                k = i + (j - i) // 2
                if file_data[0, k] == 0:
                    j = k
                else:
                    i = k
            file_data = file_data[:, :j]

        for i, label in enumerate(labels):
            data[label] = np.concatenate((data[label], file_data[i]))
    return data


def ref_taylor_green_vortex(ref_file=''):
    """
    Returns reference data for the Taylor-Green vortex

    :param str ref_file: defaults to '../data/refdata_TGV.dat'
    :return:
    :rtype: dict
    """
    if not ref_file:
        ref_file = importlib.resources.files(__package__).joinpath('../data/refdata_TGV.dat')
    with open(ref_file) as dat_file:
        labels = [label for label in dat_file.readline().split('"') if label.strip()]
    data = np.loadtxt(ref_file, skiprows=1).T
    return {label: data[i] for i, label in enumerate(labels)}
