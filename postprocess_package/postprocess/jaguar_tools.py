import collections
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
