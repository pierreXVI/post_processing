import collections
import os.path
import subprocess

import numpy as np

import utils

UCHAR = 'B'
DOUBLE = '>d'
INT = '>i'


class BibarchReader:

    def __init__(self, filename):
        self.chapter_readers = {
            'VERSION': self._ch_version,
            'TITRE': self._ch_title,
            'GLOBAL': self._ch_global,
            'THERMODYNAMIQUE': self._ch_thermo,
            'ESPECES': self._ch_specie,
            'DOMUTIL': self._ch_dom,
            'STRUCTURE': self._ch_structure,
            'NUMEROTATION': self._ch_numbering,
            'CONNEXION': self._ch_connection,
            'VARIABLES': self._ch_variable,
            'MAILLAGE': self._ch_mesh,
            'GRAVITE CELLULE': self._ch_cell_g,
            'GRAVITE FACE': self._ch_face_g,
            'CELLULES': self._ch_cells,
            'SURFACE': self._ch_surface,
            'VALEUR VOLUMIQUE': self._ch_sca_vol,
            'VALEUR SURFACIQUE': self._ch_sca_surf,
            'VECTEUR SURFACIQUE': self._ch_vec_surf,
            'VECTEUR VOLUMIQUE': self._ch_vec_vol,
            'TENSEUR VOLUMIQUE': self._ch_tens_vol,
        }

        self._data = collections.defaultdict(
            lambda: collections.defaultdict(
                lambda: collections.defaultdict(
                    lambda: np.array([])
                )
            )
        )

        # self._offset = collections.defaultdict(
        #     lambda: collections.defaultdict(
        #         lambda: collections.defaultdict(
        #             int
        #         )
        #     )
        # )

        self._filename = filename
        self._ff = self._nff = self._save_nff = None

        self.nblocks = int(subprocess.check_output(['grep', '-c', 'VARIABLES', self._filename]))
        self._i_variable = -1
        self._count = utils.Counter(self.nblocks, 'Merging')

    def _ch_version(self):
        self._read_val(UCHAR)  # version

    def _ch_title(self):
        self._read_str()  # title
        self._read_str()  # date
        self._read_str()  # arch
        self._read_val(DOUBLE)  # t0
        self._read_val(DOUBLE)  # eps

    def _ch_global(self):
        self._read_str()  # solver
        self._read_val(UCHAR)  # dim

    def _ch_thermo(self):
        flag = self._read_val(INT)  # flag thermodynamic values
        self._read_str()  # filename
        if flag == 0:
            ge = self._read_val(INT)  # ge
            gr = self._read_val(INT)  # gr
            e = self._read_val(INT)  # e
            r = self._read_val(INT)  # r
            self._read_val(INT, n=ge)
            self._read_val(INT, n=gr)
            self._read_val(INT, n=e)
            self._read_val(DOUBLE, n=r)

    def _ch_specie(self):
        self._read_val(INT)  # mix number
        self._read_str()  # mix name
        n_elem = self._read_val(INT)  # n elements
        self._read_str(n=n_elem)  # elements names

    def _ch_dom(self):
        self._read_val(INT)  # idom
        self._read_str()  # name

    def _ch_structure(self):
        self._read_val(INT)  # i_dom abs
        self._read_val(INT)  # i_dom user
        self._read_str()  # dom name
        self._read_val(UCHAR)  # mesh type
        self._read_val(UCHAR)  # values location (0: nodes, 1: centers)
        self._read_val(INT)  # dom spicies mix
        self._read_val(INT)  # dom scalars

    def _ch_numbering(self):
        self._read_val(INT)  # idom
        self._read_val(DOUBLE)  # time
        n_v = self._read_val(INT)  # n vertices
        n_f = self._read_val(INT)  # n faces
        n_bf = self._read_val(INT)  # n boundary faces
        n_of = self._read_val(INT)  # n overlap faces
        n_c = self._read_val(INT)  # n cells
        self._read_val(INT, n=n_v)  # i vertices
        self._read_val(INT, n=n_f)  # i faces
        self._read_val(INT, n=n_c + n_bf + n_of)  # i cells

    def _ch_connection(self):
        self._read_val(INT)  # idom
        self._read_val(DOUBLE)  # time
        self._read_val(INT)  # f_min
        nfaces = self._read_val(INT)  # n faces
        self._read_val(INT, n=2 * nfaces)  # cell numbers
        nbpoints = self._read_val(UCHAR, n=nfaces)  # number of vertices
        self._read_val(INT, n=sum(nbpoints))  # vertice numbers

    def _ch_variable(self):
        self._i_variable += 1

        self._read_val(INT)  # idom
        nvar = self._read_val(INT)  # n variables
        names = self._read_str(n=nvar)  # variable names
        group = self._read_str()  # group
        category = self._read_str()  # category
        self._read_val(UCHAR)  # global
        self._read_val(INT)  # elem min
        nelem = self._read_val(INT)  # n elem
        values = self._read_bloc_real(nvar * nelem).reshape((nvar, nelem))

        for i in range(nvar):
            # foo = self._offset[category][group][names[i]]
            # if len(self[category, group, names[i]]) < foo + nelem:
            #     if names[0] == 'Temps':
            #         print(2 * len(self[category, group, names[i]]) + nelem)
            #     buffer = np.zeros(2 * len(self[category, group, names[i]]) + nelem)
            #     buffer[:len(self[category, group, names[i]])] = self[category, group, names[i]]
            #     self[category, group, names[i]] = buffer

            # self[category, group, names[i]][foo:foo + nvar] = values[i]
            # self._offset[category][group][names[i]] = foo + nvar
            self[category, group, names[i]] = np.concatenate((self[category, group, names[i]], values[i]))

    def _ch_mesh(self):
        self._read_val(INT)  # idom
        self._read_val(DOUBLE)  # time
        self._read_val(INT)  # v_min
        n_v = self._read_val(INT)  # n vertices
        self._read_bloc_real(n_v)  # x
        self._read_bloc_real(n_v)  # y
        self._read_bloc_real(n_v)  # z

    def _ch_cell_g(self):
        self._read_val(INT)  # idom
        self._read_val(DOUBLE)  # time
        self._read_val(INT)  # c_min
        n_c = self._read_val(INT)  # n cells
        self._read_bloc_real(n_c)  # x
        self._read_bloc_real(n_c)  # y
        self._read_bloc_real(n_c)  # z

    def _ch_face_g(self):
        self._read_val(INT)  # idom
        self._read_val(DOUBLE)  # time
        self._read_val(INT)  # f_min
        n_f = self._read_val(INT)  # n faces
        self._read_bloc_real(n_f)  # x
        self._read_bloc_real(n_f)  # y
        self._read_bloc_real(n_f)  # z

    def _ch_cells(self):
        self._read_val(INT)  # idom
        self._read_val(DOUBLE)  # time
        self._read_val(INT)  # f_min
        n_c = self._read_val(INT)  # n cells
        self._read_val(UCHAR, n=n_c)  # cell types

    def _ch_surface(self):
        self._read_val(INT)  # idom
        self._read_val(DOUBLE)  # time
        self._read_val(INT)  # s_min
        self._read_str()  # name
        self._read_val(UCHAR)  # type
        self._read_val(INT)  # elem min
        nelem = self._read_val(INT)  # n elem
        self._read_val(INT, n=nelem)  # n face
        self._read_val(UCHAR, n=nelem)  # type face

    def _ch_sca_vol(self):
        self._read_val(UCHAR)  # moyen
        self._read_val(INT)  # idom
        self._read_val(DOUBLE)  # time
        self._read_val(DOUBLE)  # dt
        self._read_val(UCHAR)  # class
        self._read_str()  # name
        self._read_val(INT)  # elem min
        nelem = self._read_val(INT)  # n elem
        self._read_bloc_real(nelem)  # scalar

    def _ch_sca_surf(self):
        self._read_val(UCHAR)  # moyen
        self._read_val(INT)  # idom
        self._read_val(DOUBLE)  # time
        self._read_val(DOUBLE)  # dt
        self._read_val(UCHAR)  # class
        self._read_str()  # name
        self._read_val(INT)  # surface
        self._read_val(INT)  # elem min
        nelem = self._read_val(INT)  # n elem
        self._read_bloc_real(nelem)  # scalar

    def _ch_vec_vol(self):
        self._read_val(UCHAR)  # moyen
        self._read_val(INT)  # idom
        self._read_val(DOUBLE)  # time
        self._read_val(DOUBLE)  # dt
        self._read_val(UCHAR)  # class
        self._read_str()  # name
        self._read_val(INT)  # elem min
        nelem = self._read_val(INT)  # n elem
        self._read_bloc_real(nelem)  # v_x
        self._read_bloc_real(nelem)  # v_y
        self._read_bloc_real(nelem)  # v_z

    def _ch_vec_surf(self):
        self._read_val(UCHAR)  # moyen
        self._read_val(INT)  # idom
        self._read_val(DOUBLE)  # time
        self._read_val(DOUBLE)  # dt
        self._read_val(UCHAR)  # class
        self._read_str()  # name
        self._read_val(INT)  # surface
        self._read_val(INT)  # elem min
        nelem = self._read_val(INT)  # n elem
        self._read_bloc_real(nelem)  # v_x
        self._read_bloc_real(nelem)  # v_y
        self._read_bloc_real(nelem)  # v_z

    def _ch_tens_vol(self):
        self._read_val(UCHAR)  # moyen
        self._read_val(INT)  # idom
        self._read_val(DOUBLE)  # time
        self._read_val(DOUBLE)  # dt
        self._read_val(UCHAR)  # class
        self._read_str()  # name
        self._read_val(INT)  # elem min
        nelem = self._read_val(INT)  # n elem
        self._read_bloc_real(nelem)  # t_xx
        self._read_bloc_real(nelem)  # t_yx
        self._read_bloc_real(nelem)  # t_zx
        self._read_bloc_real(nelem)  # t_xy
        self._read_bloc_real(nelem)  # t_yy
        self._read_bloc_real(nelem)  # t_zy
        self._read_bloc_real(nelem)  # t_xz
        self._read_bloc_real(nelem)  # t_yz
        self._read_bloc_real(nelem)  # t_zz

    def read(self, verbose=False):
        self._ff = open(self._filename, 'rb')
        while True:
            chapter = self._read_str()
            if verbose:
                print(chapter)
            if chapter == 'FIN_DU_FICHIER':
                break
            elif chapter == '':
                break
            self.chapter_readers[chapter]()
        self._ff.close()

    def read_and_merge_variable(self, verbose=False):
        self._ff = open(self._filename, 'rb')
        self._nff = self._save_nff = open(self._filename + '.TMP', 'wb')
        while True:
            chapter = self._read_str()
            if verbose:
                print(chapter)
            else:
                self._count(self._i_variable)
            if chapter == 'VARIABLES':
                self._nff = None
            elif self._nff is None:
                self._nff = self._save_nff
                first = True
                name = ''
                for category in self:
                    for group in self[category]:
                        if first:
                            first = False
                        else:
                            self._nff.write(b'VARIABLES\x00')
                        self._nff.write(np.array([0], dtype=INT).tobytes())  # idom
                        self._nff.write(np.array([len(self[category, group])], dtype=INT).tobytes())  # n variables
                        for name in self[category][group]:
                            self._nff.write(name.encode() + b'\x00')  # variable names
                        self._nff.write(group.encode() + b'\x00')  # group
                        self._nff.write(category.encode() + b'\x00')  # group
                        self._nff.write(np.array([0], dtype=UCHAR).tobytes())  # global
                        self._nff.write(np.array([0], dtype=INT).tobytes())  # elem min
                        self._nff.write(np.array(len(self[category, group, name]), dtype=INT).tobytes())  # n elem
                        self._nff.write(np.array([0], dtype=UCHAR).tobytes())  # compression
                        for name in self[category, group]:
                            # noinspection PyUnresolvedReferences
                            self._nff.write(self[category, group, name].astype(DOUBLE).tobytes())
                self._nff.write(chapter.encode() + b'\x00')
            if chapter == 'FIN_DU_FICHIER' or chapter == '':
                break
            self.chapter_readers[chapter]()
        self._nff.write(b'FINFINFIN\x00')
        self._nff.close()
        self._ff.close()
        subprocess.run(['mv', self._filename + '.TMP', self._filename])

    def _read_bloc_real(self, n):
        compression = self._read_val(UCHAR)
        if compression == 4:  # Constant
            return self._read_val(DOUBLE) * np.ones((n,))
        elif compression == 1:  # None
            v_min = self._read_val(DOUBLE)
            v_max = self._read_val(DOUBLE)
            return v_min + self._read_val('>I', n=n) * (v_max - v_min) / np.iinfo('>I').max
        elif compression == 0:  # None
            return self._read_val(DOUBLE, n=n)
        else:
            raise ValueError("Compression mode not implemented yet: {0}".format(compression))

    def _read_val(self, dtype, n=1):
        data = self._ff.read(n * np.dtype(dtype).itemsize)
        if self._nff:
            self._nff.write(data)
        if n == 1:
            return np.frombuffer(data, dtype=dtype)[0]
        return np.frombuffer(data, dtype=dtype, count=n)

    def _read_str(self, n=0):
        buffer = b''
        while True:
            b = self._ff.read(1)
            if b == b'' or b == b'\x00':
                break
            buffer += b
        if self._nff:
            self._nff.write(buffer + b)
        if n == 0:
            return buffer.decode()
        elif n == 1:
            return [buffer.decode()]
        else:
            return [buffer.decode(), *self._read_str(n=n - 1)]

    def __getitem__(self, items):
        items = items if isinstance(items, (list, tuple)) else [items]
        out = self._data
        for item in items:
            out = out[item]
        return out

    def __setitem__(self, items, value):
        items = items if isinstance(items, (list, tuple)) else [items]
        out = self._data
        for item in items[:-1]:
            out = out[item]
        out[items[-1]] = value

    def __iter__(self):
        return self._data.__iter__()

    def inspect_data(self, tab='    '):
        for i in self:
            print(i)
            for j in self[i]:
                print(tab, j)
                print(tab, tab, *[repr(k) for k in self[i, j]])


def read_histo(runs, category, group, name, use_time=False):
    """
    Read a set of "Explore's historique" files and return the expected data

    :param str, list[str] runs: path to the files, may be remote
    :param category:
    :param group:
    :param name:
    :param bool use_time:
    :return: the x and y data, and the restart indices
    """
    x_data = np.array([])
    y_data = np.array([])
    restarts = []
    g_x, n_x = ('DATES', 'Temps') if use_time else ('CYCLES', 'Ncycle')
    runs = runs if isinstance(runs, (list, tuple)) else [runs]

    for r in runs:
        f, = utils.fetch_file(os.path.join(r, 'histo_CHARME.0'))
        bibarch_reader = BibarchReader(f)
        if bibarch_reader.nblocks > 100:
            bibarch_reader.read_and_merge_variable()
        else:
            bibarch_reader.read()
        x_data = np.concatenate((x_data, bibarch_reader['HORLOGE', g_x, n_x]))
        y_data = np.concatenate((y_data, bibarch_reader[category, group, name]))
        restarts.append(len(x_data))
    return x_data, y_data, restarts[:-1]
