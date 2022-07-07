import os

import pvlib

if __name__ == '__main__':
    def cpath(name, volu=True):
        root = "/scratchm/pseize/SPHERE_LOBB"
        # root = "/scratchm/pseize/CYLINDRE_LOBB"
        tail_volu = "_ENSIGHT_/archive_CHARME.volu.ins.case"
        tail_surf = "_ENSIGHT_/archive_CHARME.surf.ins.case"
        return os.path.join(root, name, tail_volu if volu else tail_surf)


    plotter = pvlib.SpherePlotter(view_size=(600, 600), time=False, use_angle=True,
                                  ray_array="P", surface_array='Flux~de~chaleur')

    plotter.draw(duration=10, block=True)

    # import paraview.simple as pvs
    # pvs.SaveState("/d/pseize/pythonstate.pvsm")
