import numpy as np

import pvlib
import os

if __name__ == '__main__':
    def cpath(name, volu=True):
        root = "/scratchm/pseize/SPHERE_LOBB"
        # root = "/scratchm/pseize/CYLINDRE_LOBB"
        tail_volu = "_ENSIGHT_/archive_CHARME.volu.ins.case"
        tail_surf = "_ENSIGHT_/archive_CHARME.surf.ins.case"
        return os.path.join(root, name, tail_volu if volu else tail_surf)


    plotter = pvlib.SpherePlotter(view_size=(600, 600), time=False, use_angle=True,
                                  ray_array="P", surface_array='P')

    # plotter.register_plot(cpath("JFNK_NS/RUN_KEX", True), 'P', view_size=(800, 800))
    plotter.register_surface(cpath("BASE_NS/RUN_KEX", False), label='Classical method', ms=1)
    plotter.register_surface(cpath("JFNK_NS/RUN_KEX", False), label='JFNK method', ms=2)
    plotter._current_color = 0
    plotter.register_ray(cpath("BASE_NS/RUN_KEX", True), [0,], ms=1, label='Classical method')
    plotter.register_ray(cpath("JFNK_NS/RUN_KEX", True), [0,], ms=2, ls=1, label='JFNK method')

    plotter.draw(duration=0, block=True)

    # plotter.compute_shock(cpath("BASE_NS_AMR/JOB_8232/1", True), 'P')

    # import paraview.simple as pvs
    # pvs.SaveState("/d/pseize/pythonstate.pvsm")
