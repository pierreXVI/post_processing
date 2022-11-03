from postprocess import pvlib

if __name__ == '__main__':
    plotter = pvlib.CpPlotter()
    plotter.register_plot("/scratchm/pseize/RAE_2822/BASE/RUN_1/ENSIGHT/archive_CHARME.surf.ins.case",
                          p_inf=26500, gamma=1.4, mach=0.75, block_name=['Intrados', 'Extrados'], label='Base')
    plotter.register_plot("/scratchm/pseize/RAE_2822/MF/RUN_1/ENSIGHT/archive_CHARME.surf.ins.case", marker=1,
                          p_inf=26500, gamma=1.4, mach=0.75, block_name=['Intrados', 'Extrados'], label='MF')

    plotter.draw(duration=0, block=True)

    # import paraview.simple as pvs
    # pvs.SaveState("/d/pseize/pythonstate.pvsm")
