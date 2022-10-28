import os

import numpy as np
import paraview.simple as pvs

import utils

DEFAULT_COLORS = ['1f77b4', 'ff7f0e', '2ca02c', 'd62728', '9467bd', '8c564b', 'e377c2', '7f7f7f', 'bcbd22', '17becf']


def _fetch_ensight_data(case_filename, arrays):
    out, = utils.fetch_file(case_filename)

    root, filename = os.path.split(case_filename)
    base_name = os.path.splitext(filename)[0]
    geom_name = os.path.splitext(base_name)[0] + os.path.extsep + 'geom'
    for file in [geom_name, *[base_name + os.path.extsep + array for array in arrays]]:
        utils.fetch_file(os.path.join(root, file))
    return out


def fetch_ensight_case(case_filename):
    """
    Fetches full Ensight case from multiple available hosts

    :param str case_filename: requested case path
    :return: actual case path
    :rtype: str
    """
    out = _fetch_ensight_data(case_filename, [])
    return _fetch_ensight_data(case_filename, pvs.EnSightReader(CaseFileName=out).CellArrays)


class Plotter:
    def __init__(self):
        self.data = {}
        self._current_color = 0

    def load_data(self, filename, cell_arrays, render_view=False, line_view=False, rvs=(400, 400), lvs=(400, 400)):
        """
        Loads data from an Ensight case, reuses the case render and line views if they already exist

        :param str filename: requested case path
        :param list cell_arrays: cell arrays to load
        :param bool render_view: flag to ask for a render view
        :param bool line_view: flag to ask for a line view
        :param tuple[int, int] rvs: render view size
        :param tuple[int, int] lvs: line view size
        :return:
        :rtype: tuple[str, Any, Any]
        """
        filename = _fetch_ensight_data(filename, cell_arrays)
        if filename not in self.data:
            reader, r_view, l_view = (pvs.OpenDataFile(filename), None, None)
            reader.CellArrays = cell_arrays
        else:
            reader, r_view, l_view = self.data[filename]
            reader.CellArrays = list(set(reader.CellArrays).union(set(cell_arrays)))

        if render_view and r_view is None:
            r_view = self.create_view(pvs.CreateRenderView, InteractionMode='2D', ViewSize=rvs)
        if line_view and l_view is None:
            l_view = self.create_view(pvs.CreateXYPlotView, ViewSize=lvs)

        self.data[filename] = reader, r_view, l_view
        return self.data[filename]

    def draw(self, duration, block=False):
        """
        Draws all views associated with the Plotter object

        :param float duration: duration in seconds of the animation
        :param bool block: if True, block at the end of the animation
        """
        for filename in self.data:
            if self.data[filename][1] is not None:
                self.data[filename][1].ResetCamera()

        animation_scene = pvs.GetAnimationScene()
        animation_scene.UpdateAnimationUsingDataTimeSteps()
        animation_scene.PlayMode = 'Real Time'
        animation_scene.Duration = duration
        animation_scene.Play()
        if block:
            pvs.Interact()

    @staticmethod
    def scale_display(display):
        """
        Scales a display to fit value range from the last time step

        :param display:
        """
        animation_scene = pvs.GetAnimationScene()
        animation_scene.UpdateAnimationUsingDataTimeSteps()
        animation_scene.GoToLast()
        display.RescaleTransferFunctionToDataRange(False, True)

    @staticmethod
    def annotate_time(inp, view, time, progress):
        """
        Annotates the times from the input on the view

        :param inp: input from where to get the times
        :param view: render view (can be a line view if progress is False)
        :param bool time: shows the time as text
        :param bool progress: shows the time as a progression bar
        """
        if time:
            if pvs.GetParaViewVersion().GetVersion() == 5.9:
                fmt = "Time: %g"
            else:
                fmt = "Time: {time:f}"
            pvs.Show(pvs.AnnotateTimeFilter(inp, Format=fmt), view, Interactivity=1)
        if progress:
            pvs.Show(pvs.TimeStepProgressBar(inp), view)

    @staticmethod
    def create_view(pv_creator, *args, **kwargs):
        """
        Creates a view with the given arguments and adds it to the layout

        :param pv_creator:
        :param args:
        :param kwargs:
        :return:
        """
        view = pv_creator(*args, **kwargs)
        pvs.AssignViewToLayout(view)
        return view

    def str_color(self, color=None):
        """
        Convert a color to a list of the three Paraview color coefficients

        :param str, None color: string corresponding to a color hexadecimal RGB code, 'FF0000' for red
        :rtype: list[str, str, str]
        """
        if color is None:
            color = DEFAULT_COLORS[self._current_color]
            self._current_color = (self._current_color + 1) % len(DEFAULT_COLORS)
        return [str(int(color[2 * i:2 * i + 2], base=16) / 255) for i in range(3)]

    @staticmethod
    def find_blocks(dataset, hints):
        """
        Find the blocks in a multiblock dataset

        :param dataset:
        :param str, list[str] hints: string or list of string, where each string is a substring of desired block names
        :return: a dictionary where the keys are the indices of the blocks and the values their names
        :rtype: dict
        """
        out = {}
        dataset.UpdatePipeline()
        hints = hints if isinstance(hints, (list, tuple)) else [hints]

        if pvs.GetParaViewVersion().GetVersion() == 5.9:
            info = dataset.GetDataInformation().GetCompositeDataInformation()
            for i in range(info.GetNumberOfChildren()):
                if any(h in info.GetName(i) for h in hints):
                    out[i+1] = info.GetName(i)
        else:
            info = dataset.GetDataInformation()
            for i in range(1, info.GetNumberOfDataSets() + 1):
                if any(h in info.GetBlockName(i) for h in hints):
                    out[i] = info.GetBlockName(i)

        if not out:
            raise ValueError("Cannot find block with {0}".format(', '.join(hints)))
        return out


class CpPlotter(Plotter):
    def __init__(self, view_size=(400, 400)):
        super().__init__()
        self.view = self.create_view(
            pvs.CreateXYPlotView, LeftAxisTitle='$C_p$', LeftAxisUseCustomRange=1, LeftAxisRangeMinimum=2,
            LeftAxisRangeMaximum=-1, BottomAxisTitle='$x$', BottomAxisUseCustomRange=1, BottomAxisRangeMinimum=-0.05,
            BottomAxisRangeMaximum=1.05, ViewSize=view_size
        )

    def register_plot(self, filename, p_inf, gamma, mach, label='Cp', color=None, marker=2):
        reader, _, _ = self.load_data(filename, ['P'])

        calc = pvs.Calculator(reader, AttributeType='Cell Data', ResultArrayName='Cp',
                              Function="((P / {0}) - 1) * 2 / ({1} * {2}^2)".format(p_inf, gamma, mach))
        c2p = pvs.CellDatatoPointData(calc, ProcessAllArrays=0, CellDataArraytoprocess=['Cp'])
        plot = pvs.PlotData(c2p)

        blocks = self.find_blocks(plot, ['Extrados', 'Intrados'])
        names = ['Cp ({0})'.format(blocks[i]) for i in blocks]
        labels = utils.insert_repeat(names, '')
        labels[1] = label
        pvs.Show(plot, self.view,
                 UseIndexForXAxis=0,
                 XArrayName='Points_X',
                 CompositeDataSetIndex=list(blocks),
                 SeriesVisibility=names,
                 SeriesLabel=labels,
                 SeriesColor=utils.insert_repeat(names, *self.str_color(color)),
                 SeriesLineStyle=utils.insert_repeat(names, '0'),
                 SeriesMarkerStyle=utils.insert_repeat(names, str(marker)),
                 SeriesLineThickness=utils.insert_repeat(names, '2'))


class SpherePlotter(Plotter):
    def __init__(self, view_size=(400, 400), time=False, progress=False, use_angle=True,
                 ray_array=None, surface_array=None, radius=0.00635):
        """
        Plot a 2D Lobb sphere case

        :param tuple(int, int) view_size: line view size used to display the ray and surfaces values
        :param bool time: flag to annotate the time as a text
        :param bool progress: flag to annotate the time as a progression bar
        :param bool use_angle: use the angle as x coordinate instead of the actual x coordinate on surface plots
        :param str ray_array: array used in "ray plots"
        :param str surface_array: array used in "surface plots"
        :param float radius: sphere radius
        """
        super().__init__()
        self.line_view = None
        self.surface_view = None
        self.view_size = view_size
        self.time = time
        self.progress = progress
        self.use_angle = use_angle
        self.ray = ray_array
        self.surface = surface_array
        self.r = radius

    def register_plot(self, filename, cell_array, component='', view_size=(400, 400), stream=False, shock=False):
        """
        Displays a field for the case

        :param str filename: case path
        :param str cell_array: value to display
        :param str component: value component for vector arrays
        :param tuple(int, int) view_size:
        :param bool stream: flag to show streamlines
        :param bool shock: flag to show the shock
        """
        reader, render_view, _ = self.load_data(filename, [cell_array, 'V'] if stream else [cell_array],
                                                render_view=True, rvs=view_size)
        reader_display = pvs.Show(reader, render_view, Representation='Surface', ColorArrayName=['CELLS', cell_array])
        if component:
            pvs.ColorBy(reader_display, ('CELLS', cell_array, component))
        reader_display.SetScalarBarVisibility(render_view, True)
        self.scale_display(reader_display)

        if stream:
            stream = pvs.StreamTracer(reader, SeedType='Line', MaximumStreamlineLength=0.01, Vectors=['CELLS', 'V'])
            stream.SeedType.Point1 = [-0.00635, 0.0, 0.0]
            stream.SeedType.Point2 = [-0.007, 0.001, 0.0]
            stream.SeedType.Resolution = stream if isinstance(stream, int) else 10
            pvs.Show(stream, render_view, Representation='Surface')

        self.annotate_time(reader, render_view, self.time, self.progress)

        if shock:
            coords = []
            c2p = pvs.CellDatatoPointData(reader, ProcessAllArrays=0, CellDataArraytoprocess=[cell_array])
            for theta in np.linspace(0, 90, 101):
                plot = pvs.PlotOnIntersectionCurves(c2p, SliceType='Plane')
                plot.SliceType.Normal = [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)), 0]
                plot.SliceType.Offset = 1e-6 if theta == 0 else -1e-6 if theta == 90 else 0
                if pvs.GetParaViewVersion().GetVersion() == 5.9:
                    grad = pvs.GradientOfUnstructuredDataSet(plot, ScalarArray=['POINTS', cell_array],
                                                             ResultArrayName="Gradients")
                else:
                    grad = pvs.Gradient(plot, ScalarArray=['POINTS', cell_array], ResultArrayName="Gradients")
                pvs.QuerySelect(QueryString='(mag(Gradients)  == max(mag(Gradients)))', FieldType='POINT')
                selection = pvs.ExtractSelection(grad)
                selection.UpdatePipeline()
                xmin, xmax, ymin, ymax, zmin, zmax = selection.GetDataInformation().DataInformation.GetBounds()
                coords += [xmax, ymax, zmax]
            spline = pvs.SplineSource(ParametricFunction='Spline')
            spline.ParametricFunction.Points = coords
            pvs.Show(spline, render_view)

    def compute_shock(self, filename, cell_array, out_file):
        reader, render_view, _ = self.load_data(filename, [cell_array], render_view=True)
        reader_display = pvs.Show(reader, render_view)
        self.scale_display(reader_display)

        coords = []
        c2p = pvs.CellDatatoPointData(reader, ProcessAllArrays=0, CellDataArraytoprocess=[cell_array])
        for theta in np.linspace(0, 90, 101):
            plot = pvs.PlotOnIntersectionCurves(c2p, SliceType='Plane')
            plot.SliceType.Normal = [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)), 0]
            plot.SliceType.Offset = 1e-6 if theta == 0 else -1e-6 if theta == 90 else 0
            grad = pvs.GradientOfUnstructuredDataSet(plot, ScalarArray=['POINTS', cell_array])
            pvs.QuerySelect(QueryString='(mag(Gradients)  == max(mag(Gradients)))', FieldType='POINT')
            selection = pvs.ExtractSelection(grad)
            selection.UpdatePipeline()
            xmin, xmax, ymin, ymax, zmin, zmax = selection.GetDataInformation().DataInformation.GetBounds()
            coords += [xmax, ymax]
        coords = np.reshape(np.array(coords), (101, 2))
        # FIXME
        coords[0, 0] = coords[1, 0]
        coords[0, 1] = 0
        coords[-1, 0] = 0
        np.save(out_file, coords)

    def register_ray(self, filename, angles, colors=None, label='', flag_grad=False, ls=0, marker=1):
        """
        Plot a value on a list of rays

        :param str filename: case path
        :param iterable[float] angles: list of angles in degrees where to put the rays
        :param list[str] colors: list of colors for each ray
        :param str label: label for the legend
        :param bool flag_grad: flag to plot the gradient along the ray
        :param int ls: line style (see Paraview)
        :param int marker: marker style (see Paraview)
        """

        if self.ray is None:
            raise ValueError("Cell array for ray was not set")
        if colors is None:
            colors = len(angles) * [None]
        if len(colors) < len(angles):
            raise ValueError("More slices than colors")

        reader, render_view, _ = self.load_data(filename, [self.ray])
        c2p = pvs.CellDatatoPointData(reader, ProcessAllArrays=0, CellDataArraytoprocess=[self.ray])

        for theta, color in zip(angles, colors):
            if not 0 <= theta <= 90:
                raise ValueError("Angle value outside [0, 90]")
            if self.line_view is None:
                self.line_view = self.create_view(pvs.CreateXYPlotView, LeftAxisTitle='${0}$'.format(self.ray),
                                                  BottomAxisTitle='$r$', ViewSize=self.view_size)
                self.annotate_time(reader, self.line_view, self.time, False)

            plot = pvs.PlotOnIntersectionCurves(c2p, SliceType='Plane')
            plot.SliceType.Normal = [np.sin(np.deg2rad(theta)), np.cos(np.deg2rad(theta)), 0]
            plot.SliceType.Offset = 1e-6 if theta == 0 else -1e-6 if theta == 90 else 0
            if render_view:
                pvs.Show(plot, render_view)
            label_grad = r'{2} $\nabla${0} $\quad\left(\theta = {1:g}^\circ\right)$'.format(self.ray, theta, label)
            label = r'{2} {0} $\quad\left(\theta = {1:g}^\circ\right)$'.format(self.ray, theta, label)
            grad = pvs.GradientOfUnstructuredDataSet(plot, ScalarArray=['POINTS', self.ray]) if flag_grad else plot
            pvs.Show(grad, self.line_view,
                     UseIndexForXAxis=0, XArrayName='Points_Magnitude',
                     SeriesVisibility=[self.ray + ' (2)', 'Gradients_Magnitude (2)'],
                     SeriesColor=[self.ray + ' (2)', *self.str_color(color),
                                  'Gradients_Magnitude (2)', *(self.str_color() if flag_grad else '')],
                     SeriesLabel=[self.ray + ' (2)', label, 'Gradients_Magnitude (2)', label_grad],
                     SeriesPlotCorner=['Gradients_Magnitude (2)', '1'],
                     SeriesLineStyle=[self.ray + ' (2)', str(ls)],
                     SeriesMarkerStyle=[self.ray + ' (2)', str(marker)],
                     SeriesMarkerSize=[self.ray + ' (2)', '10'])

    def register_surface(self, filename, color=None, ls=0, marker=2, label='', sphere_label='SPHERE'):
        """
        Plot a value on the sphere surface

        :param str filename: case path
        :param str color: marker color
        :param int ls: line style (see Paraview)
        :param int marker: marker style (see Paraview)
        :param str label: label for the legend
        :param str sphere_label: label name of the wall boundary condition
        """
        reader, _, _ = self.load_data(filename, [self.surface])
        c2p = pvs.CellDatatoPointData(reader, ProcessAllArrays=0, CellDataArraytoprocess=[self.surface])
        if self.use_angle:
            calc = pvs.Calculator(c2p, AttributeType='Point Data', ResultArrayName='theta',
                                  Function="asin(coordsY / sqrt(coordsX^2 + coordsY^2)) * 180 / 3.141592653589793")
            x_array_name = 'theta'
            bottom_axis_title = r'$\theta$'
        else:
            calc = c2p
            x_array_name = 'Points_X'
            bottom_axis_title = r'$x$'
        plot = pvs.PlotData(calc)

        if self.surface_view is None:
            self.surface_view = self.create_view(pvs.CreateXYPlotView, LeftAxisTitle='${0}$'.format(self.surface),
                                                 BottomAxisTitle=bottom_axis_title, ViewSize=self.view_size)
            self.annotate_time(reader, self.surface_view, self.time, False)

        block = self.find_blocks(plot, sphere_label)
        for i in block:
            name = '{0} ({1})'.format(self.surface, block[i])
            pvs.Show(plot, self.surface_view,
                     UseIndexForXAxis=0, XArrayName=x_array_name,
                     CompositeDataSetIndex=[i],
                     SeriesVisibility=[name],
                     SeriesLabel=[name, label],
                     SeriesColor=[name, *self.str_color(color)],
                     SeriesLineStyle=[name, str(ls)],
                     SeriesMarkerStyle=[name, str(marker)],
                     SeriesMarkerSize=[name, '10'])
            break


class DPWCpPlotter(Plotter):
    def __init__(self, line_view_size=(400, 400)):
        super().__init__()
        self.line_view_size = line_view_size
        self.line_view = None

    def register_plot(self, filename, p_inf, gamma, mach, slices, label='', colors=None, marker=2,
                      time=False, progress=False):
        reader, render_view, _ = self.load_data(filename, ['P'], render_view=True)

        blocs = list(self.find_blocks(reader, 'Wing'))
        select = pvs.ExtractBlock(reader, BlockIndices=blocs)
        select_display = pvs.Show(select, render_view, Representation='Surface', ColorArrayName=['CELLS', 'P'])
        select_display.SetScalarBarVisibility(render_view, True)
        self.scale_display(select_display)

        if colors is None:
            colors = len(slices) * [None]
        if len(colors) < len(slices):
            raise ValueError("More slices than colors")

        for (slice_o, slice_n), color in zip(slices, colors):
            slice_ = pvs.Slice(select)
            slice_.SliceType.Origin = slice_o
            slice_.SliceType.Normal = slice_n
            slice_.UpdatePipeline()
            pvs.Show(slice_, render_view, Representation='Surface', ColorArrayName=['CELLS', ''],
                     DiffuseColor=[1, 1, 1])
            bounds = slice_.GetDataInformation().GetBounds()
            calc = pvs.Calculator(slice_, AttributeType='Cell Data', ResultArrayName='Cp',
                                  Function="((P / {0}) - 1) * 2 / ({1} * {2}^2)".format(p_inf, gamma, mach))
            calc = pvs.Calculator(calc, AttributeType='Point Data', ResultArrayName='X/c',
                                  Function="(coordsX - {0}) / ({1} - {0})".format(bounds[0], bounds[1]))
            c2p = pvs.CellDatatoPointData(calc, ProcessAllArrays=0, CellDataArraytoprocess=['Cp'])
            plot = pvs.PlotData(c2p)
            plot.UpdatePipeline()

            if self.line_view is None:
                self.line_view = self.create_view(
                    pvs.CreateXYPlotView, LeftAxisTitle='$C_p$', LeftAxisUseCustomRange=1, LeftAxisRangeMinimum=2,
                    LeftAxisRangeMaximum=-1, BottomAxisTitle='$x/c$', BottomAxisUseCustomRange=1,
                    BottomAxisRangeMinimum=-0.05, BottomAxisRangeMaximum=1.05, ViewSize=self.line_view_size
                )
            info = plot.GetDataInformation().DataInformation.GetCompositeDataInformation()
            names = ['Cp ({0})'.format(info.GetName(i)) for i in range(info.GetNumberOfChildren())]
            pvs.Show(
                plot, self.line_view, UseIndexForXAxis=0, XArrayName='X/c',
                CompositeDataSetIndex=list(range(1, len(names) + 1)),
                SeriesVisibility=names,
                SeriesLabel=utils.insert_repeat(names, label),
                SeriesColor=utils.insert_repeat(names, *self.str_color(color)),
                SeriesLineStyle=utils.insert_repeat(names, '0'),
                SeriesLineThickness=utils.insert_repeat(names, '2'),
                SeriesMarkerStyle=utils.insert_repeat(names, str(marker))
            )

        self.annotate_time(reader, render_view, time, progress)


class COVOPlotter(Plotter):
    def __init__(self, ):
        super().__init__()

    def register_plot(self, filename, cell_array, view_size=(400, 400), contour=0,
                      time=False, progress=False, label='', gamma=1.4, r=None):
        reader, render_view, _ = self.load_data(filename, [cell_array], render_view=True, rvs=view_size)

        if contour > 0:
            pvs.Show(reader, render_view, Representation='Surface', ColorArrayName=[])
            if cell_array not in reader.PointArrays:
                c2p = pvs.CellDatatoPointData(reader, ProcessAllArrays=0, CellDataArraytoprocess=[cell_array])
            else:
                c2p = reader
            ctr = pvs.Contour(c2p, ContourBy=['POINTS', cell_array],
                              Isosurfaces=np.linspace(*c2p.PointData.GetArray(0).GetRange(), contour + 2)[1:-1])
            print(c2p.PointData.GetArray(0).GetRange())
            display = pvs.Show(ctr, render_view, Representation='Surface', ColorArrayName=['POINTS', cell_array])
        else:
            display = pvs.Show(reader, render_view, Representation='Surface', ColorArrayName=['POINTS', cell_array])
        display.LookupTable = pvs.GetColorTransferFunction(cell_array, display, separate=True)
        display.RescaleTransferFunctionToDataRange(False, True)
        display.SetScalarBarVisibility(render_view, True)

        if label:
            text = pvs.Text(Text=label)
            pvs.Show(text, render_view, 'TextSourceRepresentation', WindowLocation='UpperCenter', Interactivity=0)

        if r is not None:
            self.load_data(filename, ['V', 'T'])
            vtk_data = pvs.servermanager.Fetch(reader)
            try:
                block = vtk_data.GetBlock(0)
                for i in range(vtk_data.GetNumberOfBlocks()):
                    block = vtk_data.GetBlock(i)
                    if block.GetBounds()[:3:2] == reader.GetDataInformation().DataInformation.GetBounds()[:3:2]:
                        break
            except AttributeError:
                block = vtk_data
            for p in range(block.GetNumberOfCells()):
                if block.GetCell(p).GetBounds()[:3:2] == block.GetBounds()[:3:2]:
                    bounds = block.GetCell(p).GetBounds()
                    dx = np.sqrt(((bounds[1] - bounds[0]) ** 2 + (bounds[3] - bounds[2]) ** 2) / 2)
                    dt = np.max(np.diff(reader.TimestepValues))
                    try:
                        t = block.GetCellData().GetArray('T').GetValue(p)
                        v = block.GetCellData().GetArray('V').GetValue(p)
                        cfl = (np.sqrt(gamma * r * t) + v) * dt / dx
                    except AttributeError:
                        rho = block.GetPointData().GetArray('rho').GetValue(p)
                        pres = block.GetPointData().GetArray('rho').GetValue(p)
                        u = block.GetPointData().GetArray('u').GetValue(p)
                        v = block.GetPointData().GetArray('v').GetValue(p)
                        cfl = (np.sqrt(gamma * pres / rho) + np.sqrt(u * u + v * v)) * dt / dx
                    text = pvs.Text(Text='CFL = {0:.3f}'.format(cfl))
                    pvs.Show(text, render_view, 'TextSourceRepresentation', WindowLocation='UpperRightCorner',
                             Interactivity=0)

        self.annotate_time(reader, render_view, time, progress)


class JetPlotter(Plotter):
    def __init__(self, ):
        super().__init__()

    def register_plot(self, filename, cell_array, view_size=(400, 400), contour=False, label=''):
        reader, render_view, _ = self.load_data(filename, [cell_array], render_view=True, rvs=view_size)
        clip = pvs.Clip(reader, Invert=0, Crinkleclip=1)
        clip.ClipType.Origin = [0, -0.3, 0]
        clip.ClipType.Normal = [0, 1, 0]
        reflect = pvs.Reflect(clip, CopyInput=1)
        if contour:
            self.load_data(filename, ['Alpha_1'])
            c2p = pvs.CellDatatoPointData(reflect, ProcessAllArrays=0, CellDataArraytoprocess=['Alpha_1'])
            ctr = pvs.Contour(c2p, ContourBy=['POINTS', 'Alpha_1'], Isosurfaces=[0.5, ])
            pvs.Show(ctr, render_view, Representation='Wireframe', LineWidth=3, ColorArrayName=['CELLS', None])

        if label:
            text = pvs.Text(Text=label)
            pvs.Show(text, render_view, 'TextSourceRepresentation', WindowLocation='UpperCenter', Interactivity=0)

        display = pvs.Show(reflect, render_view, Representation='Surface', ColorArrayName=['CELLS', 'Rho'])
        display.LookupTable = pvs.GetColorTransferFunction('Rho', display, separate=True)
        display.RescaleTransferFunctionToDataRange(False, True)
        display.SetScalarBarVisibility(render_view, True)
