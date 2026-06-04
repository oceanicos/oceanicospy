import glob
import numpy as np

from .... import utils


class CaseRunner():
    """
    Finalizes the XBeach ``params.txt`` configuration and prepares the case for execution.

    After all forcing and bathymetry inputs have been written, ``CaseRunner``
    fills the remaining sections of ``params.txt``: the output file path,
    optional output point locations, global and point-based output variable
    lists, and the computation timing block.

    Parameters
    ----------
    init : object
        Case initialization object.  Must expose ``ini_date``, ``end_date``,
        and ``dict_folders`` (a mapping with at least ``"input"`` and ``"run"``
        keys pointing to the respective directories).
    dict_comp_data : dict
        Mutable dictionary whose keys correspond to ``$placeholder`` tokens in
        ``params.txt``.  Methods in this class populate or update values in
        this dictionary before ``fill_computation_section`` writes it to disk.
    """

    def __init__(self, init, dict_comp_data):
        self.init = init
        self.dict_comp_data = dict_comp_data
        print('\n*** Initializing Case Runner ***\n')

    def write_output_file(self, filename):
        """
        Set the NetCDF output file name in the computation parameters.

        Parameters
        ----------
        filename : str
            Name (or relative path) of the NetCDF output file that XBeach
            will create, e.g. ``'output.nc'``.
        """
        self.dict_comp_data['outputfilepath'] = filename

    def write_output_points(self, filename=None):
        """
        Load output point coordinates from a text file and register them.

        Searches the case input directory for ``filename``, reads the two-column
        (x, y) coordinate table, and stores the formatted point list and its
        length in ``dict_comp_data`` so they can be injected into ``params.txt``.
        If the file is not found, the point list is left empty (zero points).

        Parameters
        ----------
        filename : str, optional
            Name of the whitespace-delimited text file with output point
            coordinates (two columns: x, y).  The file must reside in the
            case input directory (``init.dict_folders["input"]``).
        """
        try:
            points_file = glob.glob(f'{self.init.dict_folders["input"]}{filename}')[0]
        except IndexError:
            points_file = None

        if points_file:
            points_data = np.loadtxt(points_file)
            self.dict_comp_data['len_points'] = len(points_data)
            string_points = [f'{point[0]} {point[1]}\n' for point in points_data]
            string_points[-1] = string_points[-1].strip()  # remove trailing newline
            self.dict_comp_data['string_points'] = ''.join(string_points)
        else:
            self.dict_comp_data['len_points'] = 0
            self.dict_comp_data['string_points'] = ''

    def select_global_vars(self, list_vars=None):
        """
        Register global (grid-wide) output variables.

        Global variables are written to the NetCDF output file for every grid
        cell at every output time step.

        Parameters
        ----------
        list_vars : list of str, optional
            XBeach variable names to include in the global output block
            (e.g. ``['zb', 'zs', 'urms']``).  When ``None`` or empty, the
            global output block is left empty (zero variables).
        """
        if list_vars:
            self.dict_comp_data['len_global_vars'] = len(list_vars)
            self.dict_comp_data['global_vars'] = '\n'.join(str(var) for var in list_vars)
        else:
            self.dict_comp_data['len_global_vars'] = 0
            self.dict_comp_data['global_vars'] = ''

    def select_point_vars(self, list_vars=None):
        """
        Register point-based output variables.

        Point variables are written to the NetCDF output file only at the
        locations defined via :meth:`write_output_points`.

        Parameters
        ----------
        list_vars : list of str, optional
            XBeach variable names to include in the point output block
            (e.g. ``['zs', 'u', 'v']``).  When ``None`` or empty, the point
            output block is left empty (zero variables).
        """
        if list_vars:
            self.dict_comp_data['len_point_vars'] = len(list_vars)
            self.dict_comp_data['point_vars'] = '\n'.join(str(var) for var in list_vars)
        else:
            self.dict_comp_data['len_point_vars'] = 0
            self.dict_comp_data['point_vars'] = ''

    def fill_computation_section(self):
        """
        Compute the simulation duration and write all parameters to ``params.txt``.

        Derives ``tstop`` (in seconds) from the difference between
        ``init.end_date`` and ``init.ini_date``, stores it in
        ``dict_comp_data``, converts every value to ``str`` (required by the
        placeholder-substitution engine), and calls :func:`utils.fill_files` to
        replace all ``$placeholder`` tokens in ``params.txt`` with their
        corresponding values.
        """
        seconds = (self.init.end_date - self.init.ini_date).total_seconds()
        self.dict_comp_data['tstop_value'] = int(seconds)

        if 'tstart_value' not in self.dict_comp_data:
            self.dict_comp_data['tstart_value'] = 10800  # Default to 3 hours

        str_comp_data = {k: str(v) for k, v in self.dict_comp_data.items()}
        self.dict_comp_data.update(str_comp_data)

        utils.fill_files(f'{self.init.dict_folders["run"]}params.txt', self.dict_comp_data)