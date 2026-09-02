import glob as glob
from .... import utils

class GridMaker:
    """
    GridMaker is a utility class for generating and managing the grid information for SWAN.

    Parameters
    ----------
    init : object
        An initialization object containing configuration data and folder paths.
    domain_number : int
        Identifier for the domain being processed.
    grid_info : dict or None, optional
        User-provided grid information dictionary.
        If the grid_info dictionary is passed it should contain the following keys: 
        ``lon_ll_corner``: longitude of the lower-left corner of the grid
        ``lat_ll_corner``: latitude of the lower-left corner of the grid
        ``x_extent``: total extent of the grid in the x-direction (longitude)
        ``y_extent``: total extent of the grid in the y-direction (latitude)
        ``nx`` : number of grid cells in the x-direction
        ``ny``: number of grid cells in the y-direction
    dx : float or None, optional
        Grid spacing in the x-direction (longitude).
    dy : float or None, optional
        Grid spacing in the y-direction (latitude).

    Notes
    -----
    This class is used to generate and manage grid information for SWAN simulations.
    """

    def __init__(self, init, domain_number, grid_info = None, dx = None, dy = None):
        self.init = init
        self.domain_number = domain_number
        self.grid_info = grid_info
        self.dx = dx
        self.dy = dy     
        print(f'\n*** Initializing gridmaker for domain {self.domain_number} ***\n')
    
    def fill_grid_section(self):
        """
        Replaces and updates the `.swn` file with the grid configuration for a specific domain. 
        The info used to fill the grid section is obtained from the `grid_info` attribute, 
        which can be set either by passing a dictionary during initialization or 
        by calling the `get_info_from_bathy()` method to extract it from the bathymetry file.

        """

        if self.grid_info == None:
            raise ValueError(f'Grid information is not provided for domain {self.domain_number}. \
                             Please provide grid_info or ensure that get_info_from_bathy() is called to extract grid information from the bathymetry file.')

        self.grid_info["domain_number"] = self.domain_number

        print (f'\n \t*** Adding/Editing grid information for domain {self.domain_number} in configuration file ***\n')
        utils.fill_files(f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/run.swn',self.grid_info)
