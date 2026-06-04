import numpy as np
import glob as glob

from scipy.interpolate import griddata
from pathlib import Path
from typing import Optional

from .... import utils
from ....gis import PointFileIO, XYZFormatSpec


class BathyMaker:
    """
    BathyMaker is a utility class for generating and managing the bathymetry information for SWAN.

    Parameters
    ----------
    init : object
        An initialization object containing configuration data and folder paths.
    domain_number : int
        Identifier for the domain being processed.
    bathy_info : dict or None, optional
        Dictionary containing bathymetric information. If None, spatial info for bathymetry must be provided via `get_info_from_bathy()`.
    use_link: bool, optional
        If True, creates symbolic links for the bathymetry file instead of copying it. Defaults to True.
    """

    def __init__(self, init, domain_number, bathy_info = None, use_link = None):
        self.init = init
        self.domain_number = domain_number
        self.bathy_info = bathy_info
        self.use_link = use_link
        print(f'\n*** Initializing bathymaker for domain {self.domain_number} ***\n')

    def use_ascii_file_from_user(self):
        """
        Handles the selection and linking or copying of a bathymetry file for the current domain.

        Searches for a `.bot` bathymetry file in the input directory for the specified domain.
        Depending on ``use_link``, the file is either symlinked or physically copied into the
        run directory.

        Returns
        -------
        dict or None
            The updated ``bathy_info`` dictionary if it was provided at initialisation,
            otherwise ``None``.

        Raises
        ------
        FileNotFoundError
            If no `.bot` file is found in the expected input directory.
        """
    
        bathy_filepaths = glob.glob(f'{self.init.dict_folders["input"]}domain_0{self.domain_number}/*.bot')
        if not bathy_filepaths:
            raise FileNotFoundError(f'Bathymetry file not found in {self.init.dict_folders["input"]}domain_0{self.domain_number}/ or file extension is not .bot')
        bathy_filepath = Path(bathy_filepaths[0])
        bathy_filename = bathy_filepath.name

        run_domain_dir = f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/'

        utils.deploy_input_file(bathy_filename, f'{self.init.dict_folders["input"]}domain_0{self.domain_number}/', run_domain_dir, self.use_link)
        
        if self.bathy_info!=None:
            self.bathy_info.update({"bathy_file":f"../../input/domain_0{self.domain_number}/{bathy_filename}"})
            return self.bathy_info
        else:
            raise ValueError('No bathymetry information provided at initialization. ' \
            'Bathymetry file has been linked/copied to run directory, but no metadata dictionary to return.')
        
    def convert_xyz2asc(self,xyz_filepath: str, dx: float, dy: float, nodata_value: float = -9999,
                        xyz_format:  Optional[XYZFormatSpec] = None,  
                         ):
        """
        Converts bathymetry data from XYZ format to ESRI ASCII Grid format.

        Reads a space-delimited `.dat` file, interpolates the scattered data onto a
        regular grid using linear interpolation, and writes the result as an ESRI
        ASCII Grid (`.bot`) file.

        Parameters
        ----------
        xyz_filepath : str
            The filename of the input XYZ bathymetry file located in the input directory for the current domain.
        dx : float
            Grid spacing in the x-direction (longitude) for the output ASCII grid.
        dy : float
            Grid spacing in the y-direction (latitude) for the output ASCII grid.
        nodata_value : float, optional
            The value to use for grid cells where no data is available after interpolation. Default is -9999.
        xyz_format : XYZFormatSpec, optional
            An optional XYZFormatSpec object specifying the format of the input XYZ file. If not provided,
            the method will assume a default format of space-delimited with no header.

        Returns
        -------
        dict
            A dictionary containing the bathymetry information extracted and calculated from the XYZ file, including:
            - 'lon_ll_corner_bot': longitude of the lower-left corner of the grid
            - 'lat_ll_corner_bot': latitude of the lower-left corner of the grid
            - 'nx_bot': number of grid cells in the x-direction
            - 'ny_bot': number of grid cells in the y-direction
            - 'dx_bot': grid spacing in the x-direction
            - 'dy_bot': grid spacing in the y-direction
            - 'bathy_file': relative path to the generated ASCII bathymetry file
        
        Notes
        -----
        This method only supports cartesian coordinates for the input XYZ file. The output ASCII grid will be in the same coordinate system as the input XYZ data.
        The coordinates of the grid corners are rounded to the nearest 100 to avoid issues with interpolation at the edges of the grid. 
        """
        bathy_xyz_path =  f'{self.init.dict_folders["input"]}domain_0{self.domain_number}/{xyz_filepath}'
        bathy_xyz_path = Path(bathy_xyz_path)
        ascii_file_path = f'{self.init.dict_folders["input"]}domain_0{self.domain_number}/{bathy_xyz_path.stem}.bot'
        ascii_file_path = Path(ascii_file_path)

        point_io_handler = PointFileIO(bathy_xyz_path, xyz_format)
        bathy_xyz_df = point_io_handler.read()

        x_column = point_io_handler.format_spec.x_column
        y_column = point_io_handler.format_spec.y_column
        z_column = point_io_handler.format_spec.z_column

        longitude = bathy_xyz_df[x_column]
        latitude = bathy_xyz_df[y_column]
        z = bathy_xyz_df[z_column]

        min_longitude,max_longitude = longitude.min(), longitude.max()
        min_latitude, max_latitude = latitude.min(), latitude.max()

        # ensuring the grid corners are rounded to the nearest 100 to avoid issues with interpolation at the edges of the grid
        min_longitude = int(np.ceil(min_longitude / 100) * 100)
        max_longitude = int(np.floor(max_longitude / 100) * 100)
        min_latitude = int(np.ceil(min_latitude / 100) * 100)
        max_latitude = int(np.floor(max_latitude / 100) * 100)

        nx_bathy = int((max_longitude - min_longitude)/dx)
        ny_bathy = int((max_latitude - min_latitude)/dy)

        # Generate grid with data
        xi, yi = np.meshgrid(np.arange(min_longitude, max_longitude + dx, dx), np.arange(min_latitude, max_latitude + dy, dy))
        # yi = np.flip(yi, axis=0)  # Flip the y-axis to match map-like orientation

        # Interpolate bathymetry. Method can be 'linear', 'nearest' or 'cubic'
        zi = griddata((longitude,latitude), z, (xi, yi), method='linear')
        zi[np.isnan(zi)] = nodata_value

        # Write ESRI ASCII Grid file
        zi_str = np.where(zi == nodata_value, str(nodata_value), np.round(zi, 3))
        np.savetxt(ascii_file_path, zi_str, fmt='%8s', delimiter=' ')
        print('File %s saved successfuly.' % ascii_file_path)

        utils.deploy_input_file(ascii_file_path.name, ascii_file_path.parent, f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/', self.use_link)

        self.bathy_info={'lon_ll_corner_bot':min_longitude,
                        'lat_ll_corner_bot':min_latitude,
                        'nx_bot':nx_bathy,
                        'ny_bot':ny_bathy,
                        'dx_bot':dx,
                        'dy_bot':dy,
                        'bathy_file':f"../../input/domain_0{self.domain_number}/{ascii_file_path.name}"}
        
        for key,value in self.bathy_info.items():
            self.bathy_info[key] = str(value)
        return self.bathy_info
    
    def fill_bathy_section(self):
        """
        Replaces and updates the `.swn` file with the bathymetry configuration for a specific domain.

        Raises
        ------
        ValueError
            If no bathymetry information was provided at initialization.
        """

        if self.bathy_info == None:
            raise ValueError('No bathymetry information provided at initialization. Cannot fill bathymetry section in configuration file.')

        print (f'\n \t*** Adding/Editing bathymetry information for domain {self.domain_number} in configuration file ***\n')
        utils.fill_files(f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/run.swn',self.bathy_info)
