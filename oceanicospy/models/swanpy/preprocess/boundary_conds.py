import numpy as np
import pandas as pd
import xarray as xr
import os
from pathlib import Path

from .... import utils
from ....downloads import *

class BoundaryConditions:
    """
    Utility class for configuring and writing SWAN boundary condition files.

    Supports constant and variable (file-driven) side-boundary conditions, as
    well as nested-grid boundary conditions generated from parent-domain output.
    Wave data can be sourced from ERA5 or CMDS reanalysis products and written
    as SWAN-compatible TPAR or boundary files.

    Parameters
    ----------
    init : object
        Initialization object containing configuration data and folder paths.
    domain_number : int
        Identifier for the domain being processed.
    bound_info : dict or None, optional
        Dictionary describing the boundary type and whether boundaries are
        variable (file-driven) or constant.  Expected keys:

        * ``'bound_type'`` (*str*) – boundary specification type (e.g. ``'side'``).
        * ``'variable_bound'`` (*bool*) – ``True`` for file-driven boundaries,
          ``False`` for constant parametric boundaries.
    input_filename : str or None, optional
        Name of the input file to use as boundary data. Defaults to ``None``.
    use_link : bool or None, optional
        If ``True``, creates symbolic links for boundary files instead of
        copying them.  If ``False``, copies the files.  If ``None``, no file
        placement is performed.
    """
    def __init__ (self,init,domain_number,bound_info=None,input_filename=None,use_link=None):
        self.init = init
        self.domain_number = domain_number
        self.bound_info = bound_info
        self.input_filename = input_filename
        self.use_link = use_link
        self.filepath_localtime = None

        if "parent_domains" in self.init.dict_ini_data:
            if self.init.dict_ini_data["parent_domains"][domain_number] != None:
                self.isnested = True
            else:
                self.isnested = False
        else:
            self.isnested = False

        self.boundary_line = None
        print(f'\n*** Initializing boundary conditions for domain {self.domain_number} ***\n')

    def _download_ERA5(self,utc_offset_hours, filepath=None,wind_info=None,format_localtime=False):
        """
        Download ERA5 wave data for the specified region and time period.

        Initializes an :class:`ERA5Downloader` with the required wave variables
        and spatial bounds derived from *wind_info*, then downloads the data and
        optionally reformats timestamps to local time.

        Parameters
        ----------
        utc_offset_hours : int
            Time difference to UTC in hours for local-time conversion.
        filepath : str or None, optional
            Destination path for the downloaded ERA5 NetCDF file.
            If ``None``, a default path is used.
        wind_info : dict or None, optional
            Spatial extent dictionary (same format as :class:`WindForcing`).
            Expected keys: ``lon_ll_corner_wind``, ``lat_ll_corner_wind``,
            ``nx_wind``, ``ny_wind``, ``dx_wind``, ``dy_wind``.
        format_localtime : bool, optional
            If ``True``, converts timestamps to local time after download.
            Defaults to ``False``.

        Returns
        -------
        str
            Path to the downloaded (or local-time-formatted) NetCDF file.
        """
        filepath = Path(filepath)
        ERA5download_obj = ERA5Downloader(
                        variables = ["mean_wave_direction",
                                   "significant_height_of_combined_wind_waves_and_swell",
                                   "peak_wave_period"],
                        lon_min = wind_info['lon_ll_corner_wind'],
                        lon_max = wind_info['lon_ll_corner_wind'] + (wind_info['nx_wind'] * wind_info['dx_wind']),
                        lat_min = wind_info['lat_ll_corner_wind'],
                        lat_max = wind_info['lat_ll_corner_wind'] + (wind_info['ny_wind'] * wind_info['dy_wind']),
                        start_datetime_local = self.init.ini_date,
                        end_datetime_local = self.init.end_date,
                        utc_offset_hours = utc_offset_hours,
                        output_path = filepath.parent,
                        output_filename = filepath.name
                        )

        self.filepath_utc = ERA5download_obj.download()
        print("\t ERA5 wind data downloaded successfully")

        if format_localtime:
            self.filepath_localtime = ERA5download_obj.format_to_localtime()
            return self.filepath_localtime
        return self.filepath_utc 

    def _download_CMDS(self,utc_offset_hours, filepath=None,wind_info=None,format_localtime=False):
        """
        Download CMDS wave data for the specified region and time period.

        Initializes a :class:`CMDSDownloader` configured for wave variables
        and spatial bounds derived from *wind_info*, then downloads the data
        and optionally reformats timestamps to local time.

        Parameters
        ----------
        utc_offset_hours : int
            Time difference to UTC in hours for local-time conversion.
        filepath : str or None, optional
            Destination path for the downloaded CMDS NetCDF file.
            If ``None``, a default path is used.
        wind_info : dict or None, optional
            Spatial extent dictionary.  Expected keys: ``lon_ll_corner_wind``,
            ``lat_ll_corner_wind``, ``nx_wind``, ``ny_wind``, ``dx_wind``,
            ``dy_wind``.
        format_localtime : bool, optional
            If ``True``, converts timestamps to local time after download.
            Defaults to ``False``.

        Returns
        -------
        str
            Path to the downloaded (or local-time-formatted) NetCDF file.
        """
        filepath = Path(filepath)
        CMDSdownload_obj = CMDSDownloader.for_waves(
                        lon_min = wind_info['lon_ll_corner_wind'],
                        lon_max = wind_info['lon_ll_corner_wind'] + (wind_info['nx_wind'] * wind_info['dx_wind']),
                        lat_min = wind_info['lat_ll_corner_wind'],
                        lat_max = wind_info['lat_ll_corner_wind'] + (wind_info['ny_wind'] * wind_info['dy_wind']),
                        start_datetime_local = self.init.ini_date,
                        end_datetime_local = self.init.end_date,
                        utc_offset_hours = utc_offset_hours,
                        output_path = filepath.parent,
                        output_filename = filepath.name
                        )
        self.filepath_utc = CMDSdownload_obj.download()
        print("\t CMDS wind data downloaded successfully")
        if format_localtime:
            self.filepath_localtime = CMDSdownload_obj.format_to_localtime()
            return self.filepath_localtime
        return self.filepath_utc

    def _single_tpar_from_ERA5(self,tpar_filename,lati,long,wave_filename='waves_era5.nc'):
        """
        Write a TPAR boundary file for a single point from ERA5 wave data.

        Reads the ERA5 wave NetCDF file (local-time version if available),
        extracts significant wave height, peak period, and mean wave direction
        at the grid cell nearest to (*lati*, *long*), and writes a
        SWAN-compatible TPAR file.

        Parameters
        ----------
        tpar_filename : str
            Base output path (without ``.bnd`` extension) for the TPAR file.
        lati : float
            Target latitude of the boundary point.
        long : float
            Target longitude of the boundary point.
        wave_filename : str, optional
            Name of the ERA5 wave NetCDF file in the domain input directory.
            Defaults to ``'waves_era5.nc'``.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``Tiempo``, ``Altura``, ``Periodo``,
            ``Direccion``, and ``dd`` for the selected point.
        """
        filepath = f'{self.init.dict_folders["input"]}domain_0{self.domain_number}/{wave_filename}'

        if Path(filepath).with_name(Path(filepath).stem + '_localtime.nc').exists():
            filepath_localtime = Path(filepath).with_name(Path(filepath).stem + '_localtime.nc')
            ds = xr.open_dataset(filepath_localtime)
        else:
            ds = xr.open_dataset(filepath)

        time = ds.valid_time.values
        strtime = [pd.to_datetime(t).strftime("%Y%m%d.%H%M") for t in time]
        lat_idx = np.argmin(np.abs(ds.latitude.values - lati))
        lon_idx = np.argmin(np.abs(ds.longitude.values - long))

        swh = np.zeros(len(time))
        pp = np.zeros(len(time))
        mwd = np.zeros(len(time))
        for i in range(len(time)):
            data_per_time = ds.isel(valid_time=np.where(ds.valid_time.values==time[i])[0][0],
                            latitude=lat_idx,longitude=lon_idx)
            swh[i] = data_per_time.swh.values
            pp[i] = data_per_time.pp1d.values
            mwd[i] = data_per_time.mwd.values
        df_tpar = pd.DataFrame({'Tiempo':strtime,'Altura':swh,'Periodo':pp,'Direccion':mwd,'dd':40})
        with open (tpar_filename+'.bnd', "w") as file:
                file.write("TPAR \n")
                np.savetxt(file,df_tpar,fmt =('%s  %7.9f  %8.9f  %9.9f  %5.1f'))
        return df_tpar

    def _single_tpar_from_CMDS(self,tpar_filename,lati,long,wave_filename='waves_cmds.nc'):
        """
        Write a TPAR boundary file for a single point from CMDS wave data.

        Reads the CMDS wave NetCDF file (local-time version if available),
        extracts significant wave height (``VHM0``), peak period (``VTPK``),
        and mean wave direction (``VMDR``) at the grid cell nearest to
        (*lati*, *long*), and writes a SWAN-compatible TPAR file.

        Parameters
        ----------
        tpar_filename : str
            Base output path (without ``.bnd`` extension) for the TPAR file.
        lati : float
            Target latitude of the boundary point.
        long : float
            Target longitude of the boundary point.
        wave_filename : str, optional
            Name of the CMDS wave NetCDF file in the domain input directory.
            Defaults to ``'waves_cmds.nc'``.

        Returns
        -------
        pd.DataFrame
            DataFrame with columns ``Tiempo``, ``Altura``, ``Periodo``,
            ``Direccion``, and ``dd`` for the selected point.
        """
        filepath = f'{self.init.dict_folders["input"]}domain_0{self.domain_number}/{wave_filename}'

        if Path(filepath).with_name(Path(filepath).stem + '_localtime.nc').exists():
            filepath_localtime = Path(filepath).with_name(Path(filepath).stem + '_localtime.nc')
            ds = xr.open_dataset(filepath_localtime)
        else:
            ds = xr.open_dataset(filepath)

        time = ds.time.values
        strtime = [pd.to_datetime(t).strftime("%Y%m%d.%H%M") for t in time]
        lat_idx = np.argmin(np.abs(ds.latitude.values - lati))
        lon_idx = np.argmin(np.abs(ds.longitude.values - long))

        swh = np.zeros(len(time))
        pp = np.zeros(len(time))
        mwd = np.zeros(len(time))
        for i in range(len(time)):
            data_per_time = ds.isel(time=np.where(ds.time.values==time[i])[0][0],
                            latitude=lat_idx,longitude=lon_idx)
            swh[i] = data_per_time.VHM0.values
            pp[i] = data_per_time.VTPK.values
            mwd[i] = data_per_time.VMDR.values
        df_tpar = pd.DataFrame({'Tiempo':strtime,'Altura':swh,'Periodo':pp,'Direccion':mwd,'dd':40})
        with open (tpar_filename+'.bnd', "w") as file:
                file.write("TPAR \n")
                np.savetxt(file,df_tpar,fmt =('%s  %7.9f  %8.9f  %9.9f  %5.1f'))
        return df_tpar

    def _process_boundary_points(self,points_lat,points_lon,single_tpar_fn):
        """
        Generate TPAR files for all four boundary sides and deploy them.

        For non-nested domains, iterates over *points_lon* and *points_lat* to
        write TPAR files for the north, south, east, and west boundaries using
        *single_tpar_fn*.  Nested domains are silently skipped.  After all
        files are written, calls :meth:`_copy_or_link_bnd_files` to place them
        in the run directory.

        Parameters
        ----------
        points_lat : list or array-like
            Latitude values of the boundary points.
        points_lon : list or array-like
            Longitude values of the boundary points.
        single_tpar_fn : callable
            Function with signature ``(tpar_filename, lati, long)`` used to
            write individual TPAR files (e.g. :meth:`_single_tpar_from_ERA5`).
        """
        if self.isnested:
            return
        self.input_path = f'{self.init.dict_folders["input"]}domain_0{self.domain_number}/'
        for idx_lon,lon in enumerate(points_lon):
            single_tpar_fn(f'{self.input_path}TparN_{idx_lon+1}',max(points_lat),lon)
            single_tpar_fn(f'{self.input_path}TparS_{idx_lon+1}',min(points_lat),lon)

        for idx_lat,lat in enumerate(points_lat):
            single_tpar_fn(f'{self.input_path}TparE_{idx_lat+1}',lat,max(points_lon))
            single_tpar_fn(f'{self.input_path}TparW_{idx_lat+1}',lat,min(points_lon))

        self._copy_or_link_bnd_files()
        print(f'\t*** Finished processing boundary files for domain {self.domain_number} ***\n')

    def _copy_or_link_bnd_files(self):
        """
        Copy or symlink all ``.bnd`` files from the input to the run directory.

        Scans the domain input directory for files ending in ``.bnd`` and
        deploys each one to the run directory via
        :func:`~oceanicospy.utils.deploy_input_file`.  If ``use_link`` is
        ``None`` no action is taken.
        """
        if self.use_link is None:
            return
        run_domain_dir = f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/'
        origin_domain_dir = f'{self.init.dict_folders["input"]}domain_0{self.domain_number}/'

        bnd_files = [f for f in os.listdir(self.input_path) if f.endswith('.bnd')]

        for bnd_file in bnd_files:
            utils.deploy_input_file(bnd_file, origin_domain_dir, run_domain_dir, self.use_link)

    def get_waves_from_ERA5(self,utc_offset_hours,wind_info_dict,filename='waves_era5.nc',override=False,format_localtime=False):
        """
        Download ERA5 wave data for the current domain, or skip if already present.

        Checks whether the ERA5 wave NetCDF file already exists in the domain
        input directory.  If it does not exist (or *override* is ``True``),
        the data are downloaded via :meth:`_download_ERA5`.

        Parameters
        ----------
        utc_offset_hours : int
            Time difference to UTC in hours for local-time conversion.
        wind_info_dict : dict
            Spatial extent dictionary forwarded to :meth:`_download_ERA5`.
        filename : str, optional
            Name of the ERA5 wave NetCDF output file.
            Defaults to ``'waves_era5.nc'``.
        override : bool, optional
            If ``True``, re-downloads the file even if it already exists.
            Defaults to ``False``.
        format_localtime : bool, optional
            If ``True``, converts timestamps to local time after download.
            Defaults to ``False``.
        """
        if self.isnested == False:
            filepath = f"{self.init.dict_folders['input']}domain_0{self.domain_number}/{filename}"
            file_exists = utils.verify_file(filepath)
            if not file_exists or override:
                self._download_ERA5(utc_offset_hours,wind_info=wind_info_dict,filepath=filepath,format_localtime=format_localtime)
            else:
                print("\t ERA5 wave data already exists, skipping download")

    def get_waves_from_CMDS(self,utc_offset_hours,wind_info_dict,filename='waves_cmds.nc',override=False,format_localtime=False):
        """
        Download CMDS wave data for the current domain, or skip if already present.

        Checks whether the CMDS wave NetCDF file already exists in the domain
        input directory.  If it does not exist (or *override* is ``True``),
        the data are downloaded via :meth:`_download_CMDS`.

        Parameters
        ----------
        utc_offset_hours : int
            Time difference to UTC in hours for local-time conversion.
        wind_info_dict : dict
            Spatial extent dictionary forwarded to :meth:`_download_CMDS`.
        filename : str, optional
            Name of the CMDS wave NetCDF output file.
            Defaults to ``'waves_cmds.nc'``.
        override : bool, optional
            If ``True``, re-downloads the file even if it already exists.
            Defaults to ``False``.
        format_localtime : bool, optional
            If ``True``, converts timestamps to local time after download.
            Defaults to ``False``.
        """
        if self.isnested == False:
            filepath = f"{self.init.dict_folders['input']}domain_0{self.domain_number}/{filename}"
            file_exists = utils.verify_file(filepath)
            if not file_exists or override:
                self._download_CMDS(utc_offset_hours,wind_info=wind_info_dict,filepath=filepath,format_localtime=format_localtime)
            else:
                print("\t CMDS wave data already exists, skipping download")

    def tpar_from_ERA5(self,points_lat,points_lon):
        """
        Generates TPAR files from ERA5 data for specified boundary points.

        For a non-nested domain, this method iterates over the provided latitude and longitude points,
        generating TPAR files for the north, south, east, and west boundaries using ERA5 data.

        Parameters
        ----------
        points_lat : list or array-like
            List of latitude points defining the boundary.
        points_lon : list or array-like
            List of longitude points defining the boundary.
        """
        points_lon = np.array(points_lon)
        if np.any(points_lon<0):
            points_lon += 360

        self._process_boundary_points(points_lat,points_lon,self._single_tpar_from_ERA5)

    def tpar_from_CMDS(self,points_lat,points_lon):
        """
        Generates TPAR files from CMDS data for specified boundary points.

        For a non-nested domain, this method iterates over the provided latitude and longitude points,
        generating TPAR files for the north, south, east, and west boundaries using CMDS data.

        Parameters
        ----------
        points_lat : list or array-like
            List of latitude points defining the boundary.
        points_lon : list or array-like
            List of longitude points defining the boundary.
        """
        points_lon = np.array(points_lon)
        self._process_boundary_points(points_lat,points_lon,self._single_tpar_from_CMDS)

    _VALID_SIDES = {'N', 'S', 'E', 'W'}

    def _build_side_boundary_line(self, side: str, wave_params: dict | None, 
                                  points_lon: list[float], points_lat: list[float]) -> str:
        """
        Build a single SWAN ``BOUN SIDE`` command string for one boundary side.

        Parameters
        ----------
        side : str
            Cardinal direction of the boundary side.  Must be one of
            ``'N'``, ``'S'``, ``'E'``, ``'W'``.
        wave_params : dict or None
            Wave parameters required when ``bound_info['variable_bound']``
            is ``False`` ('constant').  Expected keys:

            * ``'hs'``     — significant wave height (m)
            * ``'tp'``     — peak period (s)
            * ``'dir'``    — peak wave direction (deg)
            * ``'spr'``    — wave spread (power of cosine function)

            Must be ``None`` or omitted for variable (file-driven) boundaries.
        points_lon : list of float
            Longitude values of the boundary points along this side.
            Used to compute relative offsets for ``VAR FILE`` boundary commands.
        points_lat : list of float
            Latitude values of the boundary points along this side.
            Used to compute relative offsets for ``VAR FILE`` boundary commands.

        Returns
        -------
        str
            A single SWAN boundary command, e.g.
            ``"BOUN SIDE N CLOCKW CON PAR 1.5 10.0 270 4"`` or
            ``"BOUN SIDE N CLOCKW VAR FILE …"``.

        Raises
        ------
        ValueError
            If *side* is not in ``{'N', 'S', 'E', 'W'}``.
        ValueError
            If ``variable_bound`` is ``False`` but *wave_params* is ``None``.
        """
        if side not in self._VALID_SIDES:
            raise ValueError(
                f"Invalid boundary side '{side}'. "
                f"Valid options are: {sorted(self._VALID_SIDES)}."
            )

        if not self.bound_info['variable_bound']:
            if wave_params is None:
                raise ValueError(
                    "wave_params must be provided when variable_bound is False ('constant')."
                )
            lines_per_side = f"BOUN SIDE {side} CLOCKW CON PAR {wave_params['hs']} {wave_params['tp']} {wave_params['dir']} {wave_params['spr']}"
        else:
            self.input_path = f'{self.init.dict_folders["input"]}domain_0{self.domain_number}/'
            bnd_files = [f for f in os.listdir(self.input_path) if f.endswith('.bnd') and f'{side}' in f]
            sorted_bnd_files = sorted(bnd_files, key=lambda x: int(''.join(filter(str.isdigit, x))))

            if side in ['S', 'E']:
                sorted_bnd_files = sorted_bnd_files[::-1]

            lines_per_side = ""
            for idx, bnd_file in enumerate(sorted_bnd_files):
                if side in ['N', 'S']:
                    difference = points_lon[idx] - points_lon[0]
                else:
                    difference = points_lat[idx] - points_lat[0]
                round_difference = round(difference, 2)

                if idx == 0:
                    lines_per_side += f"BOUN SIDE {side} CLOCKW VAR FILE {round_difference} '../../input/domain_0{self.domain_number}/{bnd_file}' 1 & \n"
                else:
                    is_last = idx == len(sorted_bnd_files) - 1
                    newline = '' if is_last else ' \n'
                    lines_per_side += f"{round_difference} '../../input/domain_0{self.domain_number}/{bnd_file}' 1 &{newline}"
            
        return lines_per_side

    def create_boundary_line(self, list_sides: list[str] | None = None,
                              wave_params: dict | None = None,
                              points_lon: list[float] = None, points_lat: list[float] = None) -> None:
        """
        Build and store the SWAN boundary command lines for all requested sides.

        For each entry in *list_sides*, delegates to
        :meth:`_build_side_boundary_line` to produce one ``BOUN SIDE``
        command.  All commands are joined with newlines and stored in
        :attr:`boundary_line`, ready for :meth:`fill_boundaries_section`.

        Parameters
        ----------
        list_sides : list of str, optional
            Ordered list of boundary sides to configure, e.g.
            ``['N', 'S', 'E', 'W']``.  Each entry must be a valid side
            accepted by :meth:`_build_side_boundary_line`.
        wave_params : dict or None, optional
            Wave parameters forwarded to :meth:`_build_side_boundary_line`
            when ``bound_info['variable_bound']`` is ``False`` (constant).
            Ignored for variable (file-driven) boundaries.
        points_lon : list of float or None, optional
            Longitude values of the boundary points.  Required for variable
            (file-driven) boundaries to compute offsets per side.
        points_lat : list of float or None, optional
            Latitude values of the boundary points.  Required for variable
            (file-driven) boundaries to compute offsets per side.

        Returns
        -------
        None

        Raises
        ------
        ValueError
            If :attr:`bound_info` is ``None`` (boundary type not configured).
        NotImplementedError
            If ``bound_info['bound_type']`` is not ``'side'``.
        ValueError
            If any entry in *list_sides* is not a valid boundary side.
        """
        if self.bound_info is None:
            raise ValueError(
                "Boundary type information is missing. "
                "Please ensure 'bound_info' is provided at initialisation."
            )

        if not self.isnested:
            if self.bound_info['bound_type'] != 'side':
                raise NotImplementedError(
                    f"Boundary type '{self.bound_info['bound_type']}' is not supported. "
                    "Only 'side' boundaries are currently implemented."
                )
            self.list_sides = list_sides or []
            self.boundary_line = "\n".join(self._build_side_boundary_line(side, wave_params, points_lon, points_lat) for side in self.list_sides)
        else:
            self.boundary_line = f"BOUN NEST 'child0{self.init.dict_ini_data['parent_domains'][self.domain_number]}_0{self.domain_number}.NEST' CLOSED"


    def fill_boundaries_section(self):
        """
        Replaces and updates the .swn file with the boundary configuration for a specific domain.

        Raises
        ------
        ValueError
            If no boundary information was provided at initialization.
        """
        if self.boundary_line is not None:
            self.dict_boundaries={'boundaries_line':self.boundary_line}
            print (f'\n \t*** Adding/Editing boundary information for domain {self.domain_number} in configuration file ***\n')
            utils.fill_files(f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/run.swn',self.dict_boundaries)
        else:
            raise ValueError("Boundary line information is missing. Please ensure 'create_boundary_line' method has been called successfully before filling the boundaries section.")