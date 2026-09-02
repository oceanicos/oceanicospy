import xarray as xr
import pandas as pd
import numpy as np
import glob as glob

from .... import utils
from ....utils.wind import download_era5_winds, download_cmds_winds


class WindForcing:
    """
    Preprocess wind forcing files for XBeach.

    Parameters
    ----------
    init : object
        Project initialization object.  Must provide ``init.ini_date``,
        ``init.end_date``, and ``init.dict_folders`` with keys
        ``"input"`` and ``"run"``.
    input_filename : str or None, optional
        Name of a pre-existing wind file in the input folder.
    wind_info : dict or None, optional
        Dictionary containing spatial wind domain information.  Required
        by the download and conversion methods.  Expected keys:
        ``lon_ll_corner_wind``, ``lat_ll_corner_wind``, ``nx_wind``,
        ``ny_wind``, ``dx_wind``, ``dy_wind``.
    use_link : bool or None, optional
        If ``True``, deploy wind files as symbolic links.
        If ``False``, copy them physically.
        If ``None``, no file deployment is performed.
    """

    def __init__(self, init, input_filename=None, wind_info=None, use_link=None):
        self.init = init
        self.input_filename = input_filename
        self.wind_info = wind_info
        self.use_link = use_link
        print(f'\n*** Initializing winds ***\n')

    def _download_ERA5(self, utc_offset_hours, filepath=None, format_localtime=False):
        """
        Download ERA5 wind data for the specified region and time period.

        Parameters
        ----------
        utc_offset_hours : int
            Time difference to UTC in hours for local time conversion.
        filepath : str or None, optional
            Full path where the downloaded ERA5 file will be saved.
        format_localtime : bool, optional
            If True, formats the time in local time. Defaults to False.
        """
        download_era5_winds(self.wind_info, self.init.ini_date, self.init.end_date, utc_offset_hours, filepath, format_localtime)

    def _download_CMDS(self, utc_offset_hours, filepath=None, format_localtime=False):
        """
        Download CMDS wind data for the specified region and time period.

        Parameters
        ----------
        utc_offset_hours : int
            Time difference to UTC in hours for local time conversion.
        filepath : str or None, optional
            Full path where the downloaded CMDS file will be saved.
        format_localtime : bool, optional
            If True, formats the time in local time. Defaults to False.
        """
        download_cmds_winds(self.wind_info, self.init.ini_date, self.init.end_date, utc_offset_hours, filepath, format_localtime)

    def get_winds_from_ERA5(self, utc_offset_hours, filename='winds_era5.nc', format_localtime=False, override=False):
        """
        Download ERA5 wind data, or skip if already present.

        Checks if the ERA5 NetCDF file exists in the input directory.  If not,
        downloads it using the parameters in ``self.wind_info``.

        Parameters
        ----------
        utc_offset_hours : int
            Time difference to UTC in hours for local time conversion.
        filename : str, optional
            Name of the ERA5 NetCDF output file. Defaults to
            ``'winds_era5.nc'``.
        format_localtime : bool, optional
            If True, formats the time in local time. Defaults to False.
        override : bool, optional
            If ``True``, re-downloads the file even if it already exists.
            Defaults to ``False``.
        """
        filepath = f"{self.init.dict_folders['input']}{filename}"
        file_exists = utils.verify_file(filepath)

        if not file_exists or override:
            self._download_ERA5(utc_offset_hours, filepath=filepath, format_localtime=format_localtime)
        else:
            print("\t ERA5 wind data already exists, skipping download")

    def get_winds_from_CMDS(self, utc_offset_hours, filename='winds_cmds.nc', format_localtime=False, override=False):
        """
        Download CMDS wind data, or skip if already present.

        Checks if the CMDS NetCDF file exists in the input directory.  If not,
        downloads it using the parameters in ``self.wind_info``.

        Parameters
        ----------
        utc_offset_hours : int
            Time difference to UTC in hours for local time conversion.
        filename : str, optional
            Name of the CMDS NetCDF output file. Defaults to
            ``'winds_cmds.nc'``.
        format_localtime : bool, optional
            If True, formats the time in local time. Defaults to False.
        override : bool, optional
            If ``True``, re-downloads the file even if it already exists.
            Defaults to ``False``.
        """
        filepath = f"{self.init.dict_folders['input']}{filename}"
        file_exists = utils.verify_file(filepath)

        if not file_exists or override:
            self._download_CMDS(utc_offset_hours, filepath=filepath, format_localtime=format_localtime)
        else:
            print("\t CMDS wind data already exists, skipping download")

    def _ERA5_nc_to_ascii(self, era5_filename, ascii_filename, lon_target=None, lat_target=None):
        """
        Convert ERA5 wind data from a NetCDF file to XBeach ASCII format.

        Reads ``u10`` and ``v10`` from the NetCDF file, clips to the
        simulation period, computes wind speed and nautical direction, and
        writes a space-separated time-series file suitable for XBeach.

        Parameters
        ----------
        era5_filename : str
            Name of the ERA5 NetCDF file (``u10``, ``v10``, ``valid_time``)
            located in the input folder.
        ascii_filename : str
            Name of the output ASCII file to write in the input folder.
        lon_target : float
            Target longitude for single-point extraction.
        lat_target : float
            Target latitude for single-point extraction.

        Raises
        ------
        ValueError
            If no ERA5 data falls within the simulation dates, or if
            ``lon_target`` or ``lat_target`` is ``None``.
        """
        if lon_target is None or lat_target is None:
            raise ValueError(
                "lon_target and lat_target must be specified for ERA5 wind extraction."
            )

        ds_era5 = xr.load_dataset(
            f'{self.init.dict_folders["input"]}{era5_filename}', engine='netcdf4'
        )

        v10 = ds_era5.variables['v10'].values
        u10 = ds_era5.variables['u10'].values
        time = pd.to_datetime(ds_era5.variables['valid_time'].values)

        mask = (time >= self.init.ini_date) & (time <= self.init.end_date)
        if not mask.any():
            raise ValueError(
                f"No ERA5 data within simulation dates: "
                f"{self.init.ini_date} – {self.init.end_date}"
            )
        time = time[mask]
        u10 = u10[mask]
        v10 = v10[mask]
        time_to_write = (time - time[0]).total_seconds().astype(int).tolist()

        wind_speed = np.sqrt(v10 ** 2 + u10 ** 2)
        wind_dir_cart = np.degrees(np.arctan2(v10, u10))
        wind_dir_naut = (270 - wind_dir_cart) % 360

        lats = ds_era5.variables['latitude'].values
        lons = ds_era5.variables['longitude'].values
        if lon_target < 0:
            lon_target += 360
        lat_idx = np.argmin(np.abs(lats - lat_target))
        lon_idx = np.argmin(np.abs(lons - lon_target))
        df_to_save = pd.DataFrame(
            {'Time': time_to_write, 'Vel': wind_speed[:, lat_idx, lon_idx], 'Dir': wind_dir_naut[:, lat_idx, lon_idx]},
            index=time,
        )

        df_to_save.to_csv(
            f'{self.init.dict_folders["input"]}{ascii_filename}',
            sep=' ', header=False, index=False,
        )

    def _CMDS_nc_to_ascii(self, cmds_filename, ascii_filename, lon_target=None, lat_target=None):
        """
        Convert CMDS wind data from a NetCDF file to XBeach ASCII format.

        Reads ``eastward_wind`` and ``northward_wind`` from the NetCDF file,
        clips to the simulation period, computes wind speed and nautical
        direction, and writes a space-separated time-series file suitable
        for XBeach.

        Parameters
        ----------
        cmds_filename : str
            Name of the CMDS NetCDF file (``eastward_wind``, ``northward_wind``,
            ``time``) located in the input folder.
        ascii_filename : str
            Name of the output ASCII file to write in the input folder.
        lon_target : float
            Target longitude for single-point extraction.
        lat_target : float
            Target latitude for single-point extraction.

        Raises
        ------
        ValueError
            If no CMDS data falls within the simulation dates, or if
            ``lon_target`` or ``lat_target`` is ``None``.
        """
        if lon_target is None or lat_target is None:
            raise ValueError(
                "lon_target and lat_target must be specified for CMDS wind extraction."
            )

        ds_cmds = xr.load_dataset(
            f'{self.init.dict_folders["input"]}{cmds_filename}', engine='netcdf4'
        )

        v10 = ds_cmds.variables['northward_wind'].values
        u10 = ds_cmds.variables['eastward_wind'].values
        tcoord = 'time' if 'time' in ds_cmds.variables else 'valid_time'
        time = pd.to_datetime(ds_cmds.variables[tcoord].values)

        mask = (time >= self.init.ini_date) & (time <= self.init.end_date)
        if not mask.any():
            raise ValueError(
                f"No CMDS data within simulation dates: "
                f"{self.init.ini_date} – {self.init.end_date}"
            )
        time = time[mask]
        u10 = u10[mask]
        v10 = v10[mask]
        time_to_write = (time - time[0]).total_seconds().astype(int).tolist()

        wind_speed = np.sqrt(v10 ** 2 + u10 ** 2)
        wind_dir_cart = np.degrees(np.arctan2(v10, u10))
        wind_dir_naut = (270 - wind_dir_cart) % 360

        lats = ds_cmds.variables['latitude'].values
        lons = ds_cmds.variables['longitude'].values
        if lon_target < 0:
            lon_target += 360
        lat_idx = np.argmin(np.abs(lats - lat_target))
        lon_idx = np.argmin(np.abs(lons - lon_target))
        df_to_save = pd.DataFrame(
            {'Time': time_to_write, 'Vel': wind_speed[:, lat_idx, lon_idx], 'Dir': wind_dir_naut[:, lat_idx, lon_idx]},
            index=time,
        )

        df_to_save.to_csv(
            f'{self.init.dict_folders["input"]}{ascii_filename}',
            sep=' ', header=False, index=False,
        )

    def write_ERA5_ascii(self, era5_filename, ascii_filename, lon_target=None, lat_target=None):
        """
        Convert ERA5 wind data to XBeach ASCII format and deploy it to the run folder.

        Converts the ERA5 NetCDF wind file to ASCII format and places the
        result in the run folder either as a symbolic link or a physical copy,
        depending on ``self.use_link``.

        Parameters
        ----------
        era5_filename : str
            Name of the ERA5 NetCDF wind file in the input folder.
        ascii_filename : str
            Name of the output ASCII file to generate and deploy.
        lon_target : float or None, optional
            Target longitude for single-point extraction.
        lat_target : float or None, optional
            Target latitude for single-point extraction.

        Returns
        -------
        dict or None
            Updated wind information dictionary if ``self.wind_info`` is not
            ``None``, otherwise ``None``.
        """
        input_dir = self.init.dict_folders["input"]
        run_dir = self.init.dict_folders["run"]

        self._ERA5_nc_to_ascii(era5_filename, ascii_filename, lon_target, lat_target)
        print(f"\t ERA5 wind data converted to ASCII format and saved as {ascii_filename}")

        utils.deploy_input_file(ascii_filename, input_dir, run_dir, self.use_link)

        if self.wind_info is not None:
            self.wind_info.update({"windfilepath": ascii_filename})
            return self.wind_info
        return None

    def write_CMDS_ascii(self, cmds_filename, ascii_filename, lon_target=None, lat_target=None):
        """
        Convert CMDS wind data to XBeach ASCII format and deploy it to the run folder.

        Converts the CMDS NetCDF wind file to ASCII format and places the
        result in the run folder either as a symbolic link or a physical copy,
        depending on ``self.use_link``.

        Parameters
        ----------
        cmds_filename : str
            Name of the CMDS NetCDF wind file in the input folder.
        ascii_filename : str
            Name of the output ASCII file to generate and deploy.
        lon_target : float or None, optional
            Target longitude for single-point extraction.
        lat_target : float or None, optional
            Target latitude for single-point extraction.

        Returns
        -------
        dict or None
            Updated wind information dictionary if ``self.wind_info`` is not
            ``None``, otherwise ``None``.
        """
        input_dir = self.init.dict_folders["input"]
        run_dir = self.init.dict_folders["run"]

        self._CMDS_nc_to_ascii(cmds_filename, ascii_filename, lon_target, lat_target)
        print(f"\t CMDS wind data converted to ASCII format and saved as {ascii_filename}")

        utils.deploy_input_file(ascii_filename, input_dir, run_dir, self.use_link)

        if self.wind_info is not None:
            self.wind_info.update({"windfilepath": ascii_filename})
            return self.wind_info
        return None

    def fill_wind_section(self):
        """
        Write wind configuration into the XBeach params file.

        Raises
        ------
        ValueError
            No wind information is available in ``self.wind_info`` to fill the params file.
        """

        if self.wind_info == None:
            raise ValueError("No wind information available to fill in params file.")

        print('\n*** Adding/Editing winds information in params file ***\n')
        utils.fill_files(f'{self.init.dict_folders["run"]}params.txt', self.wind_info)
