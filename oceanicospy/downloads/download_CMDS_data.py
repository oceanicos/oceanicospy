from datetime import datetime, timedelta
import os
import numpy as np
from pathlib import Path
import xarray as xr
import copernicusmarine
import shutil
import gc
import time
import psutil


class CMDSDownloader:
    """
    Downloader for oceanographic data from the Copernicus Marine Data Store (CMDS).

    Submits a spatial/temporal subset request via the ``copernicusmarine`` toolbox
    and optionally rewrites the resulting NetCDF in local time.

    Parameters
    ----------
    dataset_id : str
        CMDS dataset identifier (case-sensitive, as listed in the CMDS catalog).
    variables : list of str
        Variable names to request (dataset-specific names).
    lon_min : float
        Western boundary of the bounding box in degrees (EPSG:4326).
    lon_max : float
        Eastern boundary of the bounding box in degrees (EPSG:4326).
    lat_min : float
        Southern boundary of the bounding box in degrees (EPSG:4326).
    lat_max : float
        Northern boundary of the bounding box in degrees (EPSG:4326).
    start_datetime_local : datetime
        Start of the desired time window expressed in local time.
    end_datetime_local : datetime
        End of the desired time window expressed in local time.
    utc_offset_hours : float
        Local-time offset from UTC in hours, following the convention
        ``local - UTC``. For example, UTC-5 (Colombia) is ``-5``.
    output_path : str or Path
        Directory where the output file will be written.  Created automatically
        if it does not exist.
    output_filename : str, optional
        Name of the file to write inside ``output_path``
        (e.g. ``"winds_CMDS.nc"``).  When ``None`` the CMDS toolbox chooses
        a default name based on the dataset and request parameters.
    file_format : str, optional
        Output format accepted by the CMDS toolbox: ``"netcdf"`` (default)
        or ``"zarr"``.  :meth:`format_to_localtime` only supports ``"netcdf"``.
    """

    def __init__(
        self,
        dataset_id: str,
        variables: list[str],
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        start_datetime_local: datetime,
        end_datetime_local: datetime,
        utc_offset_hours: float,
        output_path: str | Path,
        output_filename: str | None = None,
        file_format: str = "netcdf",
    ) -> None:
        self.dataset_id = dataset_id
        self.variables = variables
        self.lon_min = lon_min
        self.lon_max = lon_max
        self.lat_min = lat_min
        self.lat_max = lat_max
        self.start_datetime_local = start_datetime_local
        self.end_datetime_local = end_datetime_local
        self.utc_offset_hours = utc_offset_hours
        self.output_path = Path(output_path)
        self.output_filename = output_filename
        self.file_format = file_format

        # Convert local datetimes to UTC; CMDS subset requests must use UTC.
        self.start_datetime_utc = start_datetime_local - timedelta(hours=utc_offset_hours)
        self.end_datetime_utc = end_datetime_local - timedelta(hours=utc_offset_hours)

        # Populated by download(); consumed by format_to_localtime.
        self.last_result_path: Path | None = None

    def download(self) -> Path:
        """
        Submit the CMDS subset request and write the result to ``output_path``.

        Creates the output directory if it does not exist, then calls
        ``copernicusmarine.subset`` with the spatial/temporal window
        configured at construction time.

        Returns
        -------
        Path
            Absolute path of the NetCDF file created by the CMDS toolbox.

        Raises
        ------
        copernicusmarine.exceptions.CopernicusMarineError
            If the toolbox rejects the request (invalid dataset, credentials, etc.).
        OSError
            If the output directory cannot be created.
        """
        self.output_path.mkdir(parents=True, exist_ok=True)

        subset_kwargs = dict(
            dataset_id=self.dataset_id,
            variables=self.variables,
            minimum_longitude=self.lon_min,
            maximum_longitude=self.lon_max,
            minimum_latitude=self.lat_min,
            maximum_latitude=self.lat_max,
            start_datetime=self.start_datetime_utc.isoformat(),
            end_datetime=self.end_datetime_utc.isoformat(),
            output_directory=str(self.output_path),
            file_format=self.file_format,
            overwrite=True
        )
        if self.output_filename is not None:
            subset_kwargs["output_filename"] = self.output_filename

        copernicusmarine.subset(**subset_kwargs)

        if self.output_filename is not None:
            abs_path = (self.output_path / self.output_filename).resolve()
        else:
            abs_path = self.output_path.resolve()

        self.last_result_path = abs_path

        label = self.output_filename or self.output_path.name
        print(f"Downloaded {label} to {self.output_path / self.output_filename}.")
        return abs_path

    def format_to_localtime(self) -> Path:
        """
        Shift the time coordinate from UTC to local time and write a new file.

        Reads the NetCDF produced by :meth:`download`, adds
        ``utc_offset_hours`` hours to every timestamp (converting UTC to
        local time), trims the dataset to
        ``[start_datetime_local, end_datetime_local]``, and saves the result
        to a new file whose name is the original stem suffixed with
        ``_localtime`` (e.g. ``winds_CMDS_localtime.nc``).

        Returns
        -------
        Path
            Absolute path of the newly written local-time NetCDF file.

        Raises
        ------
        ValueError
            If ``file_format`` is not ``"netcdf"``.
        KeyError
            If the dataset contains neither a ``"time"`` nor a
            ``"valid_time"`` coordinate.
        FileNotFoundError
            If the output file written by :meth:`download` cannot be located.

        Notes
        -----
        Zarr outputs are not supported; use ``file_format="netcdf"`` or
        implement a separate workflow for Zarr.
        """
        if self.file_format.lower() != "netcdf":
            raise ValueError(
                "format_to_localtime only supports NetCDF files. "
                "Use file_format='netcdf'."
            )
        target_nc = (self.output_path / self.output_filename).resolve()
            
        localtime_path = target_nc.with_name(target_nc.stem + "_localtime" + target_nc.suffix)
        ds_cropped = self._load_and_process(target_nc)
        ds_cropped.to_netcdf(localtime_path, mode="w", format="NETCDF4")
        return localtime_path

    def _load_and_process(self, path: Path) -> xr.Dataset:
        """Read a NetCDF file, shift its time coordinate to local time, and crop.

        Parameters
        ----------
        path : Path
            Path to the NetCDF file produced by :meth:`download`.

        Returns
        -------
        xr.Dataset
            In-memory dataset with timestamps expressed in local time and
            sliced to ``[start_datetime_local, end_datetime_local]``.
        """
        with xr.open_dataset(path, engine="netcdf4") as ds:
            if "valid_time" in ds.variables:
                tcoord = "valid_time"
            elif "time" in ds.variables:
                tcoord = "time"
            else:
                raise KeyError(
                    "No time coordinate found in dataset ('valid_time' or 'time')."
                )

            ds[tcoord] = ds[tcoord] + np.timedelta64(int(self.utc_offset_hours), "h")
            t0_local = np.datetime64(self.start_datetime_local)
            t1_local = np.datetime64(self.end_datetime_local)
            ds_cropped = ds.sel({tcoord: slice(t0_local, t1_local)}).load()

        return ds_cropped

    @classmethod
    def for_waves(
        cls,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        start_datetime_local: datetime,
        end_datetime_local: datetime,
        utc_offset_hours: float,
        output_path: str | Path,
        output_filename: str | None = None,
        file_format: str = "netcdf",
    ) -> "CMDSDownloader":
        """
        Create a :class:`CMDSDownloader` pre-configured for wave forecasts.

        Uses the CMEMS global wave analysis and forecast dataset
        ``cmems_mod_glo_wav_anfc_0.083deg_PT3H-i`` with variables
        ``VHM0`` (significant wave height), ``VMDR`` (mean wave direction),
        and ``VTPK`` (peak period).

        Parameters
        ----------
        lon_min : float
            Western boundary of the bounding box in degrees (EPSG:4326).
        lon_max : float
            Eastern boundary of the bounding box in degrees (EPSG:4326).
        lat_min : float
            Southern boundary of the bounding box in degrees (EPSG:4326).
        lat_max : float
            Northern boundary of the bounding box in degrees (EPSG:4326).
        start_datetime_local : datetime
            Start of the desired time window in local time.
        end_datetime_local : datetime
            End of the desired time window in local time.
        utc_offset_hours : float
            Local-time offset from UTC in hours, following the convention
            ``local - UTC``. For example, UTC-5 (Colombia) is ``-5``.
        output_path : str or Path
            Directory where the output file will be written.
        output_filename : str, optional
            Name of the output file (e.g. ``"waves_CMDS.nc"``).
        file_format : str, optional
            ``"netcdf"`` (default) or ``"zarr"``.

        Returns
        -------
        CMDSDownloader
            Instance configured for wave data.
        """
        return cls(
            dataset_id="cmems_mod_glo_wav_anfc_0.083deg_PT3H-i",
            variables=["VHM0", "VMDR", "VTPK"],
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            start_datetime_local=start_datetime_local,
            end_datetime_local=end_datetime_local,
            utc_offset_hours=utc_offset_hours,
            output_path=output_path,
            output_filename=output_filename,
            file_format=file_format,
        )

    @classmethod
    def for_winds(
        cls,
        lon_min: float,
        lon_max: float,
        lat_min: float,
        lat_max: float,
        start_datetime_local: datetime,
        end_datetime_local: datetime,
        utc_offset_hours: float,
        output_path: str | Path,
        output_filename: str | None = None,
        file_format: str = "netcdf",
    ) -> "CMDSDownloader":
        """
        Create a :class:`CMDSDownloader` pre-configured for surface wind analysis.

        Uses the CMEMS global near-real-time L4 wind dataset
        ``cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H`` with variables
        ``eastward_wind`` and ``northward_wind``.

        Parameters
        ----------
        lon_min : float
            Western boundary of the bounding box in degrees (EPSG:4326).
        lon_max : float
            Eastern boundary of the bounding box in degrees (EPSG:4326).
        lat_min : float
            Southern boundary of the bounding box in degrees (EPSG:4326).
        lat_max : float
            Northern boundary of the bounding box in degrees (EPSG:4326).
        start_datetime_local : datetime
            Start of the desired time window in local time.
        end_datetime_local : datetime
            End of the desired time window in local time.
        utc_offset_hours : float
            Local-time offset from UTC in hours, following the convention
            ``local - UTC``. For example, UTC-5 (Colombia) is ``-5``.
        output_path : str or Path
            Directory where the output file will be written.
        output_filename : str, optional
            Name of the output file (e.g. ``"winds_CMDS.nc"``).
        file_format : str, optional
            ``"netcdf"`` (default) or ``"zarr"``.

        Returns
        -------
        CMDSDownloader
            Instance configured for wind data.
        """
        return cls(
            dataset_id="cmems_obs-wind_glo_phy_nrt_l4_0.125deg_PT1H",
            variables=["eastward_wind", "northward_wind"],
            lon_min=lon_min,
            lon_max=lon_max,
            lat_min=lat_min,
            lat_max=lat_max,
            start_datetime_local=start_datetime_local,
            end_datetime_local=end_datetime_local,
            utc_offset_hours=utc_offset_hours,
            output_path=output_path,
            output_filename=output_filename,
            file_format=file_format,
        )