import contextlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Generator

import cdsapi
import numpy as np
import xarray as xr


class ERA5Downloader:
    """
    Downloader for atmospheric reanalysis data from the ERA5 dataset
    (ECMWF Reanalysis v5) via the Copernicus Climate Data Store (CDS) API.

    Submits a single-levels reanalysis subset request and optionally rewrites
    the resulting NetCDF in local time.

    Parameters
    ----------
    variables : list of str
        ERA5 variable names to request (as listed in the CDS catalog,
        e.g. ``["10m_u_component_of_wind", "10m_v_component_of_wind"]``).
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
        Name of the NetCDF file to write inside ``output_path``
        (e.g. ``"winds_ERA5.nc"``).  Required by :meth:`download`; if ``None``
        a :py:exc:`ValueError` is raised at download time.
    cdsapi_rc : str or Path, optional
        Path to a ``.cdsapirc`` credentials file.  When provided, the file is
        used instead of the default ``~/.cdsapirc`` for the duration of the
        download call only; it is never logged or exposed publicly.  If
        ``None``, credentials are resolved from the ``$CDSAPI_RC`` environment
        variable or ``~/.cdsapirc``; a :py:exc:`FileNotFoundError` is raised
        if neither is found.
    """

    def __init__(
        self,
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
        cdsapi_rc: str | Path | None = None,
    ) -> None:
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
        self._cdsapi_rc = Path(cdsapi_rc) if cdsapi_rc is not None else None

        # Convert local datetimes to UTC; CDS API requests must use UTC.
        self.start_datetime_utc = start_datetime_local - timedelta(hours=utc_offset_hours)
        self.end_datetime_utc = end_datetime_local - timedelta(hours=utc_offset_hours)

        # Populated by download(); consumed by format_to_localtime.
        self.last_result_path: Path | None = None
        
    @contextlib.contextmanager
    def _cdsapi_credentials(self) -> Generator[None, None, None]:
        """
        Context manager that resolves and temporarily activates CDS API credentials.

        If a custom ``cdsapi_rc`` path was provided at construction time, it is
        validated and set as ``CDSAPI_RC`` for the duration of the block, then
        restored on exit.  If no custom path was given, the default resolution
        order is validated (``$CDSAPI_RC`` env var, then ``~/.cdsapirc``); a
        :py:exc:`FileNotFoundError` is raised if neither exists.

        Yields
        ------
        None

        Raises
        ------
        FileNotFoundError
            If ``cdsapi_rc`` was provided but does not exist, or if no
            credentials are found at the default locations.
        """
        if self._cdsapi_rc is not None:
            if not self._cdsapi_rc.is_file():
                raise FileNotFoundError(
                    f"Credentials file not found: {self._cdsapi_rc}"
                )
            previous = os.environ.get("CDSAPI_RC")
            os.environ["CDSAPI_RC"] = str(self._cdsapi_rc)
            try:
                yield
            finally:
                if previous is None:
                    os.environ.pop("CDSAPI_RC", None)
                else:
                    os.environ["CDSAPI_RC"] = previous
            return

        default_rc = Path.home() / ".cdsapirc"
        if os.environ.get("CDSAPI_RC") is None and not default_rc.is_file():
            raise FileNotFoundError(
                f"No CDS API credentials found: $CDSAPI_RC is not set in the "
                f"environment and {default_rc} does not exist."
            )
        yield

    def _prepare_datetime_data(self) -> tuple[list[str], list[str], list[str]]:
        """
        Build the year, month, and day lists required by the CDS API request.

        Iterates from :attr:`start_datetime_utc` to :attr:`end_datetime_utc`
        (inclusive, extended by one day to cover partial trailing days),
        collecting unique zero-padded strings for each calendar component.

        Returns
        -------
        years : list of str
            Sorted unique year strings (e.g. ``["2020", "2021"]``).
        months : list of str
            Sorted unique zero-padded month strings (e.g. ``["01", "12"]``).
        days : list of str
            Sorted unique zero-padded day strings (e.g. ``["01", "15", "31"]``).
        """
        current = self.start_datetime_utc
        days_by_month: dict[tuple[int, str], list[str]] = {}
        while current <= self.end_datetime_utc + timedelta(days=1):
            month_str = current.strftime("%m")
            days_by_month.setdefault((current.year, month_str), []).append(current.strftime("%d"))
            current += timedelta(days=1)

        years = sorted({str(y) for y, _ in days_by_month})
        months = sorted({m for _, m in days_by_month})
        days = sorted({d for day_list in days_by_month.values() for d in day_list})
        return years, months, days

    def download(self) -> Path:
        """
        Submit the CDS reanalysis request and write the result to ``output_path``.

        Creates the output directory if it does not exist, then calls
        ``cdsapi.Client().retrieve`` for the ``reanalysis-era5-single-levels``
        product with all 24 hourly time steps per day and a 0.025° grid.

        Returns
        -------
        Path
            Absolute path of the NetCDF file created by the CDS API.

        Raises
        ------
        ValueError
            If ``output_filename`` is ``None``.
        FileNotFoundError
            If credentials cannot be resolved (see :meth:`_cdsapi_credentials`).
        cdsapi.api.AmbiguousParameter
            If a variable name is not recognised by the CDS catalog.
        OSError
            If the output directory cannot be created or the file cannot be written.
        """
        if self.output_filename is None:
            raise ValueError(
                "output_filename must be set before calling download(). "
                "Pass a filename such as 'winds_ERA5.nc' to the constructor."
            )

        self.output_path.mkdir(parents=True, exist_ok=True)
        dest = self.output_path / self.output_filename

        years, months, days = self._prepare_datetime_data()

        with self._cdsapi_credentials():
            client = cdsapi.Client()
            client.retrieve(
                "reanalysis-era5-single-levels",
                {
                    "product_type": "reanalysis",
                    "format": "netcdf",
                    "variable": self.variables,
                    "year": years,
                    "month": months,
                    "day": days,
                    "time": [f"{h:02d}:00" for h in range(24)],
                    "area": [self.lat_max, self.lon_min, self.lat_min, self.lon_max],  # N, W, S, E
                    "grid": [0.025, 0.025],
                },
                str(dest),
            )

        self.last_result_path = dest.resolve()

        print(f"Downloaded {self.output_filename} to {self.output_path / self.output_filename}.")
        return self.last_result_path

    def format_to_localtime(self) -> Path:
        """
        Shift the time coordinate from UTC to local time and write a new file.

        Reads the NetCDF produced by :meth:`download`, adds
        ``utc_offset_hours`` hours to every timestamp (converting UTC to
        local time), trims the dataset to
        ``[start_datetime_local, end_datetime_local]``, and saves the result
        to a new file whose name is the original stem suffixed with
        ``_localtime`` (e.g. ``winds_ERA5_localtime.nc``).

        Returns
        -------
        Path
            Absolute path of the newly written local-time NetCDF file.

        Raises
        ------
        ValueError
            If ``output_filename`` is ``None``.
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
        if self.output_filename is None:
            raise ValueError(
                "output_filename must be set before calling format_to_localtime()."
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