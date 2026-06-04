import requests
import pandas as pd
from datetime import datetime
from pathlib import Path

class UHSLCDownloader:
    """
    Downloader for hourly sea-level data from the University of Hawaii Sea Level Center (UHSLC).

    Fetches Fast-Delivery CSV files from the UHSLC public data server and writes
    them to a user-specified output directory.

    Parameters
    ----------
    station_id : str
        UHSLC station code (e.g. ``"057"``). Used to build the filename
        ``h<station_id>.csv`` as published on the UHSLC server.
    output_path : str or Path
        Directory where the downloaded file will be saved.
    output_filename : str or Path
        Name of the output CSV file (e.g. ``"station_057.csv"``). The full path will be ``output_path / output_filename``.
    start_datetime_local : datetime, optional
        Start of the time range to keep after cleaning
        (e.g. ``datetime(2010, 1, 1)``). If ``None`` the series is not trimmed
        at the start.
    end_datetime_local : datetime, optional
        End of the time range to keep after cleaning
        (e.g. ``datetime(2020, 12, 31)``). If ``None`` the series is not trimmed
        at the end.
    utc_offset_hours : float, default -5
        Local-time offset from UTC in hours, following the convention
        ``local - UTC``. Used to convert the raw UHSLC timestamps (published
        in UTC) to local time in the cleaned DataFrame. For example, UTC-5
        (Colombia) is ``-5``. The default assumes the package's primary
        Colombian audience; users in other time zones must set this explicitly.
    """

    BASE_URL = "https://uhslc.soest.hawaii.edu/data/csv/fast/hourly/"
    def __init__(self, station_id: str, output_path: str | Path, output_filename: str | Path,
                 start_datetime_local: datetime | None = None,
                 end_datetime_local: datetime | None = None,
                 utc_offset_hours: float = -5) -> None:
        self.station_id = station_id
        self.output_path = Path(output_path)
        self.output_filename = Path(output_filename)
        self.start_datetime_local = start_datetime_local
        self.end_datetime_local = end_datetime_local
        self.utc_offset_hours = utc_offset_hours
        self.last_result_path: Path | None = None
        
    def download(self) -> Path:
        """
        Download the hourly sea-level CSV for the configured station.

        Sends a GET request to the UHSLC Fast-Delivery server and writes
        the response content to ``output_path / output_filename``.
        
        Returns
        -------
        Path
            Absolute path of the saved CSV file.

        Raises
        ------
        requests.exceptions.ConnectionError
            If the UHSLC server is unreachable.
        requests.exceptions.Timeout
            If the request exceeds the 60-second timeout.
        requests.HTTPError
            If the server returns a non-200 status code.
        OSError
            If the output directory cannot be created or the file cannot be written.
        """
        filename = f"h{self.station_id}.csv"
        file_url = self.BASE_URL + filename

        response = requests.get(file_url, timeout=60)
        response.raise_for_status()

        self.output_path.mkdir(parents=True, exist_ok=True)
        dest = self.output_path / self.output_filename
        dest.write_bytes(response.content)

        self.last_result_path = dest

        print(f"Downloaded {self.output_filename} to {dest}.")
        return dest
    
    def clean_data(self) -> pd.DataFrame:
        """
        Load and clean a downloaded UHSLC CSV file.

        Parses the raw UHSLC hourly format (year, month, day, hour, depth in mm),
        converts depth to metres, and optionally trims the time series to the
        ``[start_datetime_local, end_datetime_local]`` window provided at construction time.

        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by datetime with a single column ``depth[m]``. 
            Timestamps are in local time (UTC shifted by ``utc_offset_hours`` hours).
            If ``start_datetime_local`` or ``end_datetime_local`` were set on the instance, the
            returned series is trimmed accordingly; otherwise the full record
            is returned.
        """
        df = pd.read_csv(
            self.last_result_path,
            header=None,
            names=["year", "month", "day", "hour", "depth[mm]"],
            sep=",",
        )

        df.index = pd.to_datetime(df[["year", "month", "day", "hour"]])
        df.index = df.index + pd.to_timedelta(self.utc_offset_hours, unit="h")
        df["depth[m]"] = df["depth[mm]"] / 1000.0
        df = df.drop(columns=["year", "month", "day", "hour", "depth[mm]"])

        if self.start_datetime_local is not None:
            df = df.loc[self.start_datetime_local:]
        if self.end_datetime_local is not None:
            df = df.loc[:self.end_datetime_local]
        return df