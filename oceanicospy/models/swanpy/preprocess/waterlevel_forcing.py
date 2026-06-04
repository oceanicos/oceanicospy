import pandas as pd
import numpy as np
import glob as glob
from datetime import datetime
from scipy.signal import detrend

from pathlib import Path

from .... import utils
from ....utils.waterlevel import download_uhslc_waterlevel, load_uhslc_waterlevel

class WaterLevelForcing:
    """
    WaterLevelForcing is a utility class for generating and managing the water level forcing information for
    SWAN.
    
    Parameters
    ----------
    init : object
        An initialization object containing configuration data and folder paths.
    domain_number : int
        Identifier for the domain being processed.
    wl_info : dict or None, optional
        Dictionary containing water level information. If None, water level data must be provided via `get_waterlevel_from_UHSLC()`.
    filename : str or None, optional
        Name of a existing water level ASCII file to create or link in the domain input directory. Defaults to None.
    share_wl : bool, optional
        If True, water level data is shared across domains by linking to the domain 1 file. Defaults to True.
    use_link: bool, optional
        If True, creates symbolic links for water level files instead of copying them. Defaults to True
    """
    def __init__ (self,init,domain_number,wl_info=None,filename=None,share_wl=True,use_link=None):
        self.init = init
        self.domain_number = domain_number
        self.wl_info = wl_info
        self.filename = filename
        self.share_wl = share_wl
        self.use_link = use_link
        print(f'\n*** Initializing water levels for domain {self.domain_number} ***\n')

    def _download_UHSLC(self, station_id, filepath):
        """
        Download UHSLC hourly sea-level data and return the cleaned DataFrame.

        Parameters
        ----------
        station_id : str
            UHSLC station code (e.g. ``"057"``).
        filepath : str
            Full path where the raw CSV will be saved.

        Returns
        -------
        pandas.DataFrame
            Cleaned water-level DataFrame with a datetime index and a
            ``depth[m]`` column.
        """
        return download_uhslc_waterlevel(
            station_id, self.init.ini_date, self.init.end_date, filepath
        )
    
    def _load_bathymetry(self) -> np.ndarray:
        """
        Load the bathymetry grid for the current domain from the ``*.bot`` file.

        Returns
        -------
        numpy.ndarray
            2-D array of water depths.  Positive values indicate wet cells;
            zero or negative values indicate dry/land cells.

        Raises
        ------
        IndexError
            If no ``*.bot`` file is found in the domain input directory.
        """
        bot_files = glob.glob(
            f'{self.init.dict_folders["input"]}domain_0{self.domain_number}/*.bot'
        )
        return np.genfromtxt(bot_files[0])

    def _build_wl_grid(self, bathymetry_grid: np.ndarray, wl_value: float) -> np.ndarray:
        """
        Create a 2-D water-level grid that matches the bathymetry layout.

        Wet cells (``bathymetry_grid >= 0``) are assigned *wl_value*; all
        other cells receive the SWAN no-data fill value ``-9999``.

        Parameters
        ----------
        bathymetry_grid : numpy.ndarray
            2-D depth array returned by :meth:`_load_bathymetry`.
        wl_value : float
            Water-surface elevation in metres to assign to wet cells.

        Returns
        -------
        numpy.ndarray
            2-D float array of the same shape as *bathymetry_grid*.
        """
        water_level = np.full(bathymetry_grid.shape, -9999.0)
        water_level[bathymetry_grid >= 0] = wl_value
        return water_level

    def _match_wl_to_bathy(self) -> np.ndarray:
        """
        Build a static water-level grid using the first value in the
        filtered water-level dataset.

        The method reads the domain bathymetry and stamps the first
        ``depth[m]`` value from :attr:`dataset_filtered` onto all wet
        cells, leaving dry/land cells at ``-9999``.

        Returns
        -------
        numpy.ndarray
            2-D float array with the static water-level field, shaped
            like the domain bathymetry grid.

        Raises
        ------
        AttributeError
            If :attr:`dataset_filtered` has not been populated yet
            (call :meth:`_UHSLC_csv_to_ascii` first).
        """
        bathymetry_grid = self._load_bathymetry()
        wl_value = float(self.dataset_filtered["depth[m]"].iloc[0])
        return self._build_wl_grid(bathymetry_grid, wl_value)

    def _UHSLC_csv_to_ascii(
        self,
        UHSLC_dataframe: pd.DataFrame,
        ascii_filename: str,
        detrend_wl: bool = False,
    ) -> None:
        """
        Convert a UHSLC water-level DataFrame to SWAN ASCII grid format.

        Processing steps:

        1. Flag UHSLC fill values (``depth[m] < -30.0``) as ``NaN``.
        2. Optionally detrend the water-level signal (NaN-safe).
        3. Trim the series to ``[ini_date, end_date]``.
        4. Write one timestamp header followed by a full 2-D water-level
           grid per time step to the domain input folder.

        The full series is stored in :attr:`dataset` and the
        simulation-window subset in :attr:`dataset_filtered`.

        Parameters
        ----------
        UHSLC_dataframe : pd.DataFrame
            Cleaned UHSLC DataFrame with a datetime index and a ``depth[m]``
            column, as returned by :meth:`get_waterlevel_from_UHSLC`.
        ascii_filename : str
            Name of the SWAN water-level ASCII output file to create in the
            domain input directory (e.g. ``"water_levels.wl"``).
        detrend_wl : bool, optional
            If ``True``, apply :func:`scipy.signal.detrend` to the
            ``depth[m]`` column.  ``NaN`` gaps are excluded from the
            detrending and reinserted afterwards.  Default is ``False``.

        Returns
        -------
        None

        Raises
        ------
        IndexError
            If no ``*.bot`` file is found in the domain input directory.
        """
        df = UHSLC_dataframe.copy()
        domain_dir = Path(self.init.dict_folders["input"]) / f"domain_0{self.domain_number}"

        df.loc[df["depth[m]"] < -30.0, "depth[m]"] = np.nan

        if detrend_wl:
            valid_mask = np.isfinite(df["depth[m]"])
            detrended = np.full(len(df), np.nan)
            detrended[valid_mask.values] = detrend(
                df.loc[valid_mask, "depth[m]"].values
            )
            df["depth[m]"] = detrended

        self.dataset = df
        self.dataset_filtered = df.loc[
            (df.index >= self.init.ini_date) & (df.index <= self.init.end_date)
        ]

        bathymetry_grid = self._load_bathymetry()
        out_path = domain_dir / ascii_filename
        with open(out_path, "w") as fh:
            for date, row in self.dataset_filtered.iterrows():
                fh.write(f"{date.strftime('%Y%m%d %H%M%S')}\n")
                water_level = self._build_wl_grid(
                    bathymetry_grid, float(row["depth[m]"])
                )
                np.savetxt(fh, water_level, fmt="%10.4f")

    def get_waterlevel_from_UHSLC(self,station_id):
        """
        Obtain water level data from the UHSLC for a given station ID, handling file management and optional sharing across domains.

        Parameters
        ----------
        station_id : int
            The UHSLC station ID for which to retrieve water level data.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the cleaned water level data for the specified station.
        """
        filepath = f"{self.init.dict_folders['input']}domain_0{self.domain_number}/h{station_id}.csv"
        file_exists = utils.verify_file(filepath)

        if not self.share_wl:
            if not file_exists:
                df_waterlevel = self._download_UHSLC(station_id, filepath=filepath)
            else:
                print("\t UHSLC water level data already exists, skipping download")
                df_waterlevel = load_uhslc_waterlevel(station_id, filepath)
        else:
            if self.domain_number == 1:
                if not file_exists:
                    df_waterlevel = self._download_UHSLC(station_id, filepath=filepath)
                else:
                    print("\t UHSLC water level data already exists, skipping download")
                    df_waterlevel = load_uhslc_waterlevel(station_id, filepath)
            else:
                filepath_domain1 = f"{self.init.dict_folders['input']}domain_01/h{station_id}.csv"
                print("\t UHSLC water level data already exists in domain 1, skipping download")
                df_waterlevel = load_uhslc_waterlevel(station_id, filepath_domain1)
        return df_waterlevel

    def write_UHSLC_ascii(self,UHSLC_dataframe,ascii_filename,detrend_wl=True):
        """
        Write the cleaned UHSLC water level data to a SWAN ASCII file, handling file management and optional sharing across domains.

        Parameters
        ----------
        UHSLC_dataframe : pd.DataFrame
            The cleaned UHSLC water level data as a pandas DataFrame.
        ascii_filename : str
            Name of the SWAN water-level ASCII output file to create in the same input directory (e.g. ``"water_levels.wl"``).
        detrend_wl : bool, optional
            If ``True``, apply linear detrending to the water level data before writing. Default is ``True``.
            
        Notes
        -----
        If *share_wl* is ``True``, the ASCII file is created in domain 1 and
        linked to other domains.  If ``False``, each domain gets its own ASCII
        file created from the UHSLC data.  In either case, ``wl_info`` is
        updated with the path to the written file and returned.
        """
        run_domain_dir = f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/'
        origin_domain_dir = f'{self.init.dict_folders["input"]}domain_0{self.domain_number}/'

        self._UHSLC_csv_to_ascii(UHSLC_dataframe, ascii_filename,detrend_wl)
        print('\t UHSLC water level data converted to ASCII format and saved as', ascii_filename)
        
        # validation of file creation and deployment to run directory
        utils.deploy_input_file(ascii_filename, origin_domain_dir, run_domain_dir, self.use_link)

        if self.wl_info!=None:
            self.wl_info.update({"wl_file":f"../../input/domain_0{self.domain_number}/{ascii_filename}"})
            return self.wl_info
        return None

    def fill_wl_section(self):
        """
        Replaces and updates the .swn file with the water level configuration for a specific domain.
        """

        if self.wl_info == None:
            raise ValueError(f'Water level information is not provided for domain {self.domain_number}.')

        print (f'\n \t*** Adding/Editing water level information for domain {self.domain_number} in configuration file ***\n')
        utils.fill_files(f'{self.init.dict_folders["run"]}domain_0{self.domain_number}/run.swn',self.wl_info)
