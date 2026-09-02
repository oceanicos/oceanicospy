import pandas as pd
import numpy as np
import glob as glob
from datetime import datetime
from pathlib import Path
from scipy.signal import detrend

from .... import utils
from ....utils.waterlevel import download_uhslc_waterlevel, load_uhslc_waterlevel


class WaterLevelForcing:
    """
    Preprocess water-level forcing files for XBeach.

    Parameters
    ----------
    init : object
        Project initialization object.  Must provide ``init.ini_date``,
        ``init.end_date``, and ``init.dict_folders`` with keys
        ``"input"`` and ``"run"``.
    wl_info : dict or None, optional
        Dictionary containing water-level metadata.  Updated by the write
        methods and consumed by :meth:`fill_wl_section`.
    input_filename : str or None, optional
        Name of a pre-existing water-level file in the input folder.
    use_link : bool or None, optional
        If ``True``, deploy water-level files as symbolic links.
        If ``False``, copy them physically.
        If ``None``, no file deployment is performed.
    """

    def __init__(self, init, wl_info=None, input_filename=None, use_link=None):
        self.init = init
        self.wl_info = wl_info
        self.input_filename = input_filename
        self.use_link = use_link
        print(f'\n*** Initializing water levels ***\n')

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

    def _UHSLC_csv_to_ascii(
        self,
        UHSLC_dataframe: pd.DataFrame,
        ascii_filename: str,
        detrend_wl: bool = False,
    ) -> None:
        """
        Convert a UHSLC water-level DataFrame to XBeach ASCII format.

        Processing steps:

        1. Flag UHSLC fill values (``depth[m] < -30.0``) as ``NaN``.
        2. Optionally detrend the water-level signal (NaN-safe).
        3. Trim the series to ``[ini_date, end_date]``.
        4. Write a two-column space-separated file ``(elapsed_seconds, depth_m)``
           to the input folder.

        The full series is stored in :attr:`dataset` and the simulation-window
        subset in :attr:`dataset_filtered`.

        Parameters
        ----------
        UHSLC_dataframe : pd.DataFrame
            Cleaned UHSLC DataFrame with a datetime index and a ``depth[m]``
            column, as returned by :meth:`get_waterlevel_from_UHSLC`.
        ascii_filename : str
            Name of the XBeach water-level ASCII output file to create in the
            input folder (e.g. ``"water_level.wl"``).
        detrend_wl : bool, optional
            If ``True``, apply :func:`scipy.signal.detrend` to the
            ``depth[m]`` column after the datum correction.  ``NaN`` gaps are
            excluded from the detrending and reinserted afterwards.
            Default is ``False``.

        Returns
        -------
        None
        """
        df = UHSLC_dataframe.copy()
        input_dir = Path(self.init.dict_folders["input"])

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

        time_to_write = (
            self.dataset_filtered.index - self.dataset_filtered.index[0]
        ).total_seconds().astype(int).tolist()

        df_to_save = pd.DataFrame(
            {"Time": time_to_write, "water_level[m]": self.dataset_filtered["depth[m]"]},
            index=self.dataset_filtered.index,
        )

        df_to_save["water_level[m]"] = df_to_save["water_level[m]"].round(3)

        df_to_save.to_csv(
            input_dir / ascii_filename,
            sep=" ", header=False, index=False,
        )

    def get_waterlevel_from_UHSLC(self, station_id, override=False):
        """
        Obtain UHSLC water-level data, downloading it only when necessary.

        Checks whether the raw CSV already exists in the input folder.
        If not (or when *override* is ``True``), downloads the data from
        the UHSLC server.

        Parameters
        ----------
        station_id : str
            UHSLC station code (e.g. ``"057"``).
        override : bool, optional
            If ``True``, re-downloads the file even if it already exists.
            Defaults to ``False``.

        Returns
        -------
        pandas.DataFrame
            Cleaned water-level DataFrame with a datetime index and a
            ``depth[m]`` column.
        """
        filepath = f"{self.init.dict_folders['input']}h{station_id}.csv"
        file_exists = utils.verify_file(filepath)

        if not file_exists or override:
            return self._download_UHSLC(station_id, filepath=filepath)
        
        print("\t UHSLC water level data already exists, skipping download")
        return load_uhslc_waterlevel(station_id, filepath)

    def write_UHSLC_ascii(self, UHSLC_dataframe, ascii_filename, detrend_wl=False):
        """
        Convert UHSLC water-level data to XBeach ASCII format and deploy it.

        Converts the cleaned UHSLC DataFrame to XBeach ASCII format and
        places the result in the run folder either as a symbolic link or a
        physical copy, depending on ``self.use_link``.

        Parameters
        ----------
        UHSLC_dataframe : pandas.DataFrame
            Cleaned UHSLC DataFrame as returned by
            :meth:`get_waterlevel_from_UHSLC`.
        ascii_filename : str
            Name of the ASCII output file to generate and deploy.
        detrend_wl : bool, optional
            If ``True``, detrend the water-level signal before writing.
            Defaults to ``False``.

        Returns
        -------
        dict
            Updated ``self.wl_info`` dictionary with key
            ``'sealevelfilepath'`` set to *ascii_filename*.
        """
        input_dir = self.init.dict_folders["input"]
        run_dir = self.init.dict_folders["run"]

        self._UHSLC_csv_to_ascii(UHSLC_dataframe, ascii_filename, detrend_wl)
        print('\t UHSLC water level data converted to ASCII format and saved as', ascii_filename)

        utils.deploy_input_file(ascii_filename, input_dir, run_dir, self.use_link)

        if self.wl_info != None:
            self.wl_info.update({"sealevelfilepath": ascii_filename})
        else:
            self.wl_info = {"sealevelfilepath": ascii_filename}

    def convert_data_from_CECOLDO(self, output_filename, input_filename=None):
        """
        Convert CECOLDO tide-gauge data to XBeach water-level format.

        Reads the tab-separated CECOLDO file, constructs a datetime index,
        removes mean water level, clips to the simulation period, and writes
        a two-column (time, sea-level) file to the run folder.

        Parameters
        ----------
        output_filename : str
            Name of the output file to write in the run folder.
        input_filename : str or None, optional
            Name of the CECOLDO tab-separated file in the input folder.
            Defaults to ``self.input_filename`` when ``None``.

        Raises
        ------
        ValueError
            If no input filename is available (neither passed nor set at init).

        Returns
        -------
        dict
            Dictionary with key ``'sealevelfilepath'`` set to
            ``'sealevel.txt'``.
        """
        if input_filename is None:
            input_filename = self.input_filename
        if input_filename is None:
            raise ValueError(
                "input_filename must be provided either at init or as a method argument."
            )
        data_cecoldo = pd.read_csv(
            f'{self.init.dict_folders["input"]}{input_filename}',
            sep='\t', skiprows=[1], na_values=['-99999.00'],
        )
        data_cecoldo['Fecha'] = pd.to_datetime(
            data_cecoldo['Fecha [aaaa-mm-dd UT-5]'] + ' ' +
            data_cecoldo['Hora [hh:mm:ss UT-5]']
        )
        data_cecoldo.set_index('Fecha', inplace=True)
        data_cecoldo.drop(
            columns=['Fecha [aaaa-mm-dd UT-5]', 'Hora [hh:mm:ss UT-5]',
                     'Latitud [deg]', 'Longitud [deg]', 'Estacion [#]', 'QF [IODE]'],
            inplace=True,
        )

        data_cecoldo = data_cecoldo[
            (data_cecoldo.index >= self.init.ini_date) &
            (data_cecoldo.index <= self.init.end_date)
        ]
        data_cecoldo['Nivel_mar [m]'] = (
            data_cecoldo['Nivel_mar [m]'] - np.nanmean(data_cecoldo['Nivel_mar [m]'])
        )

        time_to_write = (
            data_cecoldo.index - data_cecoldo.index[0]
        ).total_seconds().astype(int).tolist()
        df_to_save = pd.DataFrame(
            {'time [s]': time_to_write, 'sealevel': data_cecoldo['Nivel_mar [m]']},
            index=None,
        )
        df_to_save = df_to_save.round(3)
        df_to_save.to_csv(
            f'{self.init.dict_folders["run"]}{output_filename}',
            sep='\t', header=None, index=None,
        )

        self.wl_info = {'sealevelfilepath': 'sealevel.txt'}

    def constant_from_CECOLDO(self, input_filename=None):
        """
        Extract a single constant water-level value from a CECOLDO file.

        Reads the CECOLDO tab-separated file, removes the mean, clips to
        the simulation period, and returns the first value as a constant
        sea-level forcing.

        Parameters
        ----------
        input_filename : str or None, optional
            Name of the CECOLDO tab-separated file in the input folder.
            Defaults to ``self.input_filename`` when ``None``.

        Raises
        ------
        ValueError
            If no input filename is available (neither passed nor set at init).

        Returns
        -------
        dict
            Dictionary with key ``'sealevelvalue'`` set to the first
            de-meaned water-level value (rounded to three decimal places).
        """
        if input_filename is None:
            input_filename = self.input_filename
        if input_filename is None:
            raise ValueError(
                "input_filename must be provided either at init or as a method argument."
            )
        data_cecoldo = pd.read_csv(
            f'{self.init.dict_folders["input"]}{input_filename}',
            sep='\t', skiprows=[1], na_values=['-99999.00'],
        )
        data_cecoldo['Fecha'] = pd.to_datetime(
            data_cecoldo['Fecha [aaaa-mm-dd UT-5]'] + ' ' +
            data_cecoldo['Hora [hh:mm:ss UT-5]']
        )
        data_cecoldo.set_index('Fecha', inplace=True)
        data_cecoldo.drop(
            columns=['Fecha [aaaa-mm-dd UT-5]', 'Hora [hh:mm:ss UT-5]',
                     'Latitud [deg]', 'Longitud [deg]', 'Estacion [#]', 'QF [IODE]'],
            inplace=True,
        )
        data_cecoldo = data_cecoldo[
            (data_cecoldo.index >= self.init.ini_date) &
            (data_cecoldo.index <= self.init.end_date)
        ]
        data_cecoldo['Nivel_mar [m]'] = (
            data_cecoldo['Nivel_mar [m]'] - np.nanmean(data_cecoldo['Nivel_mar [m]'])
        )

        self.wl_info = {'sealevelvalue': round(data_cecoldo['Nivel_mar [m]'].values[0], 3)}

    def fill_wl_section(self):
        """
        Write water-level configuration into the XBeach params file.

        Uses ``self.wl_info`` populated by the write methods.

        Raises
        ------
        ValueError
            If no water-level information is available in ``self.wl_info``.
        """
        if self.wl_info is None:
            raise ValueError("No water level information available to fill in params file.")

        print('\n*** Adding/Editing water level information in params file ***\n')
        utils.fill_files(f'{self.init.dict_folders["run"]}params.txt', self.wl_info)
