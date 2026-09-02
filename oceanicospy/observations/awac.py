import glob
import re
import numpy as np
import pandas as pd

from ..utils import wave_props

class AWAC:
    """
    Handle reading and processing data files recorded by an ADCP AWAC (Nortek S.A.).
 
    Parameters
    ----------
    directory_path : str
        Path to the directory containing the ``.hdr`` and ``.wad`` files.
        Must end with a path separator (e.g. ``'data/awac/'``).
    sampling_data : dict
        Dictionary containing information about the device installation.
        Must include the keys ``'start_time'`` and ``'end_time'``, both
        parseable by :func:`pandas.to_datetime`.

    Notes
    -----
    **Development history**
    
    - 04-Jan-2018 : Origination - Daniel Peláez
    - 01-Sep-2023 : Migration to Python - Alejandro Henao
    - 10-Dec-2024 : Class implementation - Franklin Ayala
  
    """

    def __init__(self,directory_path,sampling_data):
        self.directory_path = directory_path
        self.sampling_data = sampling_data

    def _read_wave_header(self):
        """
        Reads and parses the header file (.hdr) to extract the column names.

        Returns
        -------
        numpy.ndarray
            An array of column names extracted from the ``.hdr`` file.
        """
        lines = self._load_hdr_lines()
        data_lines = self._extract_column_lines(lines)
        column_names = self._format_column_names(data_lines)
        return column_names
    
    def _read_wave_setup(self):
        """
        Read wave acquisition parameters from the ``.hdr`` file.
 
        Extracts the burst interval, number of samples per burst, and the
        sampling rate from the header file.
 
        Returns
        -------
        dict
            Dictionary with the following keys:
            - ``'interval'`` : int
                Burst interval in seconds.
            - ``'samples'`` : int
                Number of samples per burst.
            - ``'sampling_rate'`` : int
                Sampling rate in Hz.
        """
        lines = self._load_hdr_lines()
        filtered_wave_lines = []
        for line in lines:
            if any(keyword in line for keyword in ['Wave - Interval','Wave - Number of samples','Wave - Sampling rate']):
                filtered_wave_lines.append(line)
        dict_wave_setup = {}

        key_map = {'interval': 'interval','number of samples': 'samples',
                    'sampling rate': 'sampling_rate'}

        for item in filtered_wave_lines:
            item_lower = item.lower()
        
            # Extract numeric value
            value = int(re.search(r'\d+', item).group())
            
            # Match key
            for k in key_map:
                if k in item_lower:
                    dict_wave_setup[key_map[k]] = value
                    break
        return dict_wave_setup

    def _read_currents_header(self):
        """
        Reads the current-profile metadata from the header file (.hdr) to extract the column names for current data.

        Returns
        -------
        dict
            Dictionary with the following keys (among others, depending on
            the ``.hdr`` content):

            - ``'start_time'`` : pandas.Timestamp
                Timestamp of the first current profile measurement.
            - ``'Profile interval'`` : float
                Time between successive profiles in seconds.
            - ``'Number of cells'`` : float
                Number of depth cells.
            - ``'Cell size'`` : float
                Size of each depth cell in metres.
            - ``'Blanking distance'`` : float
                Blanking distance in metres.
        """
        lines = self._load_hdr_lines()
        filtered_lines = []
        for line in lines:
            if any(keyword in line for keyword in ['first measurement', 'Profile interval', 'Number of cells', 'Cell size', 'Blanking distance']):
                filtered_lines.append(line)
        
        dict_current_header = {}
        for line in filtered_lines[:-2]:
            # Split each line into key and value by the first occurrence of two or more spaces
            match = re.split(r'\s{2,}', line.strip(), maxsplit=1)

            if match and len(match) == 2:
                key = match[0]
                value = match[1]
                if 'first measurement' in key:
                    key = 'start_time'
                    value = pd.to_datetime(value, format='%m/%d/%Y %I:%M:%S %p')
                else:
                    numeric_value = re.findall(r"[-+]?\d*\.\d+|\d+", value)
                    value=float(numeric_value[0])
                dict_current_header[key] = value

        return dict_current_header

    def _load_hdr_lines(self) -> list:
        """
        Load all lines from the ``.hdr`` file in ``directory_path``.
 
        Returns
        -------
        list of str
            Lines of the ``.hdr`` file, without trailing newline characters.
 
        Raises
        ------
        FileNotFoundError
            If no ``.hdr`` file is found in ``directory_path``.
        """
        hdr_files = glob.glob(f"{self.directory_path}*.hdr")
        if not hdr_files:
            raise FileNotFoundError("No .hdr file found in the directory.")

        with open(hdr_files[0], 'r') as file:
            return file.read().splitlines()

    def _extract_column_lines(self, lines: list) -> list:
        """
        Extract the column-definition lines from the ``.hdr`` file.
 
        Reads lines that fall between the marker ending in ``'.wad]'`` and
        the next line ending in ``'-'``.
 
        Parameters
        ----------
        lines : list of str
            All lines from the ``.hdr`` file, as returned by
            :meth:`_load_hdr_lines`.
 
        Returns
        -------
        list of str
            Lines that correspond to column definitions in the ``.wad`` file.        
        """
        is_reading = False
        mask = []

        for line in lines:
            if is_reading:
                if line.endswith('-'):
                    is_reading = False
                    mask.append(False)
                else:
                    mask.append(True)
            else:
                mask.append(False)

            if line.endswith('.wad]'):
                is_reading = True

        return list(np.array(lines)[mask])

    def _format_column_names(self, lines: list) -> list:
        """        
        Normalise raw header lines into clean column name strings.
 
        Collapses consecutive whitespace within each line to a single space
        and removes any trailing empty entry.
 
        Parameters
        ----------
        lines : list of str
            Raw column-definition lines from the ``.hdr`` file, as returned
            by :meth:`_extract_column_lines`.
 
        Returns
        -------
        list
            A list of cleaned column name strings, ready for use as DataFrame column names
        """
        names = [' '.join(line.split()) for line in lines]
        if names and names[-1] == '':
            names = names[:-1]
        return names

    def _parse_dates_and_trim(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a datetime index and filter rows to the deployment window.
 
        If the DataFrame index is not already a :class:`pandas.DatetimeIndex`,
        the columns ``['year', 'month', 'day', 'hour', 'minute', 'second']``
        are combined into a single ``'date'`` column that becomes the index.
        The resulting time series is then sliced to the interval
        ``[start_time, end_time]`` defined in ``self.sampling_data``.
 
        Parameters
        ----------
        df : pandas.DataFrame
            Input data. Must either have a :class:`pandas.DatetimeIndex` or
            contain the columns ``year``, ``month``, ``day``, ``hour``,
            ``minute``, and ``second``.
 
        Returns
        -------
        pandas.DataFrame
            DataFrame indexed by datetime, trimmed to the deployment window.
 
        Raises
        ------
        KeyError
            If ``self.sampling_data`` does not contain ``'start_time'`` or
            ``'end_time'``.
        ValueError
            If either time value in ``self.sampling_data`` cannot be parsed
            by :func:`pandas.to_datetime`.
        """

        # Check if the DataFrame index is already a DatetimeIndex
        if isinstance(df.index, pd.DatetimeIndex):
            pass
        else:
            date_columns = ['year', 'month', 'day', 'hour', 'minute', 'second']   
            df['date'] = pd.to_datetime(df[date_columns])   
            df.drop(columns=date_columns, inplace=True, errors='ignore')
            df = df.set_index('date')

        try:
            start = pd.to_datetime(self.sampling_data['start_time'])
            end = pd.to_datetime(self.sampling_data['end_time'])
        except KeyError:
            raise KeyError("Missing 'start_time' or 'end_time' in sampling_data.")
        except Exception as e:
            raise ValueError(f"Invalid time format in 'sampling_data': {e}")

        return df[start:end]
    
    def _rename_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Renames columns in the DataFrame to a more readable format.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with columns to be renamed.

        Returns
        -------
        pandas.DataFrame
            DataFrame with renamed columns.
        
        Notes
        -----
        Pressure columns are converted from dbar to bar and renamed to 'pressure[bar]'.
        """

        relevant_columns = [column for column in df.columns if 'Pressure' in column or 'Velocity' in column or 'burst' in column]
        df = df[relevant_columns].copy()

        renamed_columns = []
        for column in relevant_columns:
            if column.startswith('burst'):
                final_column = column
            else:
                column = column.lower().split(' ')[1:] 
                column[-1] = column[-1].replace('(','[').replace(')',']')
                final_column = ''.join(column).replace('(','').replace(')','')
            renamed_columns.append(final_column)
        df.columns = renamed_columns
            
        df["pressure[bar]"] = df["pressure[dbar]"] / 10.0
        cols = ["pressure[bar]"] + [col for col in df.columns if "pressure" not in col]
        
        return df[cols]

    def get_raw_wave_records(self,from_single_wad=True):
        """
        Read ``.wad`` file(s) and return a raw burst DataFrame.
 
        Parameters
        ----------
        from_single_wad : bool, optional
            If ``True`` (default), read all burst data from the first
            ``.wad`` file found in ``directory_path``.  The file must
            therefore contain every burst.
 
            If ``False``, read one burst per ``.wad`` file (files sorted
            alphabetically).  Each file must be named so that sorting
            preserves chronological order.
 
        Returns
        -------
        pandas.DataFrame
            Raw data from the ``.wad`` file(s).
        """

        column_names = self._read_wave_header()
        wad_files = sorted(glob.glob(self.directory_path+'*.wad')) #Each .wad file represents one burst

        if from_single_wad:
            wad_filepath = wad_files[0]
            date_columns = ['month', 'day', 'year', 'hour', 'minute', 'second']
            df = pd.read_csv(wad_filepath,sep=r"\s+",names=date_columns+list(column_names[2:]))
            df = df.dropna()          
        else:
            burst_list = []

            for wad_filepath in wad_files[1:]: 
                burst_df = pd.read_csv(wad_filepath,sep=r"\s+",names=column_names)
                burst_df.rename(columns={column_names[0]:'burstId'},inplace=True)
                burst_list.append(burst_df)

            df = pd.concat(burst_list, ignore_index=True)
        return df
    
    def get_clean_wave_records(self,from_single_wad=True):
        """
        Return a cleaned, time-indexed wave DataFrame trimmed to the deployment window.
 
        Builds on :meth:`get_raw_wave_records` by parsing the datetime index,
        assigning burst identifiers, renaming columns, and deriving the
        pressure in bar.
 
        Parameters
        ----------
        from_single_wad : bool, optional
            Passed directly to :meth:`get_raw_wave_records`.  Default is
            ``True``.
 
        Returns
        -------
        pandas.DataFrame
            Cleaned DataFrame indexed by :class:`pandas.DatetimeIndex` and
            containing (at minimum) the columns:
 
            - ``'pressure[bar]'`` — absolute pressure in bar.
            - Velocity components with units in square brackets.
            - ``'burstId'`` — integer burst identifier (1-based).
 
        Notes
        -----
        When ``from_single_wad`` is ``True``, burst IDs are assigned by
        flooring the sample timestamps to the nearest hour.
 
        When ``from_single_wad`` is ``False``, timestamps are reconstructed
        from the wave setup parameters (interval, number of samples, sampling
        rate) and the deployment start/end times in ``self.sampling_data``
        """
        df_raw = self.get_raw_wave_records(from_single_wad)

        if from_single_wad:
            df_clean = self._parse_dates_and_trim(df_raw)
            df_clean['burstId'] = pd.factorize(df_clean.index.floor('h'))[0] + 1
            df_clean = self._rename_columns(df_clean)
        else:
            wave_setup = self._read_wave_setup()

            burst_start_times = pd.date_range(
                start=self.sampling_data['start_time'],
                end=self.sampling_data['end_time'],
                freq=f'{wave_setup["interval"]//3600}h')
            
            # For each burst, create a date range of 2Hz samples
            date_range = []
            for start_time in burst_start_times:
                burst_range = pd.date_range(start=start_time,periods=wave_setup['samples'],freq=f"{1000//wave_setup['sampling_rate']}ms")
                date_range.append(burst_range)

            # Concatenate all individual burst date ranges into a single DatetimeIndex
            full_index = pd.DatetimeIndex(np.concatenate([rng.values for rng in date_range]))
            df_clean = df_raw.set_index(full_index)
            df_clean = self._parse_dates_and_trim(df_clean)
            df_clean = self._rename_columns(df_clean)

        return df_clean

    def get_raw_currents_records(self):
        """
        Read the ``.v1`` and ``.v2`` files containing the east and north
        current-velocity components.
 
        Returns
        -------
        x_component_df : pandas.DataFrame
            East (x) velocity component for each depth cell.
            Columns are labelled ``'cell_1'``, ``'cell_2'``, … .
        y_component_df : pandas.DataFrame
            North (y) velocity component for each depth cell.
            Columns share the same labelling scheme.
        """
        self.currents_header = self._read_currents_header()
        x_component_filepath = sorted(glob.glob(self.directory_path+'*.v1'))[0]
        y_component_filepath = sorted(glob.glob(self.directory_path+'*.v2'))[0]
        
        column_names = [f'cell_{i}' for i in range(1, int(self.currents_header['Number of cells']) + 1)]
        x_component_df = pd.read_csv(x_component_filepath,sep=r'\s+',names=column_names)
        y_component_df = pd.read_csv(y_component_filepath,sep=r'\s+',header=None,names=column_names)

        return x_component_df, y_component_df

    def get_clean_currents_records(self,compute_speed_dir=True):
        """
        Processes the raw current data by reading the x and y components, setting the index to a date range, and optionally computing speed and direction.

        Parameters
        ----------
        compute_speed_dir : bool, optional
            If True, computes the current speed (magnitude) and direction from the x and y components. Default is True.

        Returns
        -------
        x_component_clean : pandas.DataFrame
            Cleaned DataFrame of the x component of the current.
        y_component_clean : pandas.DataFrame
            Cleaned DataFrame of the y component of the current.
        current_speed : pandas.DataFrame, optional
            DataFrame of the current speed for each depth cell, only returned if compute_speed_dir is True.
        current_dir : pandas.DataFrame, optional
            DataFrame of the current direction for each depth cell, only returned if compute_speed_dir is True.
        

        Notes
        -----
        Direction is computed via :func:`wave_props.angulo_norte`, which
        returns the angle measured clockwise from true north.
        """
        
        x_component_raw, y_component_raw = self.get_raw_currents_records()
        date_range = pd.date_range(self.currents_header['start_time'],periods=x_component_raw.shape[0],
                                        freq=f"{self.currents_header['Profile interval']}s")
        
        x_component_clean = x_component_raw.set_index(date_range)
        y_component_clean = y_component_raw.set_index(date_range)
        x_component_clean = self._parse_dates_and_trim(x_component_clean)
        y_component_clean = self._parse_dates_and_trim(y_component_clean)

        if compute_speed_dir:
            current_speed = np.sqrt((x_component_clean**2)+(y_component_clean**2))
            #TODO : Check if the angle is being calculated correctly
            current_dir = np.array([list(map(wave_props.angulo_norte,row_x,row_y)) for row_x,row_y in zip(x_component_raw.values,y_component_raw.values)])
            current_dir = pd.DataFrame(data=current_dir,index=date_range,columns=current_speed.columns)
            return x_component_clean,y_component_clean,current_speed,current_dir
        else:
            return x_component_clean,y_component_clean