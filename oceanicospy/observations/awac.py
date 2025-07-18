import glob
import pandas as pd
import numpy as np
import pandas as pd

pd.options.mode.chained_assignment = None

from ..utils import wave_props
import re

class AWAC():
    """
    A class to handle reading and processing the data files recorded by an ADCP AWAC (Nortek S.A). 

    Notes
    -----
    04-Jan-2018 : Origination - Daniel Peláez
    01-Sep-2023 : Migration to Python - Alejandro Henao
    10-Dec-2024 : Class implementation - Franklin Ayala

    """
    def __init__(self,directory_path,sampling_data):
        """
        Initializes the Awac class with the given directory path, sampling data.

        Parameters
        ----------
        directory_path : str
            Path to the directory containing the .hdr and .wad files.
        sampling_data : dict
            Dictionary containing the information about the device installation
        """
        self.directory_path = directory_path
        self.sampling_data = sampling_data

    def get_raw_wave_records(self,from_single_wad):
        """
        Reads and processes the .wad files to create a DataFrame containing the burst data.

        For each .wad file, the function reads the data, adds a 'burstId' column, and combines all the data into a single DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the concatenated data from all the .wad files with an added 'burstId' column.
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

            for wad_filepath in wad_files[1:]: #What's the differenc with reading the main .wad file?
                burst_df = pd.read_csv(wad_filepath,sep=r"\s+",names=column_names)
                burst_df.rename(columns={column_names[0]:'burstId'},inplace=True)
                burst_list.append(burst_df)

            df = pd.concat(burst_list, ignore_index=True)
        return df
    
    def get_clean_wave_records(self,from_single_wad=True):
        """
        Processes the raw data by converting certain columns to numeric types, adding a timestamp, and filtering the data 
        by the specified time range.

        The function also renames columns and returns a cleaned DataFrame that includes only relevant data.

        Returns
        -------
        pandas.DataFrame
            A cleaned DataFrame containing 'pressure', 'u', 'v', and 'burstId' columns, filtered by the specified time range.
        """
        df_raw = self.get_raw_wave_records(from_single_wad)

        if from_single_wad:
            df_clean = self._parse_dates_and_trim(df_raw)
            df_clean = self._rename_columns(df_clean)
        else:
            burst_start_times = pd.date_range(
                start=self.sampling_data['start_time'],
                end=self.sampling_data['end_time'],
                freq='1h')

            # For each burst, create a date range of 2Hz samples
            date_range = []
            for start_time in burst_start_times:
                burst_range = pd.date_range(start=start_time,periods=2048,freq='500ms')
                date_range.append(burst_range)

            # Concatenate all individual burst date ranges into a single DatetimeIndex
            full_index = pd.DatetimeIndex(np.concatenate([rng.values for rng in date_range]))
            df_clean = df_raw.set_index(full_index)
            df_clean = self._parse_dates_and_trim(df_clean)
            df_clean = self._rename_columns(df_clean)

        return df_clean

    def get_raw_currents_records(self):
        """
        Reads and processes the .v1 and .v2 files to create a DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the current magnitude and direction.
        """
        self.currents_header = self._read_currents_header()
        x_component_filepath = sorted(glob.glob(self.directory_path+'*.v1'))[0]
        y_component_filepath = sorted(glob.glob(self.directory_path+'*.v2'))[0]
        
        column_names = [f'{i}' for i in range(1, int(self.currents_header['Number of cells']) + 1)]
        x_component_df = pd.read_csv(x_component_filepath,sep=r'\s+',names=column_names)
        y_component_df = pd.read_csv(y_component_filepath,sep=r'\s+',header=None,names=column_names)

        return x_component_df, y_component_df

    def get_clean_currents_records(self,compute_speed_dir=True):
        
        x_component_raw, y_component_raw = self.get_raw_currents_records()
        date_range = pd.date_range(self.currents_header['start_time'],periods=x_component_raw.shape[0],
                                        freq=f"{self.currents_header['Profile interval']}s")

        x_component_clean = x_component_raw.set_index(date_range)
        y_component_clean = y_component_raw.set_index(date_range)
        x_component_clean = self._parse_dates_and_trim(x_component_clean)
        y_component_clean = self._parse_dates_and_trim(y_component_clean)

        if compute_speed_dir:
            current_speed = np.sqrt((x_component_clean**2)+(y_component_clean**2))
            current_dir = np.array([list(map(wave_props.angulo_norte,row_x,row_y)) for row_x,row_y in zip(self.x_component_raw.values,self.y_component_raw.values)])
            current_dir = pd.DataFrame(data=current_dir,index=date_range,columns=current_speed.columns)
            return x_component_clean,y_component_clean,current_speed,current_dir
        else:
            return x_component_clean,y_component_clean

    def _read_wave_header(self):
        """
        Reads and parses the header file (.hdr) to extract the column names.

        Returns
        -------
        numpy.ndarray
            An array of column names extracted from the .hdr file.
        """
        lines = self._load_hdr_lines()
        data_lines = self._extract_column_lines(lines)
        column_names = self._format_column_names(data_lines)
        return column_names

    def _read_currents_header(self):
        """
        Reads the header file (.hdr) to extract the column names for current data.

        Returns
        -------
        numpy.ndarray
            An array of column names extracted from the .hdr file.
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
        """Load lines from the .hdr file."""
        hdr_files = glob.glob(f"{self.directory_path}*.hdr")
        if not hdr_files:
            raise FileNotFoundError("No .hdr file found in the directory.")

        with open(hdr_files[0], 'r') as file:
            return file.read().splitlines()

    def _extract_column_lines(self, lines: list) -> list:
        """
        Extract relevant lines between '.wad]' and the next line ending in '-'.
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

    def _format_column_names(self, lines: list) -> np.ndarray:
        """Format the header lines into clean column names."""
        names = [' '.join(line.split()) for line in lines]
        if len(names) > 0 and names[-1] == '':
            names = names[:-1]
        return names

    def _parse_dates_and_trim(self, df: pd.DataFrame) -> pd.Series:
        """
        Helper function to parse date and time from separate columns.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with time columns as named columns (year, month, day, hour, minute, second).

        Returns
        -------
        pandas.Series
            Series of parsed datetime objects.
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
    
    def _rename_columns(self, df: pd.DataFrame,) -> pd.DataFrame:
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
        """

        relevant_columns = [column for column in df.columns if 'Pressure' in column or 'Velocity' in column or 'burst' in column]
        df = df[relevant_columns]

        renamed_columns = []
        for column in relevant_columns:
            if column.startswith('burst'):
                final_column = column
            else:
                column = column.lower().split(' ')[1:] 
                column[-1] = column[-1].replace('(','[').replace(')',']')
                final_column = ''.join(column).replace('(','_').replace(')','_')
            renamed_columns.append(final_column)
        df.columns = renamed_columns
            
        df["pressure[bar]"] = df["pressure[dbar]"] / 10.0
        cols = ["pressure[bar]"] + [col for col in df.columns if "pressure" not in col]
        
        return df[cols]