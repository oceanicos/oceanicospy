import pandas as pd
import glob
import os

from .pressure_sensor_base import BaseLogger

class AQUAlogger(BaseLogger):
    """
    A sensor-specific reader for AQUAlogger CSV files.

    Inherits from
    -------------
    BaseLogger
        Includes core methods like `first_record_time`, `get_raw_records`, etc.
    """
    def _get_records_file(self):
        files = glob.glob(os.path.join(self.directory_path, '*.csv'))
        if not files:
            raise FileNotFoundError("No .csv file found in the specified directory.")
        return files[0]
    
    def _load_raw_dataframe(self) -> pd.DataFrame:
        filepath = self._get_records_file()
        if self.sampling_data.get('temperature', False):
            columns = ['UNITS', 'date', 'Raw1', 'temperature', 'Raw2', 'pressure[bar]', 'Raw3', 'depth[m]', 'nan']
            drop_cols = ['Raw1', 'Raw2', 'Raw3', 'nan']
        else:
            columns = ['UNITS', 'date', 'Raw1', 'pressure[bar]', 'Raw2', 'depth[m]', 'nan']
            drop_cols = ['Raw1', 'Raw2', 'nan']

        df = pd.read_csv(filepath, names=columns, header=21, encoding='latin-1')
        df = df.drop(columns=drop_cols)
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        return df

    def _assign_burst_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df['burstId'] = (df['UNITS'] == 'BURSTSTART').cumsum()
        return df.drop(columns=['UNITS'])
