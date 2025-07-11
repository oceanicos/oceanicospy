import pandas as pd
import glob
import os

from .pressure_sensor_base import BaseLogger

class RBR(BaseLogger):
    def _get_records_file(self):
        files = glob.glob(os.path.join(self.directory_path, '*_data.txt'))

        if not files:
            raise FileNotFoundError("No .txt file found in the specified directory.")
        return files[0]

    def _load_raw_dataframe(self) -> pd.DataFrame:
        filepath = self._get_records_file()
        df = pd.read_csv(filepath)
        return df

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.drop(columns=['Sea pressure'], errors='ignore')
        df = df.rename(columns={'Pressure': 'pressure[bar]', 'Depth': 'depth[m]'})
        df['pressure[bar]'] = df['pressure[bar]'] / 10  # dbar to bar
        return df

    def _assign_burst_id(self, df: pd.DataFrame) -> pd.DataFrame:
        df['burstId'] = pd.factorize(df.index.floor('h'))[0] + 1
        return df
