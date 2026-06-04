import os
import glob
import pandas as pd


class CTDProcessor:
    """
    A class to process CTD .csv files from a given directory.

    This class reads CTD .csv files, extracts metadata from filenames,
    parses measurement data, and stores the results as a list of DataFrames.

    Parameters
    ----------
    directory : str
        Path to the directory containing CTD .csv files.

    Attributes
    ----------
    dataframes : list of pd.DataFrame
        A list containing a processed DataFrame for each CTD file.
    """

    def __init__(self, directory):
        self.directory = directory
        self.dataframes = self._load_dataframes()

    def _load_dataframes(self):
        """
        Loads all .csv files from the specified directory, parses them,
        and returns a list of DataFrames with metadata.

        Returns
        -------
        list of pd.DataFrame
            A list of DataFrames with extracted metadata and measurements.
        """
        all_files = glob.glob(os.path.join(self.directory, "*.csv"))
        dfs = []

        for file_path in all_files:
            df = self._parse_csv(file_path)
            if df is not None:
                dfs.append(df)

        return dfs

    def _parse_csv(self, file_path):
        """
        Parses a single CTD .csv file and extracts metadata and measurement data.

        Parameters
        ----------
        file_path : str
            Path to the .csv file to parse.

        Returns
        -------
        pd.DataFrame or None
            A DataFrame containing the CTD measurements and metadata,
            or None if the file is not properly formatted.
        """
        try:
            # Extract metadata from filename
            filename = os.path.basename(file_path)
            parts = filename.split("_")

            if len(parts) < 3:
                return None

            cast_id = parts[0]
            date_str = parts[1]
            time_str = parts[2].split(".")[0]

            metadata = {
                "Cast": cast_id,
                "Date": pd.to_datetime(date_str, format="%Y%m%d", errors='coerce'),
                "Time": pd.to_datetime(time_str, format="%H%M%S", errors='coerce').time()
            }

            # Read CSV and skip any blank/comment lines
            df_raw = pd.read_csv(file_path, comment="#")

            # Append metadata to each row
            for key, value in metadata.items():
                df_raw[key] = value

            return df_raw

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return None

    def merge_dataframes(self):
        """
        Merges all individual DataFrames into a single DataFrame.

        Returns
        -------
        pd.DataFrame
            A single DataFrame combining all CTD data.
        """
        if not self.dataframes:
            return pd.DataFrame()
        return pd.concat(self.dataframes, ignore_index=True)
