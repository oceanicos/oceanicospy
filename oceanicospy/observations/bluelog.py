import pandas as pd
from datetime import datetime
from io import StringIO

class BlueLogDataLoader:
    """
    Class to load BlueLog CSV data starting from the configured start time.
    """

    def __init__(self, file_path):
        # Path to the input CSV file
        self.file_path = file_path
        self.start_time = None
        self.df = None

    def load_filtered_data(self):
        """
        Reads the CSV file, extracts the configured start time, finds the data start index,
        and loads the data into a filtered pandas DataFrame.
        """
        # Read file content line by line
        with open(self.file_path, 'r') as file:
            lines = file.readlines()

        # Initialize index and time
        data_start_index = None

        for i, line in enumerate(lines):
            # Look for the configured start time
            if line.startswith("# Configured Start Time:"):
                time_str = line.strip().split(": ")[1]
                self.start_time = datetime.strptime(time_str, "%Y%m%d%H%M")

            # Find the '---' line that separates header from data
            if line.strip() == "---":
                data_start_index = i + 1
                break

        if self.start_time is None or data_start_index is None:
            raise ValueError("Configured start time or data section not properly found.")

        # Extract the lines that represent actual data
        data_lines = lines[data_start_index:]
        data_str = ''.join(data_lines)

        # Read the data into a DataFrame
        df = pd.read_csv(
            StringIO(data_str),
            names=["Timestamp", "Pressure_bar", "Temperature_C"],
            skip_blank_lines=True
        )

        # Force conversion of Timestamp to datetime, drop rows with invalid timestamps
        df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors='coerce')
        df = df.dropna(subset=["Timestamp"])

        # Filter rows based on the configured start time
        df = df[df["Timestamp"] >= self.start_time]

        # Reset index and store the result
        self.df = df.reset_index(drop=True)

    def get_dataframe(self):
        """
        Returns the filtered DataFrame. Loads the data if not already loaded.
        """
        if self.df is None:
            self.load_filtered_data()
        return self.df
