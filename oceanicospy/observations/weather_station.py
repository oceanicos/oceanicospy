import pandas as pd
import numpy as np

class WeatherStation():
    """
    A class to handle reading and processing the data files recorded by the weather station (DAVIS). 

    Notes
    -----
    10-Dec-2024 : Origination - Franklin Ayala

    """
    def __init__(self,directory_path):
        """
        Initializes the WeatherStation object with the given directory path.

        Args:
            directory_path (str): The path to the directory where weather station data is stored.
        """
        self.directory_path = directory_path
    
    def read_records(self, file_name):
        """
        Reads weather station records from a specified file and processes the data into a pandas DataFrame.
        Args:
            file_name (str): The name of the file containing the weather station records.
        Returns:
            pandas.DataFrame: A DataFrame containing the processed weather station data
        Notes:
            - The function assumes that the first two lines of the file are headers or metadata and skips them.
            - The 'AM/PM' column values are replaced with 'AM' and 'PM' for consistency.
        """

        file_path = f"{self.directory_path}/{file_name}"
        with open(file_path, 'r') as file:
            data = file.read().split('\n')[2:-1]
        
        processed_data = [(' '.join(line.split())).split(' ') for line in data]
        
        columns = ['Date', 'time', 'AM/PM', 'Out', 'Temp1', 'Temp2', 'Hum', 'Pt.', 'Speed', 'Dir1', 'Run', 'Speed2', 'Dir2', 
                   'Chill', 'Index1', 'Index2', 'Bar', 'Rain', 'Rate', 'D-D1', 'D-D2', 'Temp4', 'Hum2', 'Dew', 'Heat', 'EMC', 
                   'Density', 'Samp', 'Tx', 'Recept', 'Int.']
        
        df = pd.DataFrame(processed_data, columns=columns)
        df['AM/PM'] = df['AM/PM'].replace({'a': 'AM', 'p': 'PM'})
        
        return df

    def get_clean_reecords(self):
        """
        Cleans and processes weather station records.
        This method reads weather station data from a file, replaces missing values,
        drops columns with all missing values, combines date and time columns into a 
        single datetime index, and drops the original date and time columns.
        
        Returns:
            pd.DataFrame: A DataFrame with cleaned weather station records, indexed by datetime.
        """
        self.records=self.read_records('weather_station_data.txt')

        self.records.replace('---', np.nan, inplace=True)
        self.records.dropna(axis=1, how='all', inplace=True)
        
        self.records['date'] = pd.to_datetime(self.records['Date'] + ' ' + self.records['time'] + ' ' + self.records['AM/PM'], 
                               format='%m/%d/%y %I:%M %p')
        self.clean_records = self.records.drop(['Date', 'time', 'AM/PM'], axis=1)
        self.clean_records = self.clean_records.set_index('date')
        
        dtypes = {'Speed': float}
        self.clean_records = self.clean_records.astype(dtypes)

        maps_to_degrees = {
                                'N': 0, 'NNE': 22.5, 'NE': 45, 'ENE': 67.5,
                                'E': 90, 'ESE': 112.5, 'SE': 135, 'SSE': 157.5,
                                'S': 180, 'SSW': 202.5, 'SW': 225, 'WSW': 247.5,
                                'W': 270, 'WNW': 292.5, 'NW': 315, 'NNW': 337.5
                            }

        self.clean_records['Direction'] = self.clean_records['Dir1'].map(maps_to_degrees)

        return self.clean_records



    