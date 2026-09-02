import pandas as pd

class ResistiveSensor():
    """
    A class to handle reading and processing the data files recorded by the resistive water level sensor at lab. 

    Notes
    -----
    22-Dec-2024 : Origination - Franklin Ayala

    """
    def __init__(self,directory_path):
        """
        Initializes the ResistiveSensor object with the given directory path.

        Parameters
        ----------
        directory_path : str
            The path to the directory where resistive sensor data is stored.
        """
        self.directory_path = directory_path

    def read_resistive_header(self, file_name):
        """
        Reads resistive sensor records from a specified file and processes the data into a pandas DataFrame.

        Parameters
        ----------
        file_name : str 
            Filename containing the resistive sensor records.

        Returns
        -------
            pandas.DataFrame: A DataFrame containing the raw resistive sensor data
        """
        file_path = f"{self.directory_path}/{file_name}"

        self.headers = open(file_path,'r')
        self.headers = self.headers.read().split('\n')

        full_date = {}
        end_count = 1
        for idx,line in enumerate(self.headers):
            list_line = line.split('\t')
            if list_line[0] == 'Time' or list_line[0] == 'Date':
                full_date[list_line[0]]=list_line[1]
            elif 'End_of_Header' in line:
                if end_count == 2:
                    lines_to_trim = idx + 1
                    break
                end_count += 1
        return full_date,lines_to_trim
    
    def read_resistive_data(self, file_name):
        full_date,lines_to_trim = self.read_resistive_header(file_name)
        file_path = f"{self.directory_path}/{file_name}"
        data = pd.read_csv(file_path,skiprows=lines_to_trim,delimiter='\t')

        return data