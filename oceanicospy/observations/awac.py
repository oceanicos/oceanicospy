import glob
import pandas as pd
import numpy as np
import pandas as pd

from ..utils import wave_props

class Awac():
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
        
    def read_wave_header(self):
        """
        Reads and parses the header file (.hdr) to extract the column names.

        The function filters and processes the header information to generate a list of column names.

        Returns
        -------
        numpy.ndarray
            An array of column names extracted from the .hdr file.
        """
    
        self.header = glob.glob(self.directory_path+'*.hdr') # File with headers
        self.headers = open(self.header[0],'r')
        self.headers = self.headers.read().split('\n')

        # Replacing the title with a mask
        self.tf = []
        self.control = False
        for i in self.headers:
            if self.control == True:
                if i.endswith('-') == True:
                    self.control = False
                    self.tf.append(False)
                else:
                    self.tf.append(True)
            else:
                self.tf.append(False)
                
            if i.endswith('.wad]') == True:
                self.control = True

        self.headers = list(np.array(self.headers)[self.tf])
        self.columns = np.array([' '.join(i.split()) for i in self.headers])
        if self.columns[-1] == '':
            self.columns = self.columns[:-1]
        return self.columns

    def read_wave_records(self):
        """
        Reads and processes the .wad files to create a DataFrame containing the burst data.

        For each .wad file, the function reads the data, adds a 'burstId' column, and combines all the data into a single DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the concatenated data from all the .wad files with an added 'burstId' column.
        """

        self.columns_ = self.read_wave_header()
        self.data = sorted(glob.glob(self.directory_path+'*.wad')) #Each .wad file represents one burst
        self.wads = []
        self.burst_id = 1

        for i in self.data:
            self.e = pd.read_csv(i,header=0,delim_whitespace=True,names=self.columns_)
            self.e['burstId'] = (np.ones(len(self.e))*self.burst_id).astype(int)
            self.burst_id += 1
            self.wads.append(self.e.dropna())

        self.wads = pd.concat(self.wads)
        self.columns_ = np.append(self.columns_,['burstId'])
        return self.wads
    
    def get_clean_wave_records(self):
        """
        Processes the raw data by converting certain columns to numeric types, adding a timestamp, and filtering the data 
        by the specified time range.

        The function also renames columns and returns a cleaned DataFrame that includes only relevant data.

        Returns
        -------
        pandas.DataFrame
            A cleaned DataFrame containing 'pressure', 'u', 'v', and 'burstId' columns, filtered by the specified time range.
        """

        self.wads = self.read_wave_records()
        self.wads.iloc[:,[11,12]] = self.wads.iloc[:,[11,12]].astype(float)
        self.raw_data = self.wads.iloc[:,[0,1,2,3,4,5,6,11,12,17]]

        self.raw_data['date'] = pd.to_datetime(self.raw_data.iloc[:,[2,0,1,3,4,5]].astype(str).agg('-'.join, axis=1),
                                format='%Y-%m-%d-%H-%M-%S.%f',errors='ignore')
        self.raw_data = self.raw_data.set_index('date')
        self.clean_data = self.raw_data[self.columns_[[6,11,12,17]].tolist()]
        self.clean_data.columns = ['pressure','u','v','burstId']
        self.clean_data['pressure'] = self.clean_data['pressure'].astype(float)
        self.clean_data = self.clean_data[self.sampling_data['start_time']:self.sampling_data['end_time']]
        return self.clean_data
    
    def read_currents_header(self):
        """
        Reads the section of profile setup from the header file (.hdr) .

        Returns
        -------
        dict
            A dictionary with the most important data for the current records.
        """        
        self.header = glob.glob(self.directory_path+'*.hdr') # File with headers
        self.headers = open(self.header[0],'r')
        self.headers = self.headers.read().split('\n')

        self.filtered_lines = []
        for line in self.headers:
            if any(keyword in line for keyword in ['first measurement', 'Profile interval', 'Number of cells', 'Cell size', 'Blanking distance']):
                self.filtered_lines.append(line)
        
        self.current_header = {}
        for line in self.filtered_lines:
            key = []
            value = []
            for element in line.split():
                if element.isalpha() == True:
                    key.append(element)
                else:
                    value.append(element)
            key = ' '.join(key)
            value = ' '.join(value)

            if 'first measurement' in key:
                key = 'start_time'
            else:
                value = float(value)
            self.current_header[key] = value
        return self.current_header

    def read_currents_records(self):
        """
        Reads and processes the .v1 and .v2 files to create a DataFrame.

        Returns
        -------
        pandas.DataFrame
            A DataFrame containing the current magnitude and direction.
        """
        self.curent_header = self.read_currents_header()

        self.x_component_file = sorted(glob.glob(self.directory_path+'*.v1'))[0]
        self.y_component_file = sorted(glob.glob(self.directory_path+'*.v2'))[0]
        
        self.x_component = pd.read_csv(self.x_component_file,delim_whitespace=True,header=None)
        self.y_component = pd.read_csv(self.y_component_file,delim_whitespace=True,header=None)

        self.date_range = pd.date_range(self.current_header['start_time'],periods=self.x_component.shape[0],
                                        freq=f"{self.current_header['Profile interval sec']}s")

        self.x_component = self.x_component.set_index(self.date_range)
        self.x_component.columns = map(str,np.arange(1,self.current_header['Number of cells']+1,dtype=int))

        self.y_component = self.y_component.set_index(self.date_range)
        self.y_component.columns = map(str,np.arange(1,self.current_header['Number of cells']+1,dtype=int))

        self.current_magnitude = np.sqrt((self.x_component**2)+(self.y_component**2))
        self.current_dir = np.array([list(map(wave_props.angulo_norte,row_x,row_y)) for row_x,row_y in zip(self.x_component.values,self.y_component.values)])
        self.current_dir = pd.DataFrame(data=self.current_dir,index=self.date_range,columns=self.current_magnitude.columns)
        return self.current_magnitude,self.current_dir