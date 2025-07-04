import pandas as pd
import glob

from scipy.signal import detrend

from oceanicospy.utils import constants


class RBR():
  def __init__(self,directory_path,sampling_data):
    """
    Initializes the RBR class with the given directory path, sampling data.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing the .txt file.
    sampling_data : dict
        Dictionary containing the information about the device installation
    """
    self.directory_path = directory_path
    self.sampling_data = sampling_data

  def get_raw_records(self):
    """
    Reads the .txt file from the device to create a DataFrame containing data.

    Returns
    -------
    pandas.DataFrame
    """

    # Write a conditional to know whether or not the depth series has already been calculated with the device software

    self.filepath = glob.glob(self.directory_path+'*_data.txt')[0]
    self.raw_data = pd.read_csv(self.filepath)
    self.raw_data['date']= pd.to_datetime(self.raw_data['Time'])
    self.raw_data = self.raw_data.drop(['Time'],axis=1)   
    self.raw_data = self.raw_data.set_index('date')
    self.raw_data = self.raw_data[self.sampling_data['start_time']:self.sampling_data['end_time']]
    return self.raw_data

  def get_clean_records(self,detrended: bool=True):
    """
    Processes the raw data by grouping the series per each burst

    Returns
    -------
    pandas.DataFrame
        A cleaned DataFrame containing the columns '....', filtered by the specified time range.
    """

    self.clean_data = self.get_raw_records()
    
    self.clean_data = self.clean_data.drop(['Sea pressure'],axis=1)   
    self.clean_data = self.clean_data.rename(columns={'Pressure': 'pressure[bar]', 'Depth': 'depth[m]'})
    self.clean_data['pressure[bar]'] = self.clean_data['pressure[bar]']/10
    self.clean_data['depth_aux[m]'] = ((self.clean_data['pressure[bar]'] - constants.ATM_PRESSURE_BAR) * 1e5) / (constants.WATER_DENSITY * constants.GRAVITY)

    self.hours = self.clean_data.index.floor('h')  # Or use df.index.hour if just hour values

    # Factorize to get a unique integer ID per hour
    self.clean_data['burstId'] = pd.factorize(self.hours)[0] + 1  # start from 1
    
    # self.clean_data['burstId'] = (self.clean_data.index.floor('H') != self.clean_data.index.floor('H').shift()).cumsum()
    # self.clean_data['burstId'] = (self.clean_data['UNITS'] == 'BURSTSTART').cumsum()
    self.clean_data['eta[m]'] = self.clean_data.groupby('burstId')['depth[m]'].transform(lambda x: x - x.mean())

    if detrended:
      self.clean_data['eta[m]'] = self.clean_data.groupby('burstId')['eta[m]'].transform(lambda x: detrend(x.values, type='linear'))
    return self.clean_data