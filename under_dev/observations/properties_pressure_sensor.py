import numpy as np
import pandas as pd

@property
def first_submerged_record_time(self):
    """
    Returns the timestamp of the first record in the dataset where the sensor is submerged.

    Returns
    -------
    pandas.Timestamp
        The timestamp of the first available record when the sensor is submerged.
    """
    df = self._load_raw_dataframe()
    df = self._standardize_columns(df)
    df = self._compute_depth_from_pressure(df)
    sign_depth = np.sign(df['depth[m]'])

    # identifying changes in depth from negative to positive (submerged)
    idx_changes = np.where(np.diff(sign_depth) > 0)[0] +1
    if len(idx_changes) > 0:
        timestamp = pd.to_datetime(df.index[idx_changes[0]])
        if timestamp.minute != 0:
            return timestamp.ceil('h')
        else:
            return timestamp
    else:
        return "Sensor was never submerged"

@property
def last_submerged_record_time(self):
    """
    Returns the timestamp of the last record in the dataset where the sensor is submerged.

    Returns
    -------
    pandas.Timestamp
        The timestamp of the last available record where the sensor is submerged.
    """
    df = self._load_raw_dataframe()
    df = self._standardize_columns(df)
    df = self._compute_depth_from_pressure(df)
    sign_depth = np.sign(df['depth[m]'])

    # identifying changes in depth from positive to negative (emerged)
    idx_changes = np.where(np.diff(sign_depth) < 0)[0] + 1 # what if there are many?

    if len(idx_changes) > 0:
        timestamp = pd.to_datetime(df.index[idx_changes[0]])
    else:
        timestamp = pd.to_datetime(df.index[-1])
    return timestamp.floor('h')