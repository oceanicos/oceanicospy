import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def sampling_data():
    return {
        "sampling_freq": 1.0,
        "anchoring_depth": 5.0,
        "sensor_height": 0.5,
        "burst_length_s": 512,
    }


@pytest.fixture
def single_burst_df(sampling_data):
    """512-sample DataFrame with a 0.1 Hz sine wave and a single burst."""
    n = int(sampling_data["burst_length_s"])
    t = np.arange(n) / sampling_data["sampling_freq"]
    eta = 0.5 * np.sin(2 * np.pi * 0.1 * t)
    idx = pd.date_range("2024-01-01", periods=n, freq="s")
    return pd.DataFrame({"eta[m]": eta, "burstId": 0}, index=idx)
