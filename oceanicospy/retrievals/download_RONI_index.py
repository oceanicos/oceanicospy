import requests
import pandas as pd
import io

def classify_phase(value):
    if value > 0.5:
        return "El Niño"
    elif value < -0.5:
        return "La Niña"
    else:
        return "Neutral"

def download_roni_index():
    url = "https://www.cpc.ncep.noaa.gov/data/indices/RONI.ascii.txt"
    response = requests.get(url)
    txt = response.text
    df = pd.read_csv(io.StringIO(txt), sep='\\s+')
    df['Phase'] = df['ANOM'].apply(classify_phase)
    season_months = {
    "DJF": 1,
    "JFM": 2,
    "FMA": 3,
    "MAM": 4,
    "AMJ": 5,
    "MJJ": 6,
    "JJA": 7,
    "JAS": 8,
    "ASO": 9,
    "SON": 10,
    "OND": 11,
    "NDJ": 12
        }
    df['Month'] = df['SEAS'].map(season_months)
    df['Date'] = pd.to_datetime(df['YR'].astype(str) + '-' + df['Month'].astype(str) + '-15')
    return df