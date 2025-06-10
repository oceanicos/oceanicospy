import requests
import os

def getting_wl_data(station_code,data_path):
    # Define the base URL and the specific file you want to download
    base_url = 'https://uhslc.soest.hawaii.edu/data/csv/fast/hourly/'
    file_name = f'h{station_code}.csv'  # Replace with the actual file name

    # Construct the full URL
    file_url = base_url + file_name

    # Send a GET request to fetch the file
    response = requests.get(file_url)

    # Check if the request was successful
    if response.status_code == 200:
        # Open a local file to write the content
        with open(file_name, 'wb') as file:
            file.write(response.content)
        print(f"Downloaded {file_name} successfully.")
    else:
        print(f"Failed to download {file_name}. Status code: {response.status_code}")

    os.system(f'mv {file_name} {data_path}')