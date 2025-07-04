import matplotlib.pyplot as plt
import numpy as np
from windrose import WindroseAxes

def plot_wave_parameters(wave_parameters,output_dir,metadata_text, parameters=['Hm0', 'Hrms', 'Hmean'], figsize=(12, 3), 
                        ylabel='Wave Height [m]', xlabel='Time',
                        fig_comparison=None,label_comparison=''):
    """
    Function to plot wave parameters (e.g., Hm0, Hrms, Hmean) over time.

    Parameters:
    - wave_parameters: Dictionary containing wave parameters like 'Hm0', 'Hrms', 'Hmedia'.
    - parameters: List of keys in the dictionary to plot (default: ['Hm0', 'Hrms', 'Hmedia']).
    - figsize: Tuple defining the figure size (default: (18, 4)).
    - ylabel: Label for the y-axis (default: 'Wave Height [m]').
    - xlabel: Label for the x-axis (default: 'Time').
    """
    
    if fig_comparison == None:
        fig,ax = plt.subplots(1,1,figsize=figsize)
    else:
        fig,ax = fig_comparison

    for param in parameters:
        ax.plot(wave_parameters.index, wave_parameters[param], label=f'{param}_{label_comparison}')

    ax.set(ylabel=ylabel, xlabel=xlabel)
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    if np.any(['H' in param for param in parameters]):
        plt.savefig(f'{output_dir}/Hs_series_{metadata_text}.png',dpi=500)
    else:
        plt.savefig(f'{output_dir}/T_series_{metadata_text}.png',dpi=500)
    return fig,ax


def plot_1d_wave_spectra(wave_spectra,ax,metadata_text=None,output_dir=None,figsize=(12, 3), ylabel='f [Hz]', xlabel='Time', label_comparison=''):
    """
    Function to plot the 1d wave spectra over time.

    Parameters:
    - wave_spectra: Dictionary containing wave spectra data.
    - figsize: Tuple defining the figure size (default: (18, 4)).
    - ylabel: Label for the y-axis (default: 'f [Hz]').
    - xlabel: Label for the x-axis (default: 'Time').
    """

    if output_dir == None:
        cax = ax.pcolormesh(
            wave_spectra['time'],
            wave_spectra['freq'],
            np.transpose(wave_spectra['S']),
            cmap='magma_r',
            vmin=0,
            vmax=0.3,
            shading='auto'
        )
        # ax.colorbar(label='S [m^2/Hz]')  
        return ax,cax
    else:
        fig,ax = plt.subplots(1,1,figsize=figsize)
        ax.pcolormesh(wave_spectra['time'], wave_spectra['freq'][0],np.transpose(wave_spectra['S']),cmap='plasma')
        ax.set(ylabel=ylabel,xlabel=xlabel)
        ax.axhline(y=0.04,color='red',ls='--')
        ax.colorbar(label='S [m^2/Hz]')
        plt.savefig(f'{output_dir}/1d_spectra_series_{metadata_text}_{label_comparison}.png',dpi=500,bbox_inches='tight',pad_inches=0.1)

def plot_wind_rose(wind_data, output_dir, metadata_text, figsize=(8, 8), bins=None):
    """
    Function to plot a wind rose using the windrose library.

    Parameters
    ----------
    wind_data : pandas.DataFrame
        DataFrame containing wind speed and direction data. It should have columns 'speed' and 'direction'.
    output_dir : str
        Directory where the output figure will be saved.
    metadata_text : str
        Text to be included in the filename for the saved figure.
    figsize : tuple, optional
        Tuple defining the figure size (default is (8, 8)).
    bins : list, optional
        List of bin edges for the wind speed (default is None, which will use the default bins in windrose).
    cmap : str, optional
        Colormap to be used for the wind rose (default is 'viridis').

    Returns
    -------
    fig : matplotlib.figure.Figure
        The figure object.
    ax : windrose.WindroseAxes
        The windrose axes object.
    """
    fig = plt.figure(figsize=figsize)
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(wind_data['Direction'], wind_data['Speed'], normed=True, opening=0.8, bins=bins, edgecolor='black')
    ax.set_legend()
    plt.savefig(f'{output_dir}/wind_rose_{metadata_text}.png', dpi=500, bbox_inches='tight', pad_inches=0.1)
    return fig, ax