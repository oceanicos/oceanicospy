import matplotlib.pyplot as plt
import numpy as np

def plot_wave_parameters(wave_parameters,output_dir,metadata_text, parameters=['Hm0', 'Hrms', 'Hmean'], figsize=(12, 3), ylabel='Wave Height [m]', xlabel='Time',
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
    
    if fig_comparison==None:
        fig,ax=plt.subplots(1,1,figsize=figsize)
    else:
        fig,ax=fig_comparison

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


def plot_1d_wave_spectra(wave_spectra,output_dir,metadata_text, figsize=(12, 3), ylabel='f [Hz]', xlabel='Time', label_comparison=''):
    """
    Function to plot the 1d wave spectra over time.

    Parameters:
    - wave_spectra: Dictionary containing wave spectra data.
    - figsize: Tuple defining the figure size (default: (18, 4)).
    - ylabel: Label for the y-axis (default: 'f [Hz]').
    - xlabel: Label for the x-axis (default: 'Time').
    """
    plt.figure(figsize=figsize)

    plt.pcolormesh(wave_spectra['time'], wave_spectra['freq'][0],np.transpose(wave_spectra['S']),cmap='turbo')

    plt.ylabel(ylabel)
    plt.xlabel(xlabel)
    plt.axhline(y=0.04,color='red',ls='--')
    plt.colorbar(label='S [m^2/Hz]')
    plt.savefig(f'{output_dir}/1d_spectra_series_{metadata_text}_{label_comparison}.png',dpi=500,bbox_inches='tight',pad_inches=0.1)