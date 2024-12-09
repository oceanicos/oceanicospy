import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap
import numpy as np

class FixPointNormalize(mpl.colors.Normalize):
    """ 
    Inspired by https://stackoverflow.com/questions/20144529/shifted-colorbar-matplotlib
    Subclassing Normalize to obtain a colormap with a fixpoint 
    somewhere in the middle of the colormap.

    This may be useful for a `terrain` map, to set the "sea level" 
    to a color in the blue/turquise range. 
    """
    def __init__(self, vmin=None, vmax=None, sealevel=0, col_val = 0.4885, clip=False):
        # sealevel is the fix point of the colormap (in data units)
        self.sealevel = sealevel
        # col_val is the color value in the range [0,1] that should represent the sealevel.
        self.col_val = col_val
        mpl.colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.sealevel, self.vmax], [0, self.col_val, 1]
        return np.ma.masked_array(np.interp(value, x, y))

    # Combine the lower and upper range of the terrain colormap with a gap in the middle
    # to let the coastline appear more prominently.
    # inspired by https://stackoverflow.com/questions/31051488/combining-two-matplotlib-colormaps


def register_soft_cmap (cmap_path):
    cmap_array=np.genfromtxt(cmap_path)
    customColourMap= LinearSegmentedColormap.from_list('soft_topo', cmap_array[::-1], N=256)
    plt.register_cmap(cmap=customColourMap)
    return 

cmap_txt_path = '/home/fayalacruz/runs/smooth_topo.txt'
register_soft_cmap(cmap_txt_path)