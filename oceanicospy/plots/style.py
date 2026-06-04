import matplotlib as mpl
import matplotlib.pyplot as plt
from pathlib import Path

_fonts_dir = Path(__file__).parent / "fonts"

for font_file in _fonts_dir.glob("*.ttf"):
    mpl.font_manager.fontManager.addfont(str(font_file))

newparams = {
    'axes.grid': False,
    'lines.linewidth': 1.5,
    'ytick.labelsize': 12,
    'xtick.labelsize': 12,
    'axes.labelsize': 12,
    'axes.titlesize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 12,
    'font.family': 'Helvetica',
    'figure.dpi' : 150
}

plt.rcParams.update(newparams)
