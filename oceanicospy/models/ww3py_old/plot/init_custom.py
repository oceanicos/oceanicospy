import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.font_manager.fontManager.addfont('/home/fayalacruz/runs/Helvetica.ttf')
mpl.font_manager.fontManager.addfont('/home/fayalacruz/runs/Helvetica-Light.ttf')
mpl.font_manager.fontManager.addfont('/home/fayalacruz/runs/Helvetica-Bold.ttf')

newparams = {'axes.grid': False,
             'lines.linewidth': 1.5,
             'ytick.labelsize':12,
             'xtick.labelsize':12,
             'axes.labelsize':12,
             'axes.titlesize':12,
             'legend.fontsize':12,
             'figure.titlesize':12,
             'font.family':'Helvetica'}
plt.rcParams.update(newparams)
