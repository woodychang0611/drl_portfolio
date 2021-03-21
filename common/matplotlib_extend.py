import matplotlib
from cycler import cycler

def set_matplotlib_style():
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams['lines.linewidth'] = 2
    matplotlib.rcParams['axes.prop_cycle']=  cycler(color= ['#66FF99','#FFCC99','#FFFF99','#FF99FF'])
    for name in matplotlib.rcParams:
        if matplotlib.rcParams[name]=='black':
            matplotlib.rcParams[name] ='#B9CAFF'
        if matplotlib.rcParams[name]=='white':
            matplotlib.rcParams[name] ='#0C0C3A'   