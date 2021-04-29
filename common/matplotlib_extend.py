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

def plot_ma(series,lables,title,n):
    fig, ax = matplotlib.pyplot.subplots()
    for s,label in zip(series,lables):
        x=range(len(s))
        y_std = s.rolling(n).std()   
        y_mean = s.rolling(n).mean()
        ax.plot(y_mean,label=label)
        ax.set_title(title)
        ax.fill_between(x,y_mean-y_std, y_mean+y_std, alpha=0.2)
    matplotlib.pyplot.legend()            