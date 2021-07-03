import matplotlib
from cycler import cycler
import os
def set_matplotlib_style(mode=None):
    matplotlib.rcParams.update(matplotlib.rcParamsDefault)
    matplotlib.rcParams.update({'font.size': 15})
    matplotlib.rcParams['lines.linewidth'] = 2
    matplotlib.rcParams["text.usetex"] = True
    matplotlib.rcParams['axes.prop_cycle']=  cycler(color= ['#6600CC','#33CC33'])
    if(mode=='slide'):
        matplotlib.rcParams.update({'font.size': 20})
  #      matplotlib.rcParams['axes.prop_cycle']=  cycler(color= ['#66FF99','#FFCC99','#FFFF99','#FF99FF'])
  #      for name in matplotlib.rcParams:
  #          if matplotlib.rcParams[name]=='black':
  #              matplotlib.rcParams[name] ='#B9CAFF'
  #          if matplotlib.rcParams[name]=='white':
  #              matplotlib.rcParams[name] ='#0C0C3A'

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

def get_graph_path(name):
    current_folder = os.path.dirname(__file__)
    graph_root = os.path.join(current_folder, './../graph')
    print(os.path.join(graph_root, name))
    return os.path.join(graph_root, name)