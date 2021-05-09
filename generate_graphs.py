import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from common.matplotlib_extend import set_matplotlib_style


def get_graph_path(name):
    current_folder = os.path.dirname(__file__)
    graph_root = os.path.join(current_folder, './graph')
    return  os.path.join(graph_root, name)
   

def generate_noise_compare_graph(subfix=''):
    data=(        
        ("features only", "./data/analysis/noise/no_noise_train_out_20210503_232826"),
        ("features with noises" , "./data/analysis/noise/with_noise_train_out_20210503_233007"),
        ("noise only" , "./data/analysis/noise/noise_only_train_out_20210503_233102" ),      
    )
    
    fig, axes = plt.subplots(1, 3,figsize=(12,4),sharey='row')
    index =0
    for description,src in data:
        ax = axes[index]
        csv_path = os.path.join(src,'progress.csv')
        df = pd.read_csv(csv_path)
        ax.set_title(description)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('CAGR (%)')
        train_data = df ['exploration/env_infos/final/cagr Mean'].rolling(50).mean()*100
        validation_data = df ['evaluation/env_infos/final/cagr Mean'].rolling(50).mean()*100
        ax.yaxis.set_ticks(np.arange(0, 100, 10))

        ax.grid(True, linestyle='-')
        ax.plot(train_data,label='training')
        ax.plot(validation_data,label='validation')
        
        index+=1
        #print(df)
    plt.legend()
    plt.tight_layout()
    plt.savefig(get_graph_path(f'compare_noise{subfix}.png'))
    pass

if __name__ == '__main__':

    set_matplotlib_style(mode='slide')
    generate_noise_compare_graph('_slide')
    set_matplotlib_style()
    generate_noise_compare_graph()    
