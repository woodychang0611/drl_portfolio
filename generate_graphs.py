import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from common.matplotlib_extend import set_matplotlib_style


def get_graph_path(name):
    current_folder = os.path.dirname(__file__)
    graph_root = os.path.join(current_folder, './graph')
    return os.path.join(graph_root, name)


def generate_noise_compare_graph(subfix=''):
    data = (
        ("features only", "./data/analysis/noise/no_noise_train_out_20210503_232826"),
        ("features with noises",
         "./data/analysis/noise/with_noise_train_out_20210503_233007"),
        ("noise only", "./data/analysis/noise/noise_only_train_out_20210503_233102"),
    )

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey='row')
    index = 0
    for description, src in data:
        ax = axes[index]
        csv_path = os.path.join(src, 'progress.csv')
        df = pd.read_csv(csv_path)
        ax.set_title(description)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('CAGR (\%)')
        train_data = df['exploration/env_infos/final/cagr Mean'].rolling(
            50).mean()*100
        validation_data = df['evaluation/env_infos/final/cagr Mean'].rolling(
            50).mean()*100
        ax.yaxis.set_ticks(np.arange(0, 100, 10))

        ax.grid(True, linestyle='-')
        ax.plot(train_data, label='training')
        ax.plot(validation_data, label='validation')

        index += 1
        # print(df)
    plt.legend()
    plt.tight_layout()
    plt.savefig(get_graph_path(f'compare_noise{subfix}.png'))
    plt.clf()


def generate_penalty_negtive_profits_compare_graph(subfix=''):
    data = (
        (r"$\theta$ = 0.002, penalty upon negtive profits only",
         "./data/analysis/drop/drop_only/0.002_drop_only_train_out_20210508_132328"),
        (r"$\theta$ = 0.002, penalty upon all profits",
         "./data/analysis/drop/0.002_train_out_20210508_183303"),
        (r"$\theta$ = 0.006, penalty upon negtive profits only",
         "./data/analysis/drop/drop_only/0.006_drop_only_train_out_20210507_182555"),
        (r"$\theta$ = 0.006, penalty upon all profits",
         "./data/analysis/drop/0.006_train_out_20210507_205804"),
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 12), sharey='row')
    index = 0
    for description, src in data:
        ax = axes[int(index/2)][int(index % 2)]
        csv_path = os.path.join(src, 'progress.csv')

        df = pd.read_csv(csv_path)
        ax.set_title(description)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MDD (\%)')
        train_data = df['exploration/env_infos/final/mdd Mean'].rolling(
            50).mean()*100
        validation_data = df['evaluation/env_infos/final/mdd Mean'].rolling(
            50).mean()*100

        ax.grid(True, linestyle='-')
        ax.plot(train_data, label='training')
        ax.plot(validation_data, label='validation')
        index += 1

    plt.legend()
    plt.tight_layout()
    plt.savefig(get_graph_path(f'penalty_negtive_profits_compare{subfix}.png'))
    plt.clf()

def generate_compare_crp(subfix=''):
    data = (
        (r'$\theta$ =$\infty$',r'./data/analysis/threshold/inf_replay_20210509_222427'),
        (r'$\theta$ = 0.006',r'./data/analysis/threshold/0.006_replay_20210509_222430'),
        (r'$\theta$ = 0.002',r'./data/analysis/threshold/0.002_replay_20210516_171556'),        
    )
    
    row_count = len(data)
    fig, axes = plt.subplots(row_count, 2, figsize=(12, 4*row_count), sharey='row')
    row =0
    for description, src in data: 
        for id in (1,2):  
            ax = axes[row][id-1]
            exp_df = pd.read_csv(os.path.join(src,f'infos_{id}.csv'), parse_dates=['date'], index_col=['date'])
            crp_df = pd.read_csv(os.path.join(src,f'infos_{id}_crp.csv'), parse_dates=['date'], index_col=['date'])
            exp_data = exp_df['wealths']
            crp_data = crp_df['wealths']
            exp_cagr = exp_df['cagr'][-1]
            exp_mdd = exp_df['mdd'][-1]
            crp_cagr = crp_df['cagr'][-1]
            crp_mdd = crp_df['mdd'][-1]            

            ax.plot(exp_data.index,exp_data.to_numpy(),label='Experiment')
            ax.plot(crp_data.index,crp_data.to_numpy(),label='CRP')
            ax.set_title(f'Period {id}, {description}')
            ax.set_xlabel('Date')
            ax.set_ylabel('Wealths')
            performance_text = f"""
                CAGR: {exp_cagr:.1%} (Exp.) {crp_cagr:.1%}  (CRP)\n             
                MDD: {exp_mdd:.1%} (Exp.) {crp_mdd:.1%} (CRP) """
            #update for latex
            performance_text=performance_text.replace("%","\%")
            ax.text(0.05, 0.95,performance_text, transform=ax.transAxes, verticalalignment='top')
            plt.setp(ax.get_xticklabels(), rotation=30, horizontalalignment='right')         
        row+=1
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig(get_graph_path(f'crp_compare{subfix}.png'))
    plt.clf()

if __name__ == '__main__':
    #set_matplotlib_style(mode='slide')
    # generate_noise_compare_graph('_slide')
    #generate_penalty_negtive_profits_compare_graph('_slide')
    set_matplotlib_style()
    generate_compare_crp()
    # generate_noise_compare_graph()
    # generate_penalty_negtive_profits_compare_graph()
