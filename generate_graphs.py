import os
import matplotlib.pyplot as plt
import pandas as pd
current_folder = os.path.dirname(__file__)
graph_root = os.path.join(current_folder, './graph')


def generate_noise_compare_graph():
    srcs=dict(
        with_noise = "./data/analysis/noise/with_noise_train_out_20210503_233007",
        without_noise = "./data/analysis/noise/no_noise_train_out_20210503_232826",
        noise_only = "./data/analysis/noise/noise_only_train_out_20210503_233102" ,      
    )

    for src in srcs.keys():
        root = srcs[src]
        csv_path = os.path.join(root,'progress.csv')
        df = pd.read_csv(csv_path)
        print(df)
    pass

if __name__ == '__main__':
    generate_noise_compare_graph()
