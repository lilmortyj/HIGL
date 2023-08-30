import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

def load_file(file_path, key_words):
    # load csv files using pandas
    df = pd.read_csv(file_path)
    data = np.array(df[key_words])  # (num_data, 2)
    return data

def plot_heatmap(data, name='result'):
    X_range = (-4, 20)
    Y_range = (-4, 20)
    X_range_p = (-4, 20)
    Y_range_p = (-6, 6)
    print(f'num_data: {len(data)}')
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(name.split('/')[-1])
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    if "states" in name:
        if "AntMazeT" in name:
            for i in range(X_range_p[0], X_range_p[1]+1):
                ax.vlines(i, Y_range_p[0], Y_range_p[1], colors="k")  # x=?
            for j in range(Y_range_p[0], Y_range_p[1]+1):
                ax.hlines(j, X_range_p[0], X_range_p[1], colors="k")  # y=?
        elif "AntMaze" in name:
            for i in range(X_range[0], X_range[1]+1):
                ax.vlines(i, Y_range[0], Y_range[1], colors="k")  # x=?
            for j in range(Y_range[0], Y_range[1]+1):
                ax.hlines(j, X_range[0], X_range[1], colors="k")  # y=?
        ax.scatter(data[:,0], data[:,1], s=1, alpha=0.2)
    elif "goals" in name:
        ax.scatter(data[:,0], data[:,1], s=2, alpha=0.4)
    plt.savefig(name + '.png', dpi=400, bbox_inches='tight')

if __name__ == '__main__':
    root_dir = '../pics/'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)
    file_lists = [
        # '../results/AntMazeT-v0_hiro_2_traindata.csv',
        # '../results/20230714092309/AntMazeT-v0_higl_2_traindata.csv',
        # '../results/20230714092314/AntMazeT-v0_hrac_2_traindata.csv',
        '../results/20230718031211/AntMazeT-v0_td3_2_traindata.csv',
    ]
    key_words = [
        'h_state_x',
        'h_state_y',
    ]
    for k in range(len(file_lists)):
        name_ = file_lists[k].split('/')[-1].split('.')[0]
        name = root_dir + name_ + '_h_states'
        data = load_file(file_lists[k], key_words)
        plot_heatmap(data, name=name)