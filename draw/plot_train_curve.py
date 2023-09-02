import glob, copy
import os
import numpy as np 
import matplotlib.pyplot as plt
from matplotlib import pyplot
from scipy.ndimage.filters import gaussian_filter1d
from tensorboard.backend.event_processing import event_accumulator
plt.style.use("seaborn-whitegrid")
palette = pyplot.get_cmap('Set1')

root_dir = '../pics/'
if not os.path.exists(root_dir):
    os.makedirs(root_dir)

def load_files(file_list, key_words, dirty_to_del=list(), until=np.inf):
    data_list = [[] for _ in range(len(key_words))]
    min_len = np.inf
    for seed in file_list:
        data_seed = [[] for _ in range(len(key_words))]
        for frag in seed:
            ea = event_accumulator.EventAccumulator(frag, size_guidance={'scalars': 0})
            ea.Reload()
            # print(ea.scalars.Keys())
            for i, key in enumerate(key_words):
                item = ea.scalars.Items(key)
                data = np.stack([(i.step, i.value) for i in item if (i.step not in dirty_to_del) and (i.step <= until)], 0)  # num_data, 2
                # print(f'data_shape: {data.shape}')
                data_seed[i].append(data)
        for j in range(len(key_words)):
            data_seed[j] = np.concatenate(data_seed[j], axis=0)
            if len(data_seed[j]) < min_len:
                min_len = len(data_seed[j])
            data_list[j].append(data_seed[j])
    for i in range(len(key_words)):
        for j in range(len(data_list[i])):
            data_list[i][j] = data_list[i][j][:min_len]
    for i in range(len(key_words)):
        # key_word, num_seed, num_data, 2
        data_list[i] = [
            np.concatenate([data_list[i][0][:, 0:1], np.mean(np.stack(data_list[i], 0)[:,:,1:2], 0)], -1),
            np.concatenate([data_list[i][0][:, 0:1], np.std(np.stack(data_list[i], 0)[:,:,1:2], 0)], -1),
        ]
    return data_list


def smooth(arr_list, sigma=0.5):
    post_arr_list = []
    for arr in arr_list:
        arr[0][:, 1] = gaussian_filter1d(arr[0][:, 1], sigma=sigma)
        arr[1][:, 1] = gaussian_filter1d(arr[1][:, 1], sigma=sigma)
        post_arr_list.append(arr)
    return post_arr_list


def ploter(data_list, key_words, name='result', ylabel='Avg. Success Rate'):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(name)
    ax.set_xlabel('Timesteps')
    ax.set_ylabel(ylabel)
    for i in range(len(data_list)):
        mean_ = data_list[i][0]
        std_ = data_list[i][1]
        r1 = list(map(lambda x: x[0]-x[1], zip(mean_[:, 1], std_[:, 1])))
        r2 = list(map(lambda x: x[0]+x[1], zip(mean_[:, 1], std_[:, 1])))
        ax.plot(mean_[:, 0], mean_[:, 1], label=key_words[i], color=palette(i), linewidth=3.0)
        ax.fill_between(mean_[:, 0], r1, r2, color=palette(i), alpha=0.2)
    ax.legend(loc="best")
    plt.savefig(root_dir + name +' '+ ylabel + '.png', dpi=400, bbox_inches='tight')


if __name__ == '__main__':
    NUM_EVAL = 1
    NUM_SEED = 3
    ENV = 'AntReacher (ME, c=10)'
    NUM_EPI = 5
    file_dirs = [
        ('../logs/AntReacher-v0/hrac-ft/dense', '20230902083811'),
        ('../logs/AntReacher-v0/hrac-ft/dense', '20230902084408')
    ]
    y_labels = ['Avg. Success Rate', 'Avg. Steps2Finish']
    algos = [
        'hrac-ft-1td',
        'hrac-ft-10td',
    ]
    until = 5500000
    # dirty_to_del = [10000101, 10050000]
    dirty_to_del = []
    
    file_lists = [
        [sorted(glob.glob(f'{file_dirs[a][0]}/{t}/{file_dirs[a][1]}/events.*')) for t in range(NUM_SEED)] for a in range(len(file_dirs))
    ]
    for i in range((len(y_labels))):
        for j in range(NUM_EVAL):
            if i == 0:
                key_words = [
                    f'eval{j}/perc_env_goal_achieved',
                ]
            elif i == 1:
                key_words = [
                    f'eval{j}/avg_steps_to_finish',
                ]
            data_list = None
            key_words_list = None
            for k in range(len(file_lists)):
                data = load_files(file_lists[k], key_words, dirty_to_del=dirty_to_del, until=until)
                data = smooth(data, sigma=1.0)
                if data_list == None:
                    data_list = data
                else:
                    data_list += data
                kw_ = copy.deepcopy(key_words)
                for l in range(len(kw_)):
                    kw_[l] = algos[k] + '_' + kw_[l].split('/')[-1]
                if key_words_list == None:
                    key_words_list = kw_
                else:
                    key_words_list += kw_
            ploter(data_list,
                key_words_list,
                name=f'{ENV}: task {j+1} ({NUM_EPI} epis)',
                ylabel=y_labels[i],
            )