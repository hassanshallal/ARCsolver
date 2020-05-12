import os
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors

import numpy as np

# This utils.py is not supposed to be related to any logic

# This is to serialize the opjects
def serialize(object_, name):
    outfile = open(name, 'wb')
    pickle.dump(object_, outfile)
    outfile.close()

# This is to load serialized objects
def load_serialized(name):
    file = open(name, 'rb')
    object_ = pickle.load(file)
    file.close()
    return object_

# load set
def load_set(path):
    cases = []
    tasks = sorted(os.listdir(path))

    for n in tasks:
        task_file = str(path / n)

        with open(task_file, 'r') as f:
            task = json.load(f)
            cases.append(task)
    return cases

# critical for processing further
def fix_dim(arr):
    if type(arr) == list:
        arr = np.array(arr)
    elif type(arr) != np.ndarray:
        raise BaseException('Input must be a list or a numpy array.')

    guaranteed_dim = arr.shape[0]
    try:
        x = arr.shape[1]
    except:
        arr = arr.reshape((guaranteed_dim, 1))

    return arr

# get training: only gets the trainining inputs and outputs: there is phase_0 evaluation on the raw_task['test'] and there is phase_1 evaluation on the evaluation set train and there is a phase_2 evaluation on the evaluation set test and there is a phase_3 evaluation on the test set train and finally there is a phase_4 evaluation on the test set test (ultimate)


def get_training(raw_task):
        training = raw_task['train']
        num_train = len(training)

        # prepare
        traininputs = []
        trainoutputs = []

        for m in range(num_train):
            this_input = fix_dim(training[m]['input'])
            this_output = fix_dim(training[m]['output'])

            traininputs.append(this_input)
            trainoutputs.append(this_output)

        assert len(traininputs) == len(trainoutputs)
        return traininputs, trainoutputs


# plotting a task
def plot_task(task, plot_test=False):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    train_len = len(task['train'])
    if plot_test:
        fig_dim = train_len*2 + 2
    else:
        fig_dim = train_len*2
    fig, axs = plt.subplots(1, fig_dim, figsize=(fig_dim + 10,  fig_dim + 10))
    # 0 --> 0, 1,
    # 1 --> 2, 3
    # 2 --> 4, 5

    # 4 --> 8, 9
    for n in range(train_len):
        axs[2*n].imshow(task['train'][n]['input'], cmap=cmap, norm=norm)
        axs[2*n].axis('off')
        axs[2*n].set_title('Train ' + str(n) + ' Input')
        axs[2*n+1].imshow(task['train'][n]['output'], cmap=cmap, norm=norm)
        axs[2*n+1].axis('off')
        axs[2*n+1].set_title('Train ' + str(n) + ' output')
    if plot_test:
        axs[train_len*2].imshow(task['test'][0]['input'], cmap=cmap, norm=norm)
        axs[train_len*2].axis('off')
        axs[train_len*2].set_title('Test Input')
        axs[train_len*2 + 1].imshow(task['test'][0]
                                    ['output'], cmap=cmap, norm=norm)
        axs[train_len*2 + 1].axis('off')
        axs[train_len*2 + 1].set_title('Test Output')

    plt.tight_layout()
    plt.show()

    
## instantiation of a task manager utilities
def explore_dim_arrays(list_of_arrays):
    dim_zero_min = min([n.shape[0] for n in list_of_arrays])
    dim_zero_max = max([n.shape[0] for n in list_of_arrays])
    dim_zero_range = dim_zero_max - dim_zero_min
    dim_one_min = min([n.shape[1] for n in list_of_arrays])
    dim_one_max = max([n.shape[1] for n in list_of_arrays])
    dim_one_range = dim_one_max - dim_one_min
    
    return [(dim_zero_min, dim_zero_max, dim_zero_range), (dim_one_min, dim_one_max, dim_one_range)]

    
def explore_dimensions(traininputs, trainoutputs):
    is_similar_dim = all([n.shape == m.shape for n, m in zip(traininputs, trainoutputs)])
    input_dims = explore_dim_arrays(traininputs)
    output_dims = explore_dim_arrays(trainoutputs)
        
    return is_similar_dim, input_dims, output_dims
    
    
def get_value_map(in_, out_):
    target_indices = np.argwhere(in_ != out_)
    value_map = {}
    
    # we can't get a value map unless the shapes are equal
    if in_.shape == out_.shape:
        for n in target_indices:
            if in_[n[0], n[1]] not in value_map.keys():
                value_map[in_[n[0], n[1]]] = set()
            value_map[in_[n[0], n[1]]].add(out_[n[0], n[1]])
    
    return value_map

def get_global_value_map(list_of_value_maps):
    global_vm = {}
    for n in list_of_value_maps:
        for key, value in n.items():
            if key not in global_vm.keys():
                global_vm[key] = set()
            if len(value) > 0:
                for m in value:
                    global_vm[key].add(m)
            
    return global_vm


# get cell neighbours
# https://www.kaggle.com/arsenynerinovsky/cellular-automata-as-a-language-for-reasoning
nbh = lambda x, i, j: { #x is array, i and j are row and column indices
    (ip, jp) : x[i+ip, j+jp]
        for ip, jp in product([1, -1, 0], repeat=2)
            if (0 <= i < x.shape[0]) and (0 <= j < x.shape[1]) and (0 <= i+ip < x.shape[0]) and (0 <= j+jp < x.shape[1])
}

# get a background 
def get_background(arr):
    bincount = np.bincount(arr.flatten())
    major = bincount.argmax()
    return major

 
# Tests based on prior knowledge
# process_diff gives insights about the bg and the nonbg
def process_diff(in_, out_, in_bg):

    in_mask = np.where(in_ != in_bg, True, False)
    out_mask = np.where(in_ != out_, True, False)

    overall_set = set()
    for n in range(in_mask.shape[0]):
        for m in range(in_mask.shape[1]):
            overall_set.add(coder0[saver0[(in_mask[n, m], out_mask[n, m])]])

    return str(sorted(list(overall_set)))


