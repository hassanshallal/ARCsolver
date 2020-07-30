import os
import pickle
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors

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

# get training: only gets the trainining inputs and outputs: there is phase_0 evaluation on the raw_task['test'] and there is phase_1 evaluation on the evaluation set train and there is a phase_2 evaluation on the evaluation set test and there is a phase_3 evaluation on the test set train and finally there is a phase_4 evaluation on the test set test (ultimate)
def get_training(raw_task):
        training = raw_task['train']
        num_train = len(training)

        # prepare
        traininputs = []
        trainoutputs = []

        for m in range(num_train):
            this_input = training[m]['input']
            this_output = training[m]['output']

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
