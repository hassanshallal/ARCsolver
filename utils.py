import os
import pickle
from copy import deepcopy
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors
from itertools import permutations, combinations
import numpy as np


from deductiveV02 import *



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

def get_testing(raw_task):
        testing = raw_task['test']
        num_test = len(testing)

        # prepare
        testinputs = []
        testoutputs = []

        for m in range(num_test):
            this_input = fix_dim(testing[m]['input'])
            testinputs.append(this_input)
            if 'output' in testing[m].keys() and len(testing[m]['output']) > 0:
                this_output = fix_dim(testing[m]['output'])
                testoutputs.append(this_output)
        #assert len(traininputs) == len(trainoutputs)
        return testinputs, testoutputs


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

def plot_task_eval(task, testpreds):
    """
    Plots the first train and test pairs of a specified task,
    using same color scheme as the ARC app
    """
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])
    norm = colors.Normalize(vmin=0, vmax=9)
    train_len = len(task['train'])
    test_len = len(task['test'])

    fig_dim = 0
    if type(testpreds) == list:
        fig_dim = train_len*2 + test_len*3
    elif type(testpreds) == dict:
        fig_dim = train_len*2 + (test_len*2 + len(testpreds))

    #print(fig_dim)
    fig, axs = plt.subplots(1, fig_dim, figsize=(fig_dim + 10,  fig_dim + 10))

    for n in range(train_len):
        axs[2*n].imshow(task['train'][n]['input'], cmap=cmap, norm=norm)
        axs[2*n].axis('off')
        axs[2*n].set_title('Train ' + str(n) + ' Input')
        axs[2*n+1].imshow(task['train'][n]['output'], cmap=cmap, norm=norm)
        axs[2*n+1].axis('off')
        axs[2*n+1].set_title('Train ' + str(n) + ' output')

    for n in range(test_len):
        axs[train_len*2].imshow(task['test'][n]['input'], cmap=cmap, norm=norm)
        axs[train_len*2].axis('off')
        axs[train_len*2].set_title('Test Input')
        axs[train_len*2 + 1].imshow(task['test'][n]['output'], cmap=cmap, norm=norm)
        axs[train_len*2 + 1].axis('off')
        axs[train_len*2 + 1].set_title('Test Output')
        if type(testpreds) == list:
            axs[train_len*2 + 2].imshow(testpreds[n], cmap=cmap, norm=norm)
            axs[train_len*2 + 2].axis('off')
            axs[train_len*2 + 2].set_title('Our prediction')
        elif type(testpreds) == dict:
            iter = 0
            for k, v in testpreds.items():
                axs[train_len*2 + 2 + iter].imshow(testpreds[k][1][0], cmap=cmap, norm=norm)
                axs[train_len*2 + 2 + iter].axis('off')
                axs[train_len*2 + 2 + iter].set_title(k)
                iter += 1
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


def get_value_map(in_, out_): # for cells that change
    value_map = {}
    # we can't get a value map unless the shapes are equal
    if in_.shape == out_.shape:
        target_indices = np.argwhere(in_ != out_)
        for n in target_indices:
            if in_[n[0], n[1]] not in value_map.keys():
                value_map[in_[n[0], n[1]]] = set()
            value_map[in_[n[0], n[1]]].add(out_[n[0], n[1]])

    return value_map

def get_similars(in_, out_): # for cells that stay
    value_map = {}
    # we can't get a value map unless the shapes are equal
    if in_.shape == out_.shape:
        target_indices = np.argwhere(in_ == out_)
        for n in target_indices:
            if in_[n[0], n[1]] not in value_map.keys():
                value_map[in_[n[0], n[1]]] = set()
            value_map[in_[n[0], n[1]]].add(out_[n[0], n[1]])

    return value_map


def get_target_values(value_map):
    all_values = []
    for k, v in value_map.items():
        all_values = all_values + list(v)

    all_values = np.array(all_values)
    val_set = set(all_values)
    return val_set

def get_target_distribution(value_map):
    all_values = []
    for k, v in value_map.items():
        all_values = all_values + list(v)

    all_values = np.array(all_values)
    val_freq = np.bincount(all_values)
    return val_freq

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

    return sorted(list(overall_set))

def process_diff_spatial(in_, out_, in_bg):
    in_mask = np.where(in_ != in_bg, True, False)
    out_mask = np.where(in_ != out_, True, False)
    out_extend_mask = np.where((in_ != out_) & (in_ != in_bg) & ( out_ != in_bg), True, False)

    overall_set = set()
    for n in range(in_mask.shape[0]):
        for m in range(in_mask.shape[1]):
            overall_set.add(coder1[saver1[(in_mask[n, m], out_mask[n, m], out_extend_mask[n, m])]])

    return sorted(list(overall_set))


def get_non_bg_set_situation(in_val_list, out_val_list, bg):
    in_val_list = [x for x in in_val_list if x != bg]
    out_val_list = [x for x in out_val_list if x != bg]
    in_val_nonbg_set = set(in_val_list)
    out_val_nonbg_set = set(out_val_list)
    return in_val_nonbg_set.issubset(out_val_nonbg_set)


def get_col_to_token_class_C(in_val_list, out_val_list, couple_val_map, bg):
    in_val_list = [x for x in in_val_list if x != bg]
    out_val_list = [x for x in out_val_list if x != bg]
    in_val_nonbg_set = sorted(list(set(in_val_list)))
    out_val_nonbg_set = sorted(list(set(out_val_list)))

    nonbg_target_val_set = get_target_values(couple_val_map)

    # get your vol_to_token and token_to_col
    col_to_token = {}
    col_to_token[bg] = 'bg'

    #print(col_to_token)
    for n in range(len(in_val_nonbg_set)):
        #print('A')
        if in_val_nonbg_set[n] not in col_to_token.keys() and (in_val_nonbg_set[n] in nonbg_target_val_set or in_val_nonbg_set[n] in couple_val_map.keys()): #we are focusing on what changes
            cur_len = len(col_to_token)
            col_to_token[in_val_nonbg_set[n]] = 'nonbg' + str(cur_len-1)
            #print(col_to_token)

    for n in range(len(out_val_nonbg_set)):
        #print('B')
        if out_val_nonbg_set[n] not in col_to_token.keys() and (out_val_nonbg_set[n] in nonbg_target_val_set or out_val_nonbg_set[n] in couple_val_map.keys()):
            cur_len = len(col_to_token)
            col_to_token[out_val_nonbg_set[n]] = 'nonbg' + str(cur_len-1)
            #print(col_to_token)

    return col_to_token


def get_token_to_color_class_C(col_to_token):
    return {v: k for k, v in col_to_token.items()}


def get_problem_statement_class_C(col_to_token, couple_value_map):
#     print('col_to_token is:' , col_to_token)
#     print('couple_value_map is: ', couple_value_map)
    problem_statement = []
    for k, v in couple_value_map.items():
        value_list = sorted(list(v))
#         print('k is: ', str(k))
#         print(value_list)
        for n in value_list:
#             print(n)
#             print(col_to_token[k])
#             print(col_to_token[n])
            problem_statement.append((col_to_token[k], col_to_token[n]))

    return sorted(problem_statement)

def get_num_nonbg(token_to_colors):
    keys = list(token_to_colors.keys())
    nonbg_keys = ['nonbg' in x for x in keys]
    return len([x for x in nonbg_keys if x])

def is_only_one_nonbg_per_couple(traininputs_vals, trainoutputs_vals, bg):
    traininputs_vals_here = deepcopy(traininputs_vals)
    trainoutputs_vals_here = deepcopy(trainoutputs_vals)
    for n in range(len(traininputs_vals_here)):
        traininputs_vals_here[n] = [x for x in traininputs_vals_here[n] if x != bg]
    for n in range(len(trainoutputs_vals_here)):
        trainoutputs_vals_here[n] = [x for x in trainoutputs_vals_here[n] if x != bg]

    len_check = all([len(x) == 1 and len(y) == 1 and x[0] == y[0] for x, y in zip(traininputs_vals_here, trainoutputs_vals_here)])

    return len_check

def is_same_nonbgs_per_couple(traininputs_vals, trainoutputs_vals, bg):
    traininputs_vals_here = deepcopy(traininputs_vals)
    trainoutputs_vals_here = deepcopy(trainoutputs_vals)
    for n in range(len(traininputs_vals_here)):
        traininputs_vals_here[n] = [x for x in traininputs_vals_here[n] if x != bg]
    for n in range(len(trainoutputs_vals_here)):
        trainoutputs_vals_here[n] = [x for x in trainoutputs_vals_here[n] if x != bg]

    len_check = all([x == y for x, y in zip(traininputs_vals_here, trainoutputs_vals_here)])

    return len_check

def is_same_nonbgs_per_couple(traininputs_vals, trainoutputs_vals, bg):
    traininputs_vals_here = deepcopy(traininputs_vals)
    trainoutputs_vals_here = deepcopy(trainoutputs_vals)
    for n in range(len(traininputs_vals_here)):
        traininputs_vals_here[n] = [x for x in traininputs_vals_here[n] if x != bg]
    for n in range(len(trainoutputs_vals_here)):
        trainoutputs_vals_here[n] = [x for x in trainoutputs_vals_here[n] if x != bg]

    len_check = all([x == y for x, y in zip(traininputs_vals_here, trainoutputs_vals_here)])

    return len_check


def get_coordinates_of_tuple(x, in_, out_):
    if type(in_) == list:
        coordinates = []
        for n in range(len(in_)):
            this_in_ = in_[n]
            this_out_ = out_[n]
            target_indices = np.argwhere((this_in_ == x[0]) & (this_out_ == x[1]))
            coordinates.append(target_indices)
        return coordinates
    else:
        target_indices = np.argwhere((in_ == x[0]) & (out_ == x[1]))
        return [target_indices]

def retokenize(x, token_to_colors):
    if x[0] in token_to_colors.keys() and x[1] in token_to_colors.keys():
        if type(token_to_colors[x[0]]) == int and type(token_to_colors[x[1]]) == int:
            return(token_to_colors[x[0]], token_to_colors[x[1]])
    else:
        return ['Issue with retokenization.']

def generate_set_assignemtns_per_graph(problem_graph, token_to_colors, in_, out_):
     # specificaaly pick indices and tuples by category: int_anchor, token_anchor, difference
    int_anchors_indices = [i for i, val in enumerate(problem_graph) if val[0] not in token_to_colors.keys()]
    int_anchors = [problem_graph[x] for x in int_anchors_indices]

    token_anchors_indices = [i for i, val in enumerate(problem_graph) if val[0] == val[1] and type(val[0]) == str]
    token_anchors = [problem_graph[x] for x in token_anchors_indices]

    differences_indices = [i for i, val in enumerate(problem_graph) if val[0] != val[1] and type(val[0]) == str]
    differences = [problem_graph[x] for x in differences_indices]
    # set generation with cooridinates in all couples
    assignments_leads = sorted(list(set([x[0] for x in differences])))
    num_assignemnts  = len(assignments_leads)
    asssignments_output = []

    for n in range(num_assignemnts):
        this_assignemnt = {}
        for x in int_anchors:
            this_assignemnt[('int_anchor', x)] = get_coordinates_of_tuple(x, in_, out_)
        for x in token_anchors:
            if x[0] == assignments_leads[n]:
                this_assignemnt[('token_anchor', x)] = get_coordinates_of_tuple(retokenize(x, token_to_colors), in_, out_)
            #elif x[0] != assignments_leads[n]:
                #this_assignemnt[('token_nonanchor', x)] = get_coordinates_of_tuple(retokenize(x, token_to_colors), in_, out_)
        for x in differences:
            if x[0] == assignments_leads[n]:
                this_assignemnt[('diff_anchor', x)] = get_coordinates_of_tuple(retokenize(x, token_to_colors), in_, out_)
        asssignments_output.append(this_assignemnt)

    return assignments_leads, asssignments_output
