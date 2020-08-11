import os
import pickle
from types import FunctionType
from inspect import getmembers
from copy import deepcopy
import json
from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import colors
import numpy as np

import itertools
from itertools import permutations, combinations, product

from functools import reduce
import random
from collections import Counter
from scipy.ndimage import find_objects

# First stochastic
import random

from deductive import *

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

# The following two functions to show all attributes of an object
def api(obj):
    return [name for name in dir(obj) if name[0] != '_']

def attrs(obj):
     disallowed_properties = { name for name, value in getmembers(type(obj))  if isinstance(value, (property, FunctionType))}
     return {name: getattr(obj, name) for name in api(obj) if name not in disallowed_properties and hasattr(obj, name)}

# In order to work with all cases, use the following function instead of np.all
# load set
def load_set(path):
    cases = []
    tasks = sorted(os.listdir(path))

    for n in tasks:
        task_file = str(path /  n)

        with open(task_file, 'r') as f:
            task = json.load(f)
            cases.append(task)
    return cases

# critical for processing further
def fix_dim(arr):
    if type(arr) == int:
        arr = [[arr]]
    if type(arr) == list:
        arr = np.array(arr)
    if type(arr) != np.ndarray:
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
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm = colors.Normalize(vmin=0, vmax=9)
    train_len = len(task['train'])
    test_len = len(task['test'])
    #print('train_len: ', train_len, ' test_len: ', test_len)
    fig_dim = 0
    if type(testpreds) == list:
        fig_dim = train_len*2 + test_len*3


    fig, axs = plt.subplots(1, fig_dim, figsize=(fig_dim + 10,  fig_dim + 10))

    #print(fig_dim)
    for n in range(train_len):
        axs[2*n].imshow(task['train'][n]['input'], cmap=cmap, norm=norm)
        axs[2*n].axis('off')
        axs[2*n].set_title('Train ' + str(n) + ' Input')
        axs[2*n+1].imshow(task['train'][n]['output'], cmap=cmap, norm=norm)
        axs[2*n+1].axis('off')
        axs[2*n+1].set_title('Train ' + str(n) + ' output')

    for n in range(test_len):
        axs[train_len*2 + (3*n)].imshow(task['test'][n]['input'], cmap=cmap, norm=norm)
        axs[train_len*2 + (3*n)].axis('off')
        axs[train_len*2 + (3*n)].set_title('Test Input')
        axs[train_len*2 + 1 + (3*n)].imshow(task['test'][n]['output'], cmap=cmap, norm=norm)
        axs[train_len*2 + 1 + (3*n)].axis('off')
        axs[train_len*2 + 1 + (3*n)].set_title('Test Output')
        if type(testpreds) == list:
            axs[train_len*2 + 2 + (3*n)].imshow(testpreds[n], cmap=cmap, norm=norm)
            axs[train_len*2 + 2 + (3*n)].axis('off')
            axs[train_len*2 + 2+ (3*n)].set_title('Our prediction')

    plt.tight_layout()
    plt.show()

def is_list_of_list(this_list):
    if len(this_list) > 0 and len(this_list[0]) > 0:
        return type(this_list) == list and type(this_list[0]) == list
    else:
        return False

def is_list_of_list_of_list(this_list):
    if len(this_list) > 0 and len(this_list[0]) > 0:
        return type(this_list) == list and type(this_list[0]) == list and type(this_list[0][0]) == list
    else:
        return False

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

def get_value_map(in_, out_, in_vals, out_vals): # for cells that change
    value_map = {}
    # we can't get a value map unless the shapes are equal
    if in_.shape == out_.shape:
        target_indices = np.argwhere(in_ != out_)
        for n in target_indices:
            if in_[n[0], n[1]] not in value_map.keys():
                value_map[in_[n[0], n[1]]] = set()
            value_map[in_[n[0], n[1]]].add(out_[n[0], n[1]])
    else:
        in_vals = set(in_vals)
        out_vals = set(out_vals)
        commons = list(in_vals.intersection(out_vals)) # k : {v}
        in_extras = list(in_vals - out_vals)
        out_extras = list(out_vals - in_vals)

        for n in commons:
            value_map[n] = set()
            value_map[n].add(n)
        if len(in_extras) > 0:
            for n in in_extras:
                value_map[n] = set()
                for m in out_extras:
                    value_map[n].add(m)
        elif len(in_extras) == 0 and len(out_extras) > 0:
            for m in out_extras:
                value_map[-1] = set()
                value_map[-1].add(m)

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

    for k, v in global_vm.items():
        global_vm[k] = list(v)
    return global_vm

# get cell neighbours
# https://www.kaggle.com/arsenynerinovsky/cellular-automata-as-a-language-for-reasoning
nbh = lambda x, i, j: { #x is array, i and j are row and column indices
    (i+ip, j+jp) : x[i+ip, j+jp]
        for ip, jp in itertools.product([1, -1, 0], repeat=2)
            if (0 <= i < x.shape[0]) and (0 <= j < x.shape[1]) and (0 <= i+ip < x.shape[0]) and (0 <= j+jp < x.shape[1])
}

# get a background
def get_background(arr):
    vals, counts = np.unique(arr, return_counts = True)
    vals = vals.tolist()
    counts = counts.tolist()
    max_index = counts.index(max(counts))
    return vals[max_index]

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

def get_common_nonbg(lists):
    # s = None
    # for e in traininputs_vals:
    # if not s:
    #     s = set(e)
    # else:
    #     s &= set(e)
    return reduce((lambda x,y: x & y), map(set, lists)) #s

def get_size_sorted_nonbg_vals_test(remaining_test_nonbg, testinput):
    modified_combs = set()
    for n in remaining_test_nonbg:
        modified_combs.add((n, np.count_nonzero(testinput == n)))
    modified_combs = sorted(list(modified_combs), key=lambda tup: tup[1])
    modified_combs = [x[0] for x in modified_combs]
    return modified_combs

def get_col_to_token(arr_, in_val_list, out_val_list, couple_val_map, bg, priority_nonbg):
    col_to_token = {}
    if type(bg) == int:
        priority_nonbg_list = sorted(list(set([x for x in priority_nonbg if x != bg])))
        in_val_list = [x for x in in_val_list if x != bg]
        out_val_list = [x for x in out_val_list if x != bg]
        in_val_nonbg_set = sorted(list(set(in_val_list)))
        out_val_nonbg_set = sorted(list(set(out_val_list)))
        nonbg_target_val_set = get_target_values(couple_val_map)
        col_to_token[bg] = 'bg'

    elif type(bg) == list:
        priority_nonbg_list = sorted(priority_nonbg)
        in_val_nonbg_set = sorted(list(set(in_val_list)))
        out_val_nonbg_set = sorted(list(set(out_val_list)))
        nonbg_target_val_set = get_target_values(couple_val_map)

    for n in range(len(priority_nonbg_list)):
        if priority_nonbg_list[n] not in col_to_token.keys() and (priority_nonbg_list[n] in nonbg_target_val_set or priority_nonbg_list[n] in couple_val_map.keys()): #we are focusing on what changes
            cur_len = len(col_to_token)
            col_to_token[priority_nonbg_list[n]] = 'nonbg' + str(cur_len-1) + 'pr'

    # Arrange your nonbg labelling based on size
    in_val_nonbg_set = sort_nonbg_by_size_in_arr(in_val_nonbg_set, arr_)
    for n in range(len(in_val_nonbg_set)):
        if in_val_nonbg_set[n] not in col_to_token.keys() and (in_val_nonbg_set[n] in nonbg_target_val_set or in_val_nonbg_set[n] in couple_val_map.keys()): #we are focusing on what changes
            cur_len = len(col_to_token)
            col_to_token[in_val_nonbg_set[n]] = 'nonbg' + str(cur_len-1)

    for n in range(len(out_val_nonbg_set)):
        if out_val_nonbg_set[n] not in col_to_token.keys() and (out_val_nonbg_set[n] in nonbg_target_val_set or out_val_nonbg_set[n] in couple_val_map.keys()):
            cur_len = len(col_to_token)
            col_to_token[out_val_nonbg_set[n]] = 'nonbg' + str(cur_len-1)

    return col_to_token

def get_col_to_token_test(final_test_nonbg, pr_tokens, bg, size_sorted_nonbg, int_anchor_vals, testinput):
    int_anchor_vals = [x for x in int_anchor_vals if len(x) > 0]
    if len(int_anchor_vals) > 0:
        int_anchor_vals = list(reduce((lambda z,y: z & y), map(set, int_anchor_vals)))

    col_to_token = {}

    if type(bg) == int:
        final_test_nonbg = [x for x in final_test_nonbg if x != bg and x not in int_anchor_vals]
        col_to_token[bg] = 'bg'

    elif type(bg) == list:
        final_test_nonbg = [x for x in final_test_nonbg if x not in int_anchor_vals]

    for key in pr_tokens.keys():
        col_to_token[key] = pr_tokens[key]
        if key in final_test_nonbg:
            final_test_nonbg.remove(key)

    if len(size_sorted_nonbg) > 0:
        size_sorted_vals = get_size_sorted_nonbg_vals_test(final_test_nonbg, testinput)
        for n in range(len(size_sorted_vals)):
            if size_sorted_vals[n] not in col_to_token.keys() and n < len(size_sorted_nonbg):
                col_to_token[size_sorted_vals[n]] = size_sorted_nonbg[n]

    # arrange based on size
    final_test_nonbg = sort_nonbg_by_size_in_arr(final_test_nonbg, testinput)
    for n in range(len(final_test_nonbg)):
        if final_test_nonbg[n] not in col_to_token.keys():
            cur_len = len(col_to_token)
            col_to_token[final_test_nonbg[n]] = 'nonbg' + str(cur_len-1)

    return col_to_token

def get_token_to_color(col_to_token):
    return {v: k for k, v in col_to_token.items()}

def get_problem_statement(col_to_token, couple_value_map):
    problem_statement = []
    for k, v in couple_value_map.items():
        value_list = sorted(list(v))
        if len(value_list) > 0:
            for n in value_list:
                if k != -1:
                    problem_statement.append((col_to_token[k], col_to_token[n]))
                elif k == -1:
                    problem_statement.append(('nil', col_to_token[n]))
        elif len(value_list) == 0:
            problem_statement.append((col_to_token[k], 'nil'))

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

def get_coordinates_of_tuple(x, in_, out_):

    if type(in_) == list and len(in_) == len(out_) and len(in_[0]) == len(out_[0]):
        coordinates = []
        for n in range(len(in_)):
            this_in_ = in_[n]
            this_out_ = out_[n]
            target_indices = np.argwhere((this_in_ == x[0]) & (this_out_ == x[1]))
            coordinates.append(target_indices)
        return coordinates

    elif type(in_) == list and (len(in_) != len(out_) or len(in_[0]) != len(out_[0])):
        coordinates = []
        for n in range(len(in_)):
            this_in_ = in_[n]
            target_indices = (np.argwhere((this_in_ == x[0])), np.argwhere((this_in_ == x[0])))
            coordinates.append(target_indices)
        return coordinates

    elif in_.shape[0] == out_.shape[0] and in_.shape[1] == out_.shape[1]:
        target_indices = np.argwhere((in_ == x[0]) & (out_ == x[1]))
        return [target_indices]

    elif in_.shape[0] != out_.shape[0] or in_.shape[1] != out_.shape[1]:
        target_indices = (np.argwhere(in_ == x[0]), np.argwhere(out_ == x[1]))
        return [target_indices]

def get_coordinates_from_arr(x, in_):
    if type(in_) == list:
        coordinates = []
        for n in range(len(in_)):
            this_in_ = in_[n]
            target_indices = np.argwhere(this_in_ == x)
            coordinates.append(target_indices)
        return coordinates
    else:
        target_indices = np.argwhere(in_ == x)
        return target_indices

def get_len_coordinates_from_arr(x, in_):
    if type(in_) == list:
        coordinates = []
        for n in range(len(in_)):
            this_in_ = in_[n]
            target_indices = np.argwhere(this_in_ == x)
            coordinates.append([target_indices])
        return [len(x) for x in coordinates]
    else:
        target_indices = np.argwhere(in_ == x)
        return len(target_indices)

def sort_nonbg_by_size_in_arr(nonbg_list, arr_):
    results = [get_len_coordinates_from_arr(x, arr_) for x in nonbg_list]
    created_tuple = [(x, y) for x, y in zip(nonbg_list, results)]
    created_tuple = sorted(created_tuple, key=lambda tup: tup[1])
    return [x[0] for x in created_tuple]

def retokenize(x, token_to_colors):
    if x[0] in token_to_colors.keys() and x[1] in token_to_colors.keys():
        if type(token_to_colors[x[0]]) == int and type(token_to_colors[x[1]]) == int:
            return(token_to_colors[x[0]], token_to_colors[x[1]])
    elif x[0] == 'nil':
        if type(token_to_colors[x[1]]) == int:
            return(-1, token_to_colors[x[1]])
    elif x[1] == 'nil':
        if type(token_to_colors[x[0]]) == int:
            return(token_to_colors[x[0]], -1)
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
    assignments_leads = sorted(list(set([x[0] for x in differences])))
    num_assignemnts  = len(assignments_leads)
    asssignments_output = []
    int_anchor_vals = []

    for n in range(num_assignemnts):
        this_assignemnt = {}
        for x in int_anchors:
            this_assignemnt[('int_anchor', x)] = get_coordinates_of_tuple(x, in_, out_)
            int_anchor_vals.append(x[0])

        for x in token_anchors:
            if x[0] == assignments_leads[n]:
                this_assignemnt[('token_anchor', x)] = get_coordinates_of_tuple(retokenize(x, token_to_colors), in_, out_)
        for x in differences:
            if x[0] == assignments_leads[n]:
                this_assignemnt[('diff_anchor', x)] = get_coordinates_of_tuple(retokenize(x, token_to_colors), in_, out_)
        asssignments_output.append(this_assignemnt)

    return assignments_leads, int_anchor_vals, asssignments_output

def is_list_one_value(list_ex):
    return all([x == list_ex[0] for x in list_ex])

def is_list_equal_1(list_ex):
    return all([x == 1 for x in list_ex])

def get_set_of_values_1(this_dict):
    values = set()
    target = this_dict[1]
    for n in range(len(target)):
        values.add(target[n][0])
    return values

def get_this_objective(cur_graph, assignments_leads, is_similar_dimension):
    cur_graph = sorted([x for x in cur_graph if type(x[0]) == str or (type(x[0]) == tuple and type(x[0][0]) == str)])
    graph_similars = [x for x in cur_graph if x[0] == x[1]]
    graph_diffs = [x for x in cur_graph if x[0] != x[1]]

    bare_assignment_leads = get_bare_assignment_leads(assignments_leads)

    desired_combs = []
    if len(bare_assignment_leads) > 0 and is_similar_dimension:
        for n in sorted(list(bare_assignment_leads)):
            this_n_similars = [x for x in graph_similars if x[0] == n]
            this_n_diffs = [x for x in graph_diffs if x[0] == n]
            if len(this_n_similars) > 0 and len(this_n_diffs) > 0:
                this_n = this_n_similars + this_n_diffs
                if len(this_n) == 2:
                    nl = []
                    nl.append(this_n[0])
                    nl.append(this_n[1])
                    desired_combs.append(nl)
                elif len(this_n) > 2:
                    this_n_combs = sorted(list(combinations(sorted(this_n), 2)))
                    for x in this_n_combs:
                        nl = []
                        nl.append(x[0])
                        nl.append(x[1])
                        desired_combs.append(nl)
            elif len(this_n_similars) == 0 and len(this_n_diffs) == 1:
                for x in this_n_diffs:
                    y = (x[0], x[1], 'direct')
                    nl = []
                    nl.append(y)
                    desired_combs.append(nl)
            elif len(this_n_similars) == 0 and len(this_n_diffs) > 1:
                this_n_combs = sorted(list(combinations(sorted(this_n_diffs), 2)))
                for x in this_n_combs:
                    nl = []
                    nl.append(x[0])
                    nl.append(x[1])
                    desired_combs.append(nl)
            elif len(this_n_similars) > 0 and len(this_n_diffs) == 0:
                for x in this_n_similars:
                    y = (x[0], x[1], 'direct')
                    nl = []
                    nl.append(y)
                    desired_combs.append(nl)

    elif len(bare_assignment_leads) == 0 or not is_similar_dimension:
        if len(graph_diffs) > 0:
            for x in graph_diffs:
                y = (x[0], x[1], 'direct')
                nl = []
                nl.append(y)
                desired_combs.append(nl)

        if len(graph_similars) > 0:
            for x in graph_similars:
                y = (x[0], x[1], 'direct')
                nl = []
                nl.append(y)
                desired_combs.append(nl)

    return sorted(desired_combs, key = len)

def create_an_objective_dict(objective):
    objective_dict = {}
    for n in range(len(objective)):
        this_objective = objective[n]
        for x in this_objective:
            if type(x) == tuple:
                objective_dict[x[0]] = x[1]
    return objective_dict

def retokenize_objective_dict(objective_dict, test_token_to_color):
    retokenized = {}
    for k, v in objective_dict.items():
        retokenized[test_token_to_color[k]] = test_token_to_color[v]
    return retokenized

def get_priority_token(token_to_colors):
    pr_tokens = {}
    for n in token_to_colors:
        for k, v in n.items():
            if 'pr' in k:
                pr_tokens[v] = k
    return pr_tokens

def list_comparator(l1, l2):
    if len(l1) == len(l2):
        if all([x > y for x, y in zip(l1, l2)]):
            return '>'
        elif all([x < y for x, y in zip(l1, l2)]):
            return '<'
        elif all([x == y for x, y in zip(l1, l2)]):
            return '=='
        elif l1[0] == l2[0] and l1[1] > l2[1]:
            return '=>'
        elif l1[0] == l2[0] and l1[1] < l2[1]:
            return '=<'
        elif l1[0] > l2[0] and l1[1] == l2[1]:
            return '>='
        elif l1[0] < l2[0] and l1[1] == l2[1]:
            return '<='
        else:
            return None
    else:
        return None

def list_modulo(l1, l2):
    if len(l1) == len(l2):
        if all([x % y == 0 or y % x == 0 for x, y in zip(l1, l2)]):
            return True
        else:
            return False
    else:
        return False

# this is a terminal decision taken after a series of previous decisions
def get_int_div(l1, l2):
        return [y / x for x, y in zip(l1, l2)]

def modify_dimensiosn(dimensions_list, multi_factor_list):
    dimensions_list = deepcopy(dimensions_list)
    if len(dimensions_list[0]) == len(multi_factor_list):
        for n in range(len(dimensions_list)):
            dimensions_list[n] = [int(x * y) for x, y in zip(dimensions_list[n], multi_factor_list)]
        return dimensions_list
    else:
        return None

def is_all_nonbg_in_cur_obj(Current_objective):
    for n in range(len(Current_objective)):
        this_obj = Current_objective[n]
        for x in this_obj:
            if type(x) == tuple and x[0] == 'bg':
                return False
    return True

def is_all_nonbg_in_sub_obj(this_obj):
    for x in this_obj:
        if type(x) == tuple and x[0] == 'bg':
            return False
    return True


def get_bare_assignment_leads(assignments_leads):
    bare_assignment_leads = set()
    for n in assignments_leads:
        for x in n:
            bare_assignment_leads.add(x)
    return bare_assignment_leads

def expand_obj(objective, diff_element):
    to_add = []
    this_index = None
    for n in range(len(objective)):
        x = objective[n] # x:  [('nonbg1', 'nonbg1'), ('nonbg1', 'bg')]
        if len(x) == 2:
            if 'nonbg' in x[0][0] and 'pr' not in x[0][0]:
                this_index = n
                break

    if this_index != None:
        x = objective[this_index]
        if x[0][0] != x[0][1]:
            carry = x[0][1]
        else:
            carry = diff_element
        to_add = [(diff_element, carry)]

        if x[1][0] != x[1][1]:
            carry = x[1][1]
        else:
            carry = diff_element
        to_add.append((diff_element, carry))
        objective.append(to_add)

    return sorted(objective)

def contract_obj(objective, diff_element):
    to_add = []
    indices_to_remove = []
    for n in range(len(objective)):
        x = objective[n]
        if len(x) <= 2 and type(x[0]) == tuple and x[0][0] == diff_element:
            indices_to_remove.append(n)

    new_objective = []
    if len(indices_to_remove) > 0:
        for check_ind in range(len(objective)):
            if check_ind not in indices_to_remove:
                new_objective.append(objective[check_ind])

    return sorted(new_objective)

def retrieve_coords_from_assignments(target_tuple, asssignments_output):
    results = []
    for n in range(len(asssignments_output)): # len num_train
        for x in range(len(asssignments_output[n])): # assignment
            this_assignment = asssignments_output[n][x]
            for k, v in this_assignment.items():
                if target_tuple == k[1]:
                    results.append(v[0])
    return results

def retrieve_coords_nonbg_from_arr(nonbg, arr):
    results = []
    for n in range(arr.shape[0]): # len num_train
        for x in range(arr.shape[1]): # assignment
            if arr[n][x] == nonbg:
                results.append((n, x))

    return results

# The following examine simple matrix rotations and mirroring
def get_diagonal_mirror(arr):
    arr = fix_dim(arr)
    return np.fliplr(np.transpose(arr[::-1]))

def get_offdiagonal_mirror(arr):
    arr = fix_dim(arr)
    return np.flipud(np.transpose(arr[::-1]))

# There may be some redundancie in the screen_flips_rotation function, no time to gather test cases
def screen_flips_rotation(in_, out_):
    in_ = fix_dim(in_)
    out_ = fix_dim(out_)

    # get_diagonal_mirror is more precedent than rotation which is more precedent over flipud or fliplr
    if np.all(get_diagonal_mirror(in_) == out_):
        return get_diagonal_mirror, None
    elif np.all(get_offdiagonal_mirror(in_) == out_):
        return get_offdiagonal_mirror, None
    elif np.all(np.rot90(in_, 1, axes = (0, 1)) == out_):
        return np.rot90, 1
    elif np.all(np.rot90(in_, 2, axes = (0, 1)) == out_):
        return np.rot90, 2
    elif np.all(np.rot90(in_, 3, axes = (0, 1)) == out_):
        return np.rot90, 3
    elif np.all(np.flipud(in_) == out_):
        return np.flipud, None
    elif np.all(np.fliplr(in_) == np.array(out_)):
        return np.fliplr, None
    else:
        return None, None
