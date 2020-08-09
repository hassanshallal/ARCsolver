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
from scipy.ndimage import label, find_objects

## Book-keeping, I/O section
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
def plot_task_eval(task, testpreds):
    cmap = colors.ListedColormap(
        ['#000000', '#0074D9', '#FF4136', '#2ECC40', '#FFDC00',
         '#AAAAAA', '#F012BE', '#FF851B', '#7FDBFF', '#870C25'])

    norm = colors.Normalize(vmin=0, vmax=9)
    train_len = len(task['train'])
    test_len = len(task['test'])
    #print('train_len: ', train_len, ' test_len: ', test_len)
    fig_dim = 0
    if type(testpreds) == list and len(testpreds)  > 0:
        fig_dim = train_len*2 + test_len*3
    elif type(testpreds) == list and len(testpreds)  == 0:
        fig_dim = train_len*2 + test_len*2


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
        if type(testpreds) == list and len(testpreds) > 0:
            axs[train_len*2 + 2 + (3*n)].imshow(testpreds[n], cmap=cmap, norm=norm)
            axs[train_len*2 + 2 + (3*n)].axis('off')
            axs[train_len*2 + 2+ (3*n)].set_title('Our prediction')

    plt.tight_layout()
    plt.show()

# get a background
def get_background(arr):
    vals, counts = np.unique(arr, return_counts = True)
    vals = vals.tolist()
    counts = counts.tolist()
    max_index = counts.index(max(counts))
    return vals[max_index]

## first principles prior knowledge encoding
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

def get_int_div(l1, l2):
        return [y / x for x, y in zip(l1, l2)]

def sort_two_lists_based_on_second(l1, l2):
    zipped_lists = zip(l2, l1)
    sorted_pairs = sorted(zipped_lists, reverse = True)
    tuples = zip(*sorted_pairs)
    list2, list1 = [list(tuple) for tuple in  tuples]
    return (list1, list2)

def get_sorted_frequency_situation(x):
    freqs = np.unique(x, return_counts = True)
    return sort_two_lists_based_on_second(freqs[0].tolist(), freqs[1].tolist())

def extract_objects(x):
    _objects = find_objects(x)
    # labels, numobjects = label(x) # we introduced this, it decreses the number from 363 to 375, but it may be letter.
    # _objects = find_objects(labels)
    # if len(_objects) > 4:

    extracted = []
    for obj in _objects:
        if obj != None:
            this_object_data = [list(obj)[0].start, list(obj)[0].stop, list(obj)[1].start, list(obj)[1].stop, (list(obj)[0].stop - list(obj)[0].start), (list(obj)[1].stop - list(obj)[1].start)]
            this_object_in_x = x[this_object_data[0]:this_object_data[0]+this_object_data[4], this_object_data[2]:this_object_data[2]+this_object_data[5]]
            vals, counts = np.unique(this_object_in_x, return_counts = True)
            vals = vals.tolist()
            counts = counts.tolist()
            max_index = counts.index(max(counts))
            min_index = counts.index(min(counts))
            this_object_data = this_object_data + [vals[max_index], vals[min_index], len(vals)]
            extracted.append(this_object_data)
    return extracted

def get_dims_objects(object_lists):
    object_dims = []
    for n in range(len(object_lists)):
        this_case = object_lists[n]
        object_dims.append([x[4:6] for x in this_case])
    return object_dims

def get_freqs_object(object_lists):
    object_freqs = []
    for n in range(len(object_lists)):
        this_case = object_lists[n]
        object_freqs.append([x[6:9] for x in this_case])
    return object_freqs

def sort_objects_dims_by_size(objects_list):
    sorted_objects_by_size = []
    for m in objects_list:
        m.sort(key = lambda x: x[4]*x[5])
        sorted_objects_by_size.append(m)
    return sorted_objects_by_size

def is_out_obj_in_in_objs(out_obj, in_objs):
    for obj in in_objs:
        if obj[1] - obj[0] == out_obj[1] - out_obj[0] and obj[3] - obj[2] == out_obj[3] - out_obj[2]:
            return True
    return False

def direct_obj_movement_detection(out_objs, in_objs):
    if len(out_objs) == len(in_objs):
        for n in range(len(out_objs)):
            if is_out_obj_in_in_objs(out_objs[n], in_objs) == False:
                return False
        return True
    return False

def one_obj_move(in_objs, out_objs):
    moved = []
    for n in range(len(out_objs)):
        if out_objs[n] not in in_objs:
            for m in range(len(in_objs)):
                if out_objs[n][0] != in_objs[m][0] and out_objs[n][1] != in_objs[m][1] and out_objs[n][2] == in_objs[m][2] and out_objs[n][3] == in_objs[m][3]:
                    moved.append(True)
                elif out_objs[n][0] == in_objs[m][0] and out_objs[n][1] == in_objs[m][1] and out_objs[n][2] != in_objs[m][2] and out_objs[n][3] != in_objs[m][3]:
                    moved.append(True)
    return all(moved) and len(moved) > 0

def get_diagonal_mirror(arr):
    arr = fix_dim(arr)
    return np.fliplr(np.transpose(arr[::-1]))

def get_offdiagonal_mirror(arr):
    arr = fix_dim(arr)
    return np.flipud(np.transpose(arr[::-1]))

def get_flips_rotation(in_):
    # get_diagonal_mirror is more precedent than rotation which is more precedent over flipud or fliplr
    diagonal_mirror = get_diagonal_mirror(in_)
    offdiagonal_mirror = get_offdiagonal_mirror(in_)
    rotated90_1 = np.rot90(in_, 1, axes = (0, 1))
    rotated90_2 = np.rot90(in_, 2, axes = (0, 1))
    rotated90_3 = np.rot90(in_, 3, axes = (0, 1))
    flipped_ud = np.flipud(in_)
    flipped_lr = np.fliplr(in_)
    return [in_, diagonal_mirror, offdiagonal_mirror, rotated90_1, rotated90_2, rotated90_3, flipped_ud, flipped_lr]

def convolve_for_a_match(in_, out_, rel):
    #  in_ = np.where(in_ != bg, 1, 0)
    #  out_ = np.where(out_ != bg, 1, 0)
    matches = []
    if rel in ['>', '>=', '=>']:
        situation = 'contraction'
        smaller = get_flips_rotation(out_)
        larger = in_
    elif rel in ['<', '<=', '=<']:
        situation = 'expansion'
        larger = out_
        smaller = get_flips_rotation(in_)
    else:
        return False, matches, 'non_dimension_related'

    if smaller[0].shape == (1, 1):
        return False, matches, 'non_dimension_related'

    for n in range(0, larger.shape[0]):
        for m in range(0, larger.shape[1]):
            to_match = larger[n:n+smaller[0].shape[0], m:m+smaller[0].shape[1]]
            match_test = [np.all(to_match == smaller[x]) for x in range(len(smaller))]
            if to_match.shape == smaller[0].shape and any(match_test):
                matches.append((match_test.index(True), (n, m)))
                # we can jump here, we just want to make sure we do the right jump

    if len(matches) > 0:
        return True, matches, situation

    return False, matches, 'non_dimension_related'

def modify_dimensiosn(dimensions_list, multi_factor_list):
    dimensions_list = deepcopy(dimensions_list)
    if len(dimensions_list[0]) == len(multi_factor_list):
        for n in range(len(dimensions_list)):
            dimensions_list[n] = [int(x * y) for x, y in zip(dimensions_list[n], multi_factor_list)]
        return dimensions_list
    else:
        return None

def screen_basic_math(l1, l2):
    modula = []
    subtr = []
    to_do_modula = 'nil'
    to_do_subtra = 'nil'
    outcome = False
    if len(l1) == len(l2):
        for n in range(len(l1)):
            if list_comparator(l1, l2) in ['>', '>=', '=>']:
                to_do_modula = 'div'
                to_do_subtra = 'subtract'
                modula.append([x // y for x, y in zip(l1[n], l2[n]) if x != 0 and y != 0])
                subtr.append([x - y for x, y in zip(l1[n], l2[n])])
            elif list_comparator(l1, l2) in ['<', '<=', '=<']:
                to_do_modula = 'mult'
                to_do_subtra = 'add'
                modula.append([y // x for x, y in zip(l1[n], l2[n]) if x != 0 and y != 0])
                subtr.append([y - x for x, y in zip(l1[n], l2[n])])
        if len(subtr) == 1 and len(subtr[0]) > 2:
            subtr = subtr[0]
        if len(modula) == 1  and len(modula[0]) > 2:
            modula = modula[0]

        if len(subtr) > 0 and all([x == subtr[0] for x in  subtr]) and (subtr[0] not in [0, [0, 0]]):
            return True, to_do_subtra, subtr[0]
        elif len(modula) > 0 and all([x == modula[0] for x in  modula]) and modula[0] not in [0, 1, [0, 0], [1, 1]]:
            return True, to_do_modula, modula[0]
        return False, '', []

def apply_basic_math(todo, factor, input_list):
    factor_list = [factor] * len(input_list)
    if len(factor) == len(input_list[0]):
        if todo == 'add':
            return (np.array(input_list) + np.array(factor_list)).tolist()
        elif todo == 'subtract':
            return (np.array(input_list) - np.array(factor_list)).tolist()
        elif todo == 'mult':
            return (np.array(input_list) * np.array(factor_list)).tolist()
        elif todo == 'div':
            return (np.array(input_list) // np.array(factor_list)).tolist()

def expand_series(todo, factor, input_list, toadd):
    if type(factor) == list and all([factor[0] == x for x in factor]):
        factor = factor[0]
    if todo == 'add':
        for n in range(toadd):
            input_list.append(input_list[len(input_list)-1] + factor)
    elif todo == 'subtract':
        for n in range(toadd):
            input_list.append(input_list[len(input_list)-1] - factor)
    elif todo == 'mult':
        for n in range(toadd):
            input_list.append(input_list[len(input_list)-1] * factor)
    elif todo == 'div':
        for n in range(toadd):
            input_list.append(input_list[len(input_list)-1] // factor)
    return input_list

def series_analyzer(series_list):
    if len(series_list) < 3:
        return False, series_list
    else:
        l1 = [series_list[0:len(series_list) - 1]]
        l2 = [series_list[1:len(series_list)]]
        is_series, to_do, factor = screen_basic_math(l1, l2)
        if is_series:
            if type(factor) == list:
                new_series = expand_series(to_do, factor[0], series_list, 5)
            elif type(factor) == int:
                new_series = expand_series(to_do, factor, series_list, 5)
            return True, new_series
        else:
            return False, series_list

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
