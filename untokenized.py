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
from operator import and_, or_
from functools import reduce
from collections import namedtuple
import random

from neighbors import *

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

def get_coords_for_vals(in_, bg):
    # print('get coords of vals in_', in_)
    freqs =  np.unique(in_)
    val_to_coord = {}
    for m in freqs:
        val_to_coord[m] = get_coordinates_from_arr(m, in_)


    del val_to_coord[bg]
    # print('val_to_coord', val_to_coord)
    return val_to_coord

def merge_intersecting_pairs(intersecting_pairs_list):
    to_exclude = []
    for n in range(len(intersecting_pairs_list) - 1):
        is_intersect = len(set(intersecting_pairs_list[n]).intersection(intersecting_pairs_list[n+1]))
        if is_intersect > 0:
            to_exclude.append(n)
            intersecting_pairs_list[n+1] = set(intersecting_pairs_list[n]).union(intersecting_pairs_list[n+1])

    return [set(intersecting_pairs_list[i]) for i in range(len(intersecting_pairs_list)) if i not in to_exclude]


Component = namedtuple("Component", "lead_val x_start x_stop y_start y_stop x_dim y_dim overall_size num_lead_val_coords num_non_lead_val_coords lead_percent unique_vals unique_vals_counts lead_val_coords lead_val_coords_sign1, lead_val_coords_sign2" )

def extract_color_continious(in_, bg):
    # this will extract all components based on color contguity
    val_to_coord = get_coords_for_vals(in_, bg)
    singles = set()
    val_to_objects = {}

    for k, v in list(val_to_coord.items()):
        nb_results = [neighbor_situation(in_, x[0], x[1]) for x in v]
        check0 = [k in nb_results[x][3].keys() for x in range(len(v))]
        for n in range(len(check0)):
            if check0[n] and n == 0:
                val_to_objects[(k, tuple(v[n]))] = [v[n].tolist()]
            elif check0[n] and n > 0:
                current_neighbors = nb_results[n][2][k]
                assigned = False
                for key, value in list(val_to_objects.items()):
                    intersect = [list(x) for x in set(tuple(x) for x in current_neighbors).intersection(set(tuple(x) for x in value))]
                    if len(intersect) > 0:
                        val_to_objects[key].append(v[n].tolist())
                        assigned = True
                if not assigned:
                    val_to_objects[(k, tuple(v[n]))] = [v[n].tolist()]
            else:
                val_to_objects[(k, tuple(v[n]))] = [v[n].tolist()]
                singles.add((k, tuple(v[n])))

    # fix issues related to having no control over the order of testing coordinates
    reviewed = {}
    list_of_keys = list(val_to_objects.keys())
    values_of_keys = [x[0] for x in list_of_keys]
    unique_values_of_keys = sorted(np.unique(np.array(values_of_keys)).tolist())
    for q in unique_values_of_keys:
        keys_of_q = [x for x in list_of_keys if x[0] == q]
        sets_of_q = [set(tuple(x) for x in val_to_objects[key]) for key in keys_of_q]

        if len(keys_of_q) > 1:
            combs = sorted(list(combinations([x for x in range(len(sets_of_q))], 2)))

            combs_intersections = [sets_of_q[x[0]].intersection(sets_of_q[x[1]]) for x in combs]
            comb_intersections_length = [len(x) > 0 for x in combs_intersections]
            found_intersecting_pairs = [combs[i] for i in range(len(combs)) if comb_intersections_length[i]]
            intersecting_collections = merge_intersecting_pairs(found_intersecting_pairs)
            if len(intersecting_collections) > 0:
                non_intersecting = [i for i in range(len(sets_of_q)) if i not in reduce(or_, intersecting_collections)]
            else:
                non_intersecting = [i for i in range(len(sets_of_q))]

            for l in non_intersecting:
                reviewed[keys_of_q[l]] = val_to_objects[keys_of_q[l]]
            for l in intersecting_collections:
                reviewed[keys_of_q[min(l)]] = reduce(or_, [sets_of_q[d] for d in l])
        else:
            reviewed[keys_of_q[0]] = val_to_objects[keys_of_q[0]]

    # get the signatures of your objects, collect these signatures and label them or not
    vals_to_object_signature1 = {}
    vals_to_object_signature2 = {}
    for k, v in list(reviewed.items()):
        if len(v) > 1:
            vals_to_object_signature1[k] = [neighbor_situation(in_, x[0], x[1])[3][k[0]] for x in v]
            vals_to_object_signature2[k] = [neighbor_situation(in_, x[0], x[1])[1][k[0]] for x in v]
    # configure objects, must be done:
    trimmed_objects = []
    #print('reviewed:', reviewed)
    for k, v in reviewed.items():
        v = list(v)
        v.sort(key = lambda x:(x[0], x[1]))
        x_start = min([x[0] for x in v])
        x_stop = max([x[0] for x in v]) + 1
        y_start = min([x[1] for x in v])
        y_stop = max([x[1] for x in v]) + 1
        x_dim = x_stop - x_start
        y_dim = y_stop - y_start
        overall_size = x_dim*y_dim
        lead_percent = round(len(v) / overall_size, 3)
        if len(v) == 1:
            sign1, sign2 = None, None
        elif len(v) > 1:
            sign1, sign2 = vals_to_object_signature1[k], vals_to_object_signature2[k]
        this_object = in_[x_start:x_stop, y_start:y_stop]
        freqs = np.unique(this_object, return_counts = True)
        vals, counts = sort_two_lists_based_on_second(freqs[0], freqs[1])

        trimmed_objects.append(Component(k[0], x_start, x_stop, y_start, y_stop, x_dim, y_dim, overall_size, len(v), overall_size - len(v), lead_percent, vals, counts, v, sign1, sign2))
    return trimmed_objects # trimmed objects represent all the single color contigious components.

def get_dims_objects(object_lists):
    object_dims = []
    for n in range(len(object_lists)):
        this_case = object_lists[n]
        object_dims.append([[x.x_dim, x.y_dim] for x in this_case])
    return object_dims

def sort_objects_dims(objects_list):
    sorted_objects_by_size = []
    for m in objects_list:
        m = sorted(m, key=lambda x: (x.overall_size, x.lead_percent)) # (x.lead_percent, x.overall_size)
        sorted_objects_by_size.append(m)
    return sorted_objects_by_size

def two_points_all_nb(point1, point2):
    # each point is a (x, y) coord
    point_1_nbs = bare_nb_coordinates(point1[0], point1[1])
    point_1_nbs.remove((point1[0], point1[1]))
    if tuple(point2) in point_1_nbs:
        return True
    else:
        return False

def do_spatial_overlap(comp1, comp2):
    # your components have the coordinates, the goal is to find a point of overlap or a point of contact,
    # define a function for when two points are in contact, this is supposed to be in neighbor search
    # two have same lead_val, they are spatially isolated by default:
    if(comp1.lead_val > comp2.lead_val):
        return False
    # one component above the other
    if(comp1.x_start > comp2.x_stop or comp2.x_start > comp1.x_stop):
        return False
    # one component to the side of the other
    if(comp1.y_start > comp2.y_stop or comp2.y_start > comp1.y_stop):
        return False

    # we need to make sure none of the specific lead values are in touch
    # if we don't do this, we miss on [36, 41, 71, 96, 98, 106, 108, 136, 142]
    l1 = comp1.lead_val_coords
    l2 = comp2.lead_val_coords
    for coord1 in l1:
        for coord2 in l2:
            if two_points_all_nb(coord1, coord2):
                return True
    return False

def extract_spatial_continious(in_color_components):
    connected_couples = []
    for n in range(len(in_color_components)):
        for m in range(len(in_color_components)):
            if (m, n) not in connected_couples and n != m:
                comp1 = in_color_components[n]
                comp2 = in_color_components[m]
                current_overlap = do_spatial_overlap(comp1, comp2)
                if current_overlap:
                    connected_couples.append((n, m))
    connected_couples.sort(key = lambda x:(x[0], x[1]))
    #connected_couples = merge_intersecting_pairs(connected_couples)
    # merge 5 times
    n = 0
    while n < len(in_color_components):
        if n >= 3: # This can be done programmatically by testing for any intersection between any lists and do some intense bookkeeping, do this latter
            connected_couples = random.sample(connected_couples, len(connected_couples))
        connected_couples = merge_intersecting_pairs(connected_couples)
        n += 1

    # add single unmapped components, now, we have to have the number of spatial components smaller than or equal to the number of color components, doesn't really get any basic more than this.
    whole = [False] * len(in_color_components)
    if len(connected_couples) > 0:
        mapped_components = set(set.union(*map(set, connected_couples)))
    else:
        mapped_components = {}

    for n in range(len(in_color_components)):
        if n in mapped_components:
            whole[n] = True
    to_add = [i for i in range(len(whole)) if whole[i] == False]
    for add_it in to_add:
        connected_couples.append({add_it})

    return connected_couples

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
