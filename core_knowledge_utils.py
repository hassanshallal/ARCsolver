from utils import *
from edges import *
from neighbors import *

results_dict = {}
results_dict['ineligible'] = []
results_dict['passed_all_traininputs'] = []
results_dict['passed_some_traininputs'] = []
results_dict['unpassed_all_traininputs'] = []
results_dict['passed_all_testinputs'] = []
results_dict['passed_some_testinputs'] = []
results_dict['unpassed_all_testinputs'] = []
results_dict['unknown'] = []



# hard coding the process of testing intersections among keys-values
def find_intersection_keys_values(this_map):
    keys_set = set(this_map.keys())
    values_set = set()
    for k in this_map.keys():
        values_set = values_set | this_map[k]

    return len(keys_set.intersection(values_set)) == 0, all([len(x) == 1 for x in this_map.values()])

def extract_objects(x):
    _objects = find_objects(x)

    extracted = []
    for obj in _objects:
        if obj != None:
            extracted.append([list(obj)[0].start, list(obj)[0].stop, list(obj)[1].start, list(obj)[1].stop, (list(obj)[0].stop - list(obj)[0].start), (list(obj)[1].stop - list(obj)[1].start)])
    return extracted


def build_a_prediction(reference, **kwargs):
    for k, v in kwargs.items():
        if k == 'add_to_zero':
            cur_output = np.zeros(reference) # reference is dimensions
            cur_output += v
            cur_output = cur_output.astype(int)
            return [y.tolist() for y in cur_output]
    return None

# This is a check method, important for two routines
def check_value_maps(couple_val_map):
    results = set([find_intersection_keys_values(x) for x in couple_val_map])
    if len(results) == 1:
        result = results.pop()
        if result== (True, True):
            return True, 'apply_value_maps', {} # 15, 275, 308
        elif result == (False, True):
            return True, 'apply_direct_transformation', {} # 266, 336, 388
    return False, '', {}


def check_flips(cur_train_preds, trainoutput):
    to_pass = {}
    x, y = screen_flips_rotation(cur_train_preds[0], trainoutput[0])
    train_options = [screen_flips_rotation(x, y) for x, y in zip(cur_train_preds, trainoutput)]
    if all([x != (None, None) and x == train_options[0] for x in train_options]):
        to_pass['routine'] = train_options[0]
        return True, 'apply_flips', to_pass
    else:
        return False, '', to_pass


def check_unique_output(cur_train_preds, trainoutputs, bg_in_unique_train_outputs, bg):
    outcome = False
    what_to_try = ''

    unique_train_outputs = [np.unique(x).tolist() for x in trainoutputs]
    is_unique_train_outputs = [len(x) == 1 for x in unique_train_outputs]
    to_pass = {}
    to_pass['bg_in_unique_train_outputs'] = bg_in_unique_train_outputs
    to_pass['bg'] = bg

    if all(is_unique_train_outputs):
        unique_train_outputs = [x[0] for x in unique_train_outputs]

        # test frequency based:
        frequency_graph_traininputs = get_frequency_graph([Counter(list(itertools.chain.from_iterable(x.tolist()))).most_common() for x in cur_train_preds], bg_in_unique_train_outputs, bg)

        if all([x in y for x, y in zip(unique_train_outputs, frequency_graph_traininputs)]):
            indices = [y.index(x) for x, y in zip(unique_train_outputs, frequency_graph_traininputs)]
            if all([x == indices[0] for x in indices]) and indices[0] % 2 == 0:
                to_pass['index'] = indices[0]
                return True, 'apply_unique_output_frequency', to_pass
            return  outcome, what_to_try, to_pass
        return  outcome, what_to_try, to_pass

    return  outcome, what_to_try, to_pass

def apply_unique_output_frequency(cur_preds, output_dim_preds, pass_info): # 128, 99, 338
    freq_graph = get_frequency_graph([Counter(list(itertools.chain.from_iterable(x.tolist()))).most_common() for x in cur_preds], pass_info['bg_in_unique_train_outputs'], pass_info['bg'])

    if all([len(x) > pass_info['index'] for x in freq_graph]):
        props = [x[pass_info['index']] for x in freq_graph]
    elif all([len(x) == 1 for x in freq_graph]):
        props = [x[0] for x in freq_graph]

    this_output = [build_a_prediction(k, add_to_zero = l) for k, l in zip(output_dim_preds, props)]
    return this_output


def apply_transform_map(in_, color_to_token):
    dim_0 = in_.shape[0]
    dim_1 = in_.shape[1]

    vals_set = set(color_to_token.values())
    if len(vals_set) > 0:
        this_type = type(vals_set.pop())
        if this_type == object or this_type != np.int64:
            tokenized_situation = np.empty((dim_0, dim_1), dtype = object)
        elif this_type == int:
            tokenized_situation = np.empty((dim_0, dim_1), dtype = np.int64)
    else:
        this_type = object
        tokenized_situation = np.empty((dim_0, dim_1), dtype = object)

    visited = np.empty((dim_0, dim_1), dtype = bool)

    for n in range(dim_0):
        for m in range(dim_1):
            visited[n, m]  = False

    for n in range(dim_0):
        for m in range(dim_1):
            if in_[n, m] in color_to_token.keys() and visited[n, m] == False:
                tokenized_situation[n, m] = color_to_token[in_[n, m]]
                visited[n, m] = True
            elif in_[n, m] not in color_to_token.keys() and visited[n, m] == False:
                if this_type == object:
                    tokenized_situation[n, m] = str(in_[n, m])
                else:
                    tokenized_situation[n, m] = in_[n, m]
                visited[n, m] = True

    return tokenized_situation

def apply_value_maps(in_, signal_map, pass_info):
    dim_0 = in_.shape[0]
    dim_1 = in_.shape[1]

    for n in range(dim_0):
        for m in range(dim_1):
            if in_[n,m] in signal_map.keys():
                if type(list(signal_map.values())[0]) == list:
                    in_[n,m] = signal_map[in_[n,m]][0]
                elif type(list(signal_map.values())[0]) == set:
                    in_[n,m] = signal_map[in_[n,m]].pop()
    return in_


def apply_direct_transformation(in_, color_to_token, objective, token_to_color, pass_info):
    # apply objective
    x = apply_transform_map(in_, color_to_token)
    change_dict = {}
    for obj in objective:
        if len(obj) == 1 and obj[0][2] == 'direct':
            change_dict[obj[0][0]] = obj[0][1]

    y = apply_transform_map(x, change_dict)
    z = apply_transform_map(y, token_to_color)
    return z


def get_x_y_situation(in_):
    dim_0 = in_.shape[0]
    dim_1 = in_.shape[1]
    x_situation = np.empty((dim_0, dim_1), dtype = np.int64)
    y_situation = np.empty((dim_0, dim_1), dtype = np.int64)

    for n in range(dim_0):
        for m in range(dim_1):
            x_situation[n, m] = n
            y_situation[n, m] = m

    return x_situation, y_situation

def get_frequency_graph(frequency_counter, include_bg, bg):
    frequency_graph = []
    for x in frequency_counter:
        this_list = []
        for n in x:
            if (n[0] == bg and include_bg == True) or (n[0] != bg):
                this_list.append(n[0])
                this_list.append(n[1])
        frequency_graph.append(this_list)
    return frequency_graph

def get_frequency_situation(in_):
    frequency_counter = Counter(list(itertools.chain.from_iterable(in_.tolist()))).most_common()
    frequency_counter = sorted(frequency_counter, key=lambda tup: (tup[1], tup[0]))
    frequency_dict = {}
    for x in frequency_counter:
        frequency_dict[x[0]] = x[1]

    cur_rank = -1
    new_fc = []
    for n in range(len(frequency_counter)):
        if n == 0:
            cur_rank = 0
        elif n != 0 and frequency_counter[n][1] > frequency_counter[n-1][1]:
            cur_rank += 1
        elif n != 0 and frequency_counter[n][1] == frequency_counter[n-1][1]:
            pass
        new_fc.append((frequency_counter[n][0], cur_rank))

    sorting_dict = {}
    for x in new_fc:
        sorting_dict[x[0]] = x[1]

    return apply_transform_map(in_, frequency_dict), apply_transform_map(in_, sorting_dict)


def get_flips_rotation(in_):
    # get_diagonal_mirror is more precedent than rotation which is more precedent over flipud or fliplr
    diagonal_mirror = get_diagonal_mirror(in_)
    offdiagonal_mirror = get_offdiagonal_mirror(in_)
    rotated90_1 = np.rot90(in_, 1, axes = (0, 1))
    rotated90_2 = np.rot90(in_, 2, axes = (0, 1))
    rotated90_3 = np.rot90(in_, 3, axes = (0, 1))
    flipped_ud = np.flipud(in_)
    flipped_lr = np.fliplr(in_)
    return diagonal_mirror, offdiagonal_mirror, rotated90_1, rotated90_2, rotated90_3, flipped_ud, flipped_lr

def build_prior_knowledge(in_, bg, color_to_token):
    x_situation, y_situation = get_x_y_situation(in_)
    frequency_graph, sorted_frequency_graph = get_frequency_situation(in_)
    tokenized_graph = apply_transform_map(in_, color_to_token)
    edge_situation = get_edge_situation(in_, bg)
    neighbor_situation = neighbor_situation_whole(in_)

    return np.stack((tokenized_graph, in_, x_situation, y_situation, frequency_graph, sorted_frequency_graph, edge_situation, neighbor_situation), axis = 0)

def featurize_prior_knowledge_train(in_p_k, tokenized_target_arr):
    # print(tokenized_target_arr)
    # print(in_p_k.shape)
    # print(tokenized_target_arr.shape)
    x = []
    y = []
    dim_0 = in_p_k.shape[1]
    dim_1 = in_p_k.shape[2]

    for n in range(dim_0):
        for m in range(dim_1):
            x.append(in_p_k[:, n, m].tolist())
            y.append(tokenized_target_arr[n, m])

    return np.array(x, dtype = object), np.array(y, dtype = object)

def featurize_prior_knowledge_test(in_p_k):
    x = []
    dim_0 = in_p_k.shape[1]
    dim_1 = in_p_k.shape[2]

    for n in range(dim_0):
        for m in range(dim_1):
            x.append(in_p_k[:, n, m].tolist())

    return np.array(x, dtype = object)
