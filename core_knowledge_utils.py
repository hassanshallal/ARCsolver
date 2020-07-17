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


# This is a check method, important for two routines
def check_value_maps(couple_val_map):
    results = set([find_intersection_keys_values(x) for x in couple_val_map])
    if len(results) == 1:
        result = results.pop()
        if result== (True, True):
            return True, 'apply_value_maps' # 15, 275, 308
        elif result == (False, True):
            return True, 'apply_direct_transformation' # 266, 336, 388
    return False, ''

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
    #print(tokenized_situation)
    #print(visited)
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

            #print(tokenized_situation)
            #print(visited)
    return tokenized_situation

# This is a method to apply somthing on an input array, can be extiensible with *args in order to
# apply other dicts, for example: 0 edge to nonbgpr1 for example, so, this must be an east to
# exapnd application function that takes a dict and extra arguments to change an in_ grid
def apply_value_maps(in_, signal_map):
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


def apply_direct_transformation(in_, color_to_token, objective, token_to_color):
    # Preliminary and baseline transformation
    # input_grid using color_to_token = tokenized_grid
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


def build_prior_knowledge(in_, bg, color_to_token):
    x_situation, y_situation = get_x_y_situation(in_)
    frequency_graph, sorted_frequency_graph = get_frequency_situation(in_)
    tokenized_graph = apply_transform_map(in_, color_to_token)
    edge_situation = get_edge_situation(in_, bg)
    neighbor_situation = neighbor_situation_whole(in_)

    return np.stack((tokenized_graph, in_, x_situation, y_situation, frequency_graph, sorted_frequency_graph, edge_situation, neighbor_situation), axis = 0)

def featurize_prior_knowledge_train(in_p_k, tokenized_target_arr):
    #print(tokenized_target_arr)
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

def sets_obj_on_train(cur_x_train, cur_y_train, this_objective):
    objective = deepcopy(this_objective)
    sets_dict = {}
    for obj in objective:
        if len(obj) == 2:
            sets_dict[obj[0]] = [set() for index in range(1, cur_x_train.shape[1])]
            sets_dict[obj[1]] = [set() for index in range(1, cur_x_train.shape[1])]

    for n in range(cur_x_train.shape[0]):
        first = cur_x_train[n, 0]
        second = cur_y_train[n]
        if (first, second) in sets_dict.keys():
            for m in range(1, cur_x_train.shape[1]):
                sets_dict[(first, second)][m-1].add(cur_x_train[n, m])

    # print('sets_dict: ', sets_dict)
    for obj in objective:
        if len(obj) == 2:
            is_opprtunity = [len(x.intersection(y)) == 0 for x, y in zip(sets_dict[obj[0]], sets_dict[obj[1]])]
            #print('is_opprtunity: ', is_opprtunity)
            if any(is_opprtunity):
                columns = [i+1 for i in range(len(is_opprtunity)) if is_opprtunity[i]]
                #print('columns: ', columns)
                obj.append(columns)

    for obj in objective:
        if len(obj) == 3:
            target_columns = obj[2]
            for opp in target_columns:
                obj.append((opp, sets_dict[obj[0]][opp-1], sets_dict[obj[1]][opp-1]))
    return objective
