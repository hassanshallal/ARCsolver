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

# 'check_scenarios', self.pr_kn_apply_scenarios != None
def check_scenarios(pr_kn_apply_scenarios):
    if pr_kn_apply_scenarios != None:
        return True, 'apply_scenarios', {}
    else:
        return False, '', {}
    
    
    
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

def apply_scenarios(pr_knowledge_input, pr_kn_apply_scenarios, token_to_color, pass_info):
    # get the tokenized starter
    y = deepcopy(pr_knowledge_input[0])
    
    # organize scenarios based on leads
    leads = [x[0] for x in pr_kn_apply_scenarios]
    leads_scenarios = {}
    for k in leads:
        leads_scenarios[k] = [x for x in pr_kn_apply_scenarios if x[0] == k]
        
    for m in range(y.shape[0]):
        for n in range(y.shape[1]):
            if y[m][n] in leads_scenarios.keys():
                playgrounds = leads_scenarios[y[m][n]]
                for playground in playgrounds:
                    if playground[1] == 'direct':
                        y[m][n] = playground[2]
                    elif len(playground) == 4:
                        if 'not' not in playground[2] and pr_knowledge_input[playground[1]][m][n] in playground[2]:
                            y[m][n] = playground[3]
                        elif 'not' in playground[2] and pr_knowledge_input[playground[1]][m][n] not in playground[2]:
                            y[m][n] = playground[3]
                            
                    elif len(playground) > 4:
                        current_options = [x for x in playground if type(x) != str]
                        kn_indices = [i for i in range(len(current_options)) if i % 2 == 0]
                        vals_indices = [i for i in range(len(current_options)) if i % 2 != 0]
                        test_all = np.all([pr_knowledge_input[current_options[x]][m][n] in current_options[y] for x, y in zip(kn_indices, vals_indices)])
                        if test_all:
                            y[m][n] = playground[len(playground) - 1]
                            
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

    return np.stack((tokenized_graph, in_, x_situation, y_situation, sorted_frequency_graph, edge_situation, neighbor_situation), axis = 0) # remove frequency_graph, 

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


def merge_sub_objectives(sub_test, sub_lead):
#     print('sub_test:', sub_test)
#     print('sub_lead:', sub_lead)
    if len(sub_test[0]) == 2 and len(sub_test) > 2 and len(sub_lead) > 2 and sub_test[0:2] == sub_lead[0:2]:
        intersect = sorted(list(set(sub_test[2]).intersection(set(sub_lead[2]))))
        #print('intersect:', intersect)
        if len(intersect) > 0:
            merger = [sub_lead[0], sub_lead[1], intersect]
            #print('merger 0:', merger)
            to_remove = []
            for n in intersect:
                in_sub_test = [x for x in sub_test if x[0] == n and type(x) == tuple]
                in_sub_lead = [x for x in sub_lead if x[0] == n and type(x) == tuple]
                first = in_sub_test[0][1].union(in_sub_lead[0][1])
                second = in_sub_test[0][2].union(in_sub_lead[0][2])
                #print('first: ', first)
                #print('second: ', second)
                if len(first.intersection(second)) == 0:
                    new_sit = (n, first, second)
                    merger.append(new_sit)
                else:
                    to_remove.append(n)
            merger[2] = [x for x in merger[2] if x not in to_remove]
            
            if len(merger) > 3:
                #print('got all')
                return merger
            else:
                #print('truncated objectve')
                merger = [sub_lead[0], sub_lead[1]]
                return merger
        else:
            merger = [sub_lead[0], sub_lead[1]]
            return merger
    elif len(sub_test[0]) == 2 and len(sub_test[0]) == 2 and len(sub_lead) > 2 and sub_test[0:2] == sub_lead[0:2]:
        merger = [sub_test[0], sub_test[1]]
        return merger
    
    # From here, always consider cases 99, 128, 338, mainly `128, we extend and hope this fix 345
    elif len(sub_test[0]) == 3 and len(sub_lead[0]) >= 3 and len(sub_test) > 1 and len(sub_lead) > 1 and sub_test[0][0] == sub_lead[0][0] and sub_test[0][2] == list(sub_lead[0])[len(sub_lead[0])-1]:
        if sub_test[0][1] not in list(sub_lead[0])[1:len(sub_lead[0])]:
            modified_lead_tuple = tuple([sub_test[0][0], sub_test[0][1]] + list(sub_lead[0])[1:len(sub_lead[0])])
        else:
            modified_lead_tuple = sub_lead[0]
            
        intersect = sorted(list(set(sub_test[1]).intersection(set(sub_lead[1]))))
        if len(intersect) > 0:
            merger = [modified_lead_tuple, intersect]
            for n in intersect:
                in_sub_test = [x for x in sub_test if x[0] == n and type(x) == tuple]
                in_sub_lead = [x for x in sub_lead if x[0] == n and type(x) == tuple]
                new_sit = (n, in_sub_test[0][1].union(in_sub_lead[0][1]))
                merger.append(new_sit)
            return merger
        else:
            merger = [modified_lead_tuple]
            return merger  
    else:
        return sub_lead
        
def is_complete_sub_objective(sub_objective):
            return len(sub_objective) != 2
    
    
def is_all_nonbg_ass_leads(this_obj):
    for x in this_obj:
        if type(x) == tuple and x[0] in ['bg', 'nil']:
            return False, x[0]
    return True, 'nonbg'

def is_match(extra_sub, global_obj_sub):
    extra_sub_all_nonbg, extra_sub_sit = is_all_nonbg_ass_leads(extra_sub)
    global_obj_sub_all_nonbg, global_obj_sub_sit = is_all_nonbg_ass_leads(global_obj_sub)
    if len(extra_sub[0]) == len(extra_sub[1]) and extra_sub_sit == global_obj_sub_sit:
        return True
    return False

def normalize_tuple(this_tuple):
    new_tuple = []
    for n in this_tuple:
        if 'pr' in n:
            new_tuple.append(n)
        elif 'pr' not in n and any(map(str.isdigit, n)):
            new_tuple.append(''.join([i for i in n if not i.isdigit()]))
        else:
            new_tuple.append(n)
    return tuple(new_tuple)

def normalize_sub_obj(sub_obj):
    if len(sub_obj) >= 2 and len(sub_obj[0]) == 2:
        return [normalize_tuple(sub_obj[0]), normalize_tuple(sub_obj[1])]
    elif len(sub_obj) >= 1 and len(sub_obj[0]) == 3:
        return [normalize_tuple(sub_obj[0])]
    
def enforce_global_objective(test_objective, global_objective):
    # Find whether any test sub_obj is not exixtent oin the global objective
    for n in range(len(test_objective)):
        this_test_objective = test_objective[n]
        included_in_test = []
        for x in range(len(this_test_objective)):
            for y in range(len(global_objective)):
                test_len = len(this_test_objective[x])
                if len(global_objective[y]) >= test_len and global_objective[y][0:test_len] == this_test_objective[x]:
                    included_in_test.append(x)       
                    
        not_included = [l for l in range(len(this_test_objective)) if l not in included_in_test]
        #print('not_included:',not_included)
        # adjust global_objective to allow not included test sub objectives
        if len(not_included) == 0:
            return global_objective
        else:
            #print(not_included)
            for to_massage in not_included:
                this_extra = this_test_objective[to_massage] # [('nonbg4', 'nonbg4'), ('nonbg4', 'bg')]
                #print('this_extra:', this_extra)
                normalized_extra = normalize_sub_obj(this_extra)
                #print('normalized_extra:', normalized_extra)
                normalized_global_objective = [normalize_sub_obj(x) for x in global_objective]
                #print('normalized_global_objective:', normalized_global_objective)
                matches = [x == normalized_extra for x in normalized_global_objective]
                #print('matches:', matches)
                if any(matches):
                    #print('any(matches):', any(matches))
                    any_match = matches.index(True)
                    if len(this_extra) == 2:
                        #print('not added:', global_objective[any_match][2:len(global_objective[any_match])])
                        this_extra = this_extra + global_objective[any_match][2:len(global_objective[any_match])]
                        global_objective.append(this_extra)
                    elif len(this_extra) == 1:
                        #print('not added:', global_objective[any_match][1:len(global_objective[any_match])])
                        this_extra = this_extra + global_objective[any_match][1:len(global_objective[any_match])]
                        global_objective.append(this_extra)

            return global_objective   
    
def get_final_scenarios(global_objective):
    scenarios = []
    for n in global_objective:
        if len(n[0]) > 2 and n[0][1] != 'nil' and n[0][len(n[0])-1] == 'direct':
            if len(n) == 1:
                scenarios.append([n[0][0], n[0][2], n[0][1]])
            elif len(n) > 1:
                for candidate in n[1]:
                    candidate_soln = [x for x in n if type(x) == tuple and x[0] == candidate][0]
                    new_list = [n[0][0], candidate_soln[0], list(candidate_soln[1]), 'x'] + list(n[0][1:len(n[0])-1])
                    if new_list not in scenarios:
                        scenarios.append(new_list)
           
        elif len(n[0]) == 2:
            for candidate in n[2]:
                candidate_soln = [x for x in n if type(x) == tuple and x[0] == candidate][0]
                new_list_1 = [n[0][0], candidate_soln[0], list(candidate_soln[1]), n[0][1]]
                new_list_2 = [n[1][0], candidate_soln[0], list(candidate_soln[2]), n[1][1]]
                if new_list_1 not in scenarios:
                    scenarios.append(new_list_1)
                if new_list_2 not in scenarios:    
                    scenarios.append(new_list_2)
    #print('scenarios before removing redundant: ', scenarios)  
    
    # remove redundant
    scenarios = [y for y in scenarios if ('x' in y or 'direct' in y) or y[0] != y[len(y)-1]]
    #print('scenarios after removing redundant: ', scenarios)  
    # remove non-deterministic
    deterministic_scenarios = []
    
    for l in range(len(scenarios)):
        cur_scenario = scenarios[l]
        cur_scenario_in_scenarios = [x for x in scenarios if x[0] == cur_scenario[0] and x[len(x)-1] == cur_scenario[len(cur_scenario)-1]]
        cur_scenario_in_deterministic = [x for x in deterministic_scenarios if x[0] == cur_scenario[0] and x[len(x)-1] == cur_scenario[len(cur_scenario)-1]]
        
        if len(cur_scenario_in_scenarios) == 1 or (len(cur_scenario_in_scenarios) > 1 and len(cur_scenario_in_deterministic) == 0):
            deterministic_scenarios.append(cur_scenario)
        if (len(cur_scenario_in_scenarios) > 1 and len(cur_scenario_in_deterministic) == 1):
            if len(cur_scenario[2]) < len(cur_scenario_in_deterministic[0][2]):
                deterministic_scenarios.remove(cur_scenario_in_deterministic[0])
                cur_scenario = [y for y in cur_scenario if y != 'x']
                deterministic_scenarios.append(cur_scenario)
            elif len(cur_scenario[2]) == 1 and len(cur_scenario_in_deterministic[0][2]) == 1:
                deterministic_scenarios.remove(cur_scenario_in_deterministic[0])
                new_scenario = cur_scenario_in_deterministic[0] 
                to_add = cur_scenario[1:len(new_scenario)]
                new_scenario = new_scenario[0:len(new_scenario) - 1] + cur_scenario[1:len(new_scenario)]
                new_scenario = [y for y in new_scenario if y != 'x']
                deterministic_scenarios.append(new_scenario)
                
    #print('deterministic_scenarios before control: ', deterministic_scenarios)  
    if len(deterministic_scenarios) == 2 and deterministic_scenarios[0][0] == deterministic_scenarios[1][0] and 'x' not in deterministic_scenarios[0] and 'x' not in deterministic_scenarios[1]:
        m = len(deterministic_scenarios[0][2]) == 1
        n = len(deterministic_scenarios[1][2]) == 1
        if (m and not n) or (not m and n):
            if m:
                deterministic_scenarios[1][2] = ['not'] + deterministic_scenarios[0][2]
            elif n:
                deterministic_scenarios[0][2] = ['not'] + deterministic_scenarios[1][2]
                
    return deterministic_scenarios



