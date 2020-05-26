# should be able to get started with this very soon. may be or may be not.

from utils import *


# we will apply A-star in order to determine whether a cell is trapped with no access to edges or not
def captured_astar(grid, x, y, bg, all_):
    if (x == 0 or y == 0 or x == grid.shape[0]-1 or y == grid.shape[1]-1) and grid[x][y] == bg:
        # print('found edge at %d,%d' % (x, y))
        return True
    elif grid[x][y] == -1:
        # print('visited at %d,%d' % (x, y))
        return False
    elif grid[x][y] != bg:
        # print('wall at %d,%d' % (x, y))
        return False


    # mark as visited
    grid[x][y] = -1
    # explore neighbors clockwise starting by the one on the right
    if all_ == 'ver_hor':
        if ((x < grid.shape[0]-1 and captured_astar(grid, x+1, y, bg, 'ver_hor'))
            or (y > 0 and captured_astar(grid, x, y-1, bg, 'ver_hor'))
            or (x > 0 and captured_astar(grid, x-1, y, bg, 'ver_hor'))
            or (y < grid.shape[1]-1 and captured_astar(grid, x, y+1, bg, 'ver_hor'))):
            return True
        return (False)

    elif all_ == 'diag':
        if ((x > 0 and y > 0 and captured_astar(grid, x-1, y-1, bg, 'diag'))
            or (x < grid.shape[0]-1 and y < grid.shape[1]-1 and captured_astar(grid, x+1, y+1, bg, 'diag'))
            or (x > 0 and y < grid.shape[1]-1 and captured_astar(grid, x-1, y+1, bg, 'diag'))
            or (x < grid.shape[0]-1 and y > 0 and captured_astar(grid, x+1, y-1, bg, 'diag'))):
            return True
        return (False)

    elif all_ == 'all':
        if ((x < grid.shape[0]-1 and captured_astar(grid, x+1, y, bg, 'all'))
            or (y > 0 and captured_astar(grid, x, y-1, bg, 'all'))
            or (x > 0 and captured_astar(grid, x-1, y, bg, 'all'))
            or (y < grid.shape[1]-1 and captured_astar(grid, x, y+1, bg, 'all'))

            or (x > 0 and y > 0 and captured_astar(grid, x-1, y-1, bg, 'all'))
            or (x < grid.shape[0]-1 and y < grid.shape[1]-1 and captured_astar(grid, x+1, y+1, bg, 'all'))
            or (x > 0 and y < grid.shape[1]-1 and captured_astar(grid, x-1, y+1, bg, 'all'))
            or (x < grid.shape[0]-1 and y > 0 and captured_astar(grid, x+1, y-1, bg, 'all'))):
            return True
        return (False)

# For reasons I don't have time to investigate, we need to wrap up the astar algorithm in order to get it to work
def captured_astar_util(arr, n, m, bg, all_):
    if captured_astar(arr, n, m, bg, all_):
        return False
    else:
        return True

#
def get_edge_set(maze):
    dim_0 = maze.shape[0]
    dim_1 = maze.shape[1]
    edge_list = []
    for n in range(dim_0):
        for m in range(dim_1):
            if n == 0 or n == dim_0-1 or m == 0 or m == dim_1-1:
                edge_list.append([maze[n][m], (n, m)])

    return edge_list


def manufacture_dict_captured_whole():
    captured = {}
    captured[1] = []
    captured[0] = []
    return captured

def captured_situation_target_interior(test_points, arr, n, m, bg, order_, captured):
    if [n, m] in test_points:
        if arr[n][m] != bg:
            arr[n][m] = bg
        if captured_astar_util(arr, n, m, bg, order_):
            captured[1].append((n, m))
        else:
            captured[0].append((n, m))
    return captured

def captured_situation_target(maze, test_points, bg):
    test_points = test_points.tolist()

    captured_ver_hor = manufacture_dict_captured_whole()
    captured_diag = manufacture_dict_captured_whole()
    captured_all = manufacture_dict_captured_whole()

    for n in range(1, maze.shape[0]-1):
        for m in range(1, maze.shape[1]-1):
            arr = maze.copy()
            captured_ver_hor = captured_situation_target_interior(test_points, arr, n, m, bg, 'ver_hor', captured_ver_hor)
            arr = maze.copy()
            captured_diag = captured_situation_target_interior(test_points, arr, n, m, bg, 'diag', captured_diag)
            arr = maze.copy()
            captured_all = captured_situation_target_interior(test_points, arr, n, m, bg, 'all', captured_all)

    return [len(captured_ver_hor[1]) == len(test_points), len(captured_diag[1]) == len(test_points), len(captured_all[1]) == len(test_points)]


def captured_situation_whole_interior(maze, arr, n, m, bg, order_, captured):
    if arr[n][m] != bg: # in order to operate also on non-bg cells
        arr[n][m] = bg

    if captured_astar_util(arr, n, m, bg, order_):
        captured[1].append([maze[n][m], (n, m)])
    else:
        captured[0].append([maze[n][m], (n, m)])
    return captured

def captured_situation_whole(maze, bg = None):
    if bg == None:
        bg = get_background(maze)
    edge_list = get_edge_set(maze)
    captured_ver_hor = manufacture_dict_captured_whole()
    captured_diag = manufacture_dict_captured_whole()
    captured_all = manufacture_dict_captured_whole()

    for n in range(1, maze.shape[0]-1):
        for m in range(1, maze.shape[1]-1):
            arr = maze.copy()
            captured_ver_hor = captured_situation_whole_interior(maze, arr, n, m, bg, 'ver_hor', captured_ver_hor)
            captured_ver_hor[0] = captured_ver_hor[0] + edge_list
            arr = maze.copy()
            captured_diag = captured_situation_whole_interior(maze, arr, n, m, bg, 'diag', captured_diag)
            captured_diag[0] = captured_diag[0] + edge_list
            arr = maze.copy()
            captured_all = captured_situation_whole_interior(maze, arr, n, m, bg, 'all', captured_all)
            captured_all[0] = captured_all[0] + edge_list

    return captured_ver_hor, captured_diag, captured_all


# This assessment is unique to task 345, expand it and make it more generalized to larger dimensions

def assess_captured_holistic_testing(this_task_output, testinputs, order_, testoutputs = None):
    candidate_solutions = [np.array([[captured_situation_whole(testinput)[order_][1]]]) for testinput in testinputs]
    # again, this is a special condition for 345 and shouldn't be here
    if all([len(x) == 1 for x in this_task_output]):
        candidate_solutions = [x[0][0] for x in candidate_solutions]

    if testoutputs != None and [candidate_solutions[0][0][0] == testoutputs[0][0][0]]:
        return True, candidate_solutions[0][0][0]
    elif testoutputs != None and [candidate_solutions[0][0][0] != testoutputs[0][0][0]]:
        return False, candidate_solutions[0][0][0]
    elif testoutputs == None:
        return None, candidate_solutions[0][0][0]


def assess_captured_holistic_training(traininputs, trainoutputs, testinputs, testoutputs = None):
    this_task_output_ver_hor = []
    this_task_output_diag = []
    this_task_output_all = []

    for n in range(len(trainoutputs)):
        ver_hor_, diag_, all_ = captured_situation_whole(traininputs[n])
        this_task_output_ver_hor.append(ver_hor_[1])
        this_task_output_diag.append(diag_[1])
        this_task_output_all.append(all_[1])
    # gather results
    currently_solved_assignments = {}
    if all([len(x) > 0 for x in this_task_output_ver_hor]) and [x[0][0] for x in this_task_output_ver_hor] == [np.unique(x)[0] for x in trainoutputs]:
        currently_solved_assignments['ver_hor_captured'] = assess_captured_holistic_testing(this_task_output_ver_hor, testinputs, 0, testoutputs)
    if all([len(x) > 0 for x in this_task_output_diag]) and [x[0][0] for x in this_task_output_diag] == [np.unique(x)[0] for x in trainoutputs]:
        currently_solved_assignments['diag_captured'] = assess_captured_holistic_testing(this_task_output_diag, testinputs, 1, testoutputs)
    if all([len(x) > 0 for x in this_task_output_all]) and [x[0][0] for x in this_task_output_all] == [np.unique(x)[0] for x in trainoutputs]:
        currently_solved_assignments['all_captured'] = assess_captured_holistic_testing(this_task_output_all, testinputs, 2, testoutputs)

    currently_solved_assignments = {k:v for k, v in currently_solved_assignments.items() if v[0] == True}
    if len(currently_solved_assignments) > 0:
        return len(currently_solved_assignments) > 0, currently_solved_assignments
    else:
        return None, None


def assess_captured_target_training_interior(n, x, asssignments_output, traininputs, bg): # n is training number and x is assignment number
    observables = ['int' not in x[0] for x in list(asssignments_output[n][x].keys())]
    target_keys = [y[0] for x, y in zip(observables, list(asssignments_output[n][x].keys())) if x]
    if len(target_keys) > 1: # this is critical condition because we compare between sets in the same assignment
        results = {}
        for key, value in asssignments_output[n][x].items():
            if key[0] in target_keys:
                result = captured_situation_target(traininputs[n], value[0], bg)
                if key[1] not in results.keys():
                    results[key[1]] = []
                results[key[1]].append(result)

        return results
    else:
        return None

def assess_captured_target_training(asssignments_output, bg, traininputs, token_to_colors, testinputs, testoutputs):
    #print('input asssignments_output: ', asssignments_output)
    results = {}
    for n in range(len(traininputs)):
        for x in range(len(asssignments_output[n])):
            this_assignment_situation = assess_captured_target_training_interior(n, x, asssignments_output, traininputs, bg)
            if this_assignment_situation != None:
                for k, v in this_assignment_situation.items():
                    if (x, k) not in results.keys():
                        results[(x, k)] = []
                    results[(x, k)].append(v[0])
                    # v must be a boolean list of 3 (ver_hor, diag, all)

    for k, v in results.items():
            if all([x == v[0] for x in v]):
                results[k] = v[0]
            else:
                results[k] = None # something none consistent among training couples
    # print('results: ', results)
    determined = determine_captured_assignments(results)
    # print('determined: ', determined)
    assessed, preds = assess_captured_target_testing(determined, bg, token_to_colors, testinputs, testoutputs)
    return assessed, preds


def determine_captured_assignments(outputs):
    assignment_based = {}
    for k, v in outputs.items():
        if k[0] not in assignment_based.keys():
            assignment_based[k[0]] = []
        assignment_based[k[0]].append((k[1], v))

    result = {}
    for k, v in assignment_based.items():
        agents = [x[0] for x in v]
        outcomes = [x[1] for x in v]
        if len(agents) >= 2:
            combs = sorted(list(combinations(sorted(agents), 2)))
            # print('agents: ', agents)
            # print('combs: ', combs)
            for n in range(len(combs)): # this will be the possible pairs of players of an assignment
                first_pair = outcomes[agents.index(combs[n][0])]
                second_pair = outcomes[agents.index(combs[n][1])]
                first_handle = combs[n][0]
                second_handle = combs[n][1]
                if first_pair != None and second_pair != None and len(first_pair) == len(second_pair):
                    comparison = [x != y for x, y in zip(first_pair, second_pair)]
                    if any(comparison):
                        if (first_handle, second_handle) not in result.keys():
                            result[(first_handle, second_handle)] = []
                        if comparison[0] == True:
                            if first_pair[0] == True and second_pair[0] == False:
                                result[(first_handle, second_handle)].append(('ver_hor', first_handle, second_handle))
                            elif first_pair[0] == False and second_pair[0] == True:
                                result[(first_handle, second_handle)].append(('ver_hor', second_handle, first_handle))
                        if comparison[1] == True:
                            if first_pair[1] == True and second_pair[1] == False:
                                result[(first_handle, second_handle)].append(('diag', first_handle, second_handle))
                            elif first_pair[1] == False and second_pair[1] == True:
                                result[(first_handle, second_handle)].append(('diag', second_handle, first_handle))
                        if comparison[2] == True:
                            if first_pair[2] == True and second_pair[2] == False:
                                result[(first_handle, second_handle)].append(('all', first_handle, second_handle))
                            elif first_pair[2] == False and second_pair[2] == True:
                                result[(first_handle, second_handle)].append(('all', second_handle, first_handle))
    return result


def assess_captured_target_testing(this_task_output, bg, token_to_colors, testinputs, testoutputs = None):
    captured_cells = {}
    uncaptured_cells = {}
    if len(this_task_output) > 0:
        candidate_segmentations = [list(captured_situation_whole(testinput, bg)) for testinput in testinputs]
        for m in range(len(testinputs)):
            this_candidate_segmentations = candidate_segmentations[m]
            for k, v in this_task_output.items():
                for n in v:
                    if (m, n[1]) not in captured_cells.keys():
                        captured_cells[(m, n[1])] = []
                    if (m, n[2]) not in uncaptured_cells.keys():
                        uncaptured_cells[(m, n[2])] = []

                    if n[0] == 'ver_hor':
                        captured_cells[(m, n[1])].append(this_candidate_segmentations[0][1])
                        uncaptured_cells[(m, n[2])].append(this_candidate_segmentations[0][0])
                    elif n[0] == 'diag':
                        captured_cells[(m, n[1])].append(this_candidate_segmentations[1][1])
                        uncaptured_cells[(m, n[2])].append(this_candidate_segmentations[1][0])
                    if n[0] == 'all':
                        captured_cells[(m, n[1])].append(this_candidate_segmentations[2][1])
                        uncaptured_cells[(m, n[2])].append(this_candidate_segmentations[2][0])

    preds = []

    if testoutputs != None and all(x == token_to_colors[0] for x in token_to_colors):
        for m in range(len(testinputs)):
            this_input = deepcopy(testinputs[m])

            for k, v in captured_cells.items():
                if k[0] == m:
                    target_coordinates = v[0] # list of tuples of (value, coordinate)
                    start = token_to_colors[0][k[1][0]]
                    end = token_to_colors[0][k[1][1]]
                    for n in target_coordinates:
                        if n[0] == start:
                            this_input[n[1][0], n[1][1]] = end

            for k, v in uncaptured_cells.items():
                if k[0] == m:
                    target_coordinates = v[0] # list of tuples of (value, coordinate)
                    start = token_to_colors[0][k[1][0]]
                    end = token_to_colors[0][k[1][1]]
                    for n in target_coordinates:
                        if n[0] == start:
                            this_input[n[1][0], n[1][1]] = end
            if np.array_equal(this_input, testinputs[m]) == False:
                preds.append(this_input)

    if all([np.array_equal(x, y) for x, y in zip(preds, testoutputs)]):
        return True, preds
    else:
        return False, preds




# captured_astar(grid, x, y, bg, all_) # recursive
# captured_astar_util(arr, n, m, bg, all_)

# manufacture_dict_captured_whole()

# captured_situation_target_interior(test_points, arr, n, m, bg, order_, captured)
# captured_situation_target(maze, test_points) # return boolean list of three checks (ver_hot, diag, all)
# captured_situation_whole_interior(maze, arr, n, m, bg, order_, captured)
# captured_situation_whole(maze, bg) # return three dictionaries ver_hot, diag, all, each with two keys: 0: uncaptured coordinates and 1: captured coordinates

# Once a system goes for evaluation, it starts with evaluation on training, if things
# are fine, it progresses to evaluation on testing, this is an internal decision at the moment.

# assess_captured_holistic_training(traininputs, trainoutputs, testinputs, testoutputs = None)
# assess_captured_holistic_testing(this_task_output, trainoutputs, testinputs, order_, testoutputs = None)

## Fixed so far
#
# assess_captured_target_training_interior(n, x, asssignments_output, traininputs)
# assess_captured_target_training(asssignments_output, traininputs)
# determine_captured_assignments(outputs)
# assess_captured_target_testing(this_task_output, bg, testinputs, token_to_colors, testoutputs = None)
