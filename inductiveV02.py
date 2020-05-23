# should be able to get started with this very soon. may be or may be not.

from utils import *

# we will apply A-star in order to determine whether a cell is trapped with no access to edges or not
def captured_astar(grid, x, y, bg, all_):
    # print('investigatin: ', (x, y))
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
    # print(grid)
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
def get_captured(arr, n, m, bg, all_):
    # print('from get_captured n is: ', str(n), ' and m is: ', str(m))
    if captured_astar(arr, n, m, bg, all_):
        return False
    else:
        return True

def captured_situation(maze, test_points):
    test_points = test_points.tolist()
    bg = get_background(maze)
    captured_ver_hor= {}
    captured_ver_hor[1] = []
    captured_ver_hor[0] = []

    captured_diag= {}
    captured_diag[1] = []
    captured_diag[0] = []

    captured_all= {}
    captured_all[1] = []
    captured_all[0] = []

    for n in range(1, maze.shape[0]-1):
        for m in range(1, maze.shape[1]-1):
            arr = maze.copy()
            if [n, m] in test_points:
                if arr[n][m] != bg:
                    arr[n][m] = bg
                if get_captured(arr, n, m, bg, 'ver_hor'):
                    captured_ver_hor[1].append((n, m))
                else:
                    captured_ver_hor[0].append((n, m))
            arr = maze.copy()
            if [n, m] in test_points:
                if arr[n][m] != bg:
                    arr[n][m] = bg
                if get_captured(arr, n, m, bg, 'diag'):
                    captured_diag[1].append((n, m))
                else:
                    captured_diag[0].append((n, m))
            arr = maze.copy()
            if [n, m] in test_points:
                if arr[n][m] != bg:
                    arr[n][m] = bg
                if get_captured(arr, n, m, bg, 'all'):
                    captured_all[1].append((n, m))
                else:
                    captured_all[0].append((n, m))


    return [len(captured_ver_hor[1]) == len(test_points), len(captured_diag[1]) == len(test_points), len(captured_all[1]) == len(test_points)]

def captured_situation_whole(maze):
    bg = get_background(maze)
    #print(bg)
    captured_ver_hor= {}
    captured_ver_hor[1] = []
    captured_ver_hor[0] = []

    captured_diag= {}
    captured_diag[1] = []
    captured_diag[0] = []

    captured_all= {}
    captured_all[1] = []
    captured_all[0] = []
    for n in range(1, maze.shape[0]-1):
        for m in range(1, maze.shape[1]-1):
            # print('from captured situation n is: ', str(n), ' and m is: ', str(m))
            arr = maze.copy()
            if arr[n][m] != bg: # in order to operate also on non-bg cells
                arr[n][m] = bg
            elif get_captured(arr, n, m, bg, 'ver_hor'):
                captured_ver_hor[1].append([maze[n][m], (n, m)])
            else:
                captured_ver_hor[0].append([maze[n][m], (n, m)])

            arr = maze.copy()
            if arr[n][m] != bg: # in order to operate also on non-bg cells
                arr[n][m] = bg
            if get_captured(arr, n, m, bg, 'diag'):
                captured_diag[1].append([maze[n][m], (n, m)])
            else:
                captured_diag[0].append([maze[n][m], (n, m)])

            arr = maze.copy()
            if arr[n][m] != bg: # in order to operate also on non-bg cells
                arr[n][m] = bg
            if get_captured(arr, n, m, bg, 'all'):
                captured_all[1].append([maze[n][m], (n, m)])
            else:
                captured_all[0].append([maze[n][m], (n, m)])
    return captured_ver_hor, captured_diag, captured_all


# This assessment is unique to task 345, expand it and make it more generalized to larger dimensions
def assess_captured_output_holistics(this_task_output, trainoutputs, testinputs, order_, testoutputs = None):
    if all([len(x) == 1 for x in this_task_output]):
        if [x[0][0] for x in this_task_output] == [np.unique(x)[0] for x in trainoutputs]:
            candidate_solutions = [np.array([[captured_situation_whole(testinput)[order_][1][0][0]]]) for testinput in testinputs]
            if testoutputs != None:
                if candidate_solutions == testoutputs:
                    return True, candidate_solutions
                else:
                    return False, candidate_solutions
            else:
                return candidate_solutions
    return None

def get_captured_holistic(traininputs, trainoutputs, testinputs, testoutputs = None):
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
    currently_solved_assignments['ver_hor_captured'] = assess_captured_output_holistics(this_task_output_ver_hor, trainoutputs, testinputs, 0, testoutputs)
    currently_solved_assignments['diag_captured'] = assess_captured_output_holistics(this_task_output_diag, trainoutputs, testinputs, 1, testoutputs)
    currently_solved_assignments['all_captured'] = assess_captured_output_holistics(this_task_output_all, trainoutputs, testinputs, 2, testoutputs)
    currently_solved_assignments = {k:v for k, v in currently_solved_assignments.items() if v != None and v[0] == True}
    print(currently_solved_assignments)
    return len(currently_solved_assignments) > 0, currently_solved_assignments
    #[x[1] for x in this_task_output_ver_hor], [x[1] for x in this_task_output_diag], [x[1] for x in this_task_output_all]

def get_captured_target_assignment(n, x, asssignments_output, traininputs): # n is training number and x is assignment number
    observables = ['int' not in x[0] for x in list(asssignments_output[n][x].keys())]
    target_keys = [y[0] for x, y in zip(observables, list(asssignments_output[n][x].keys())) if x]
    if len(target_keys) > 1: # this is critical condition because we compare between sets in the same assignment
        results = {}
        for key, value in asssignments_output[n][x].items():
            if key[0] in target_keys:
                result = captured_situation(traininputs[n], value[0])
                if key[1] not in results.keys():
                    results[key[1]] = []
                results[key[1]].append(result)

        return results
    else:
        return None


def get_captured_target(asssignments_output, traininputs):
    results = {}
    for n in range(len(traininputs)):
        for x in range(len(asssignments_output[n])):
            this_assignment_situation = get_captured_target_assignment(n, x, asssignments_output, traininputs)
            if this_assignment_situation != None:
                for k, v in this_assignment_situation.items():
                    if (x, k) not in results.keys():
                        results[(x, k)] = []
                    results[(x, k)].append(v[0])
                    # v must be a boolean list of 3 (ver_hor, diag, all)
    return results

def work_out_captured_assignments(outputs):
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
            for n in range(len(combs)): # this will be the possible pairs of players of an assignment
                first_pair = outcomes[agents.index(combs[n][0])]
                second_pair = outcomes[agents.index(combs[n][1])]
                first_handle = combs[n][0]
                second_handle = combs[n][1]
                comparison = [x != y for x, y in zip(first_pair, second_pair)]
                if any(comparison):
                    if (first_handle, second_handle) not in result.keys():
                        result[(first_handle, second_handle)] = []
                    if comparison[0] == True:
                        if first_pair[0] == True and second_pair[0] == False:
                            result[(first_handle, second_handle)].append(('ver_hor', first_handle))
                        elif first_pair[0] == False and second_pair[0] == True:
                            result[(first_handle, second_handle)].append(('ver_hor', second_handle))
                    if comparison[1] == True:
                        if first_pair[1] == True and second_pair[1] == False:
                            result[(first_handle, second_handle)].append(('diag', first_handle))
                        elif first_pair[1] == False and second_pair[1] == True:
                            result[(first_handle, second_handle)].append(('diag', second_handle))
                    if comparison[2] == True:
                        if first_pair[2] == True and second_pair[2] == False:
                            result[(first_handle, second_handle)].append(('all', first_handle))
                        elif first_pair[2] == False and second_pair[2] == True:
                            result[(first_handle, second_handle)].append(('all', second_handle))
    return result

def assess_captured_output_target(this_task_output, testinputs, token_to_colors, testoutputs = None):
    if len(this_task_output) > 0:

        candidate_segmentations = [list(captured_situation_whole(testinput)) for testinput in testinputs]
        captured_testinput = []
        # Time to evaluate on the test inputs, this captured won't come here unless there is something
        for m in range(len(testinputs)):
            captured_cells = {}
            for k, v in this_task_output.items():
                for n in v:
                    if n not in captured_cells.keys():
                        captured_cells[n[1]] = []
                    if n[0] == 'ver_hor':
                        captured_cells[n[1]].append(candidate_segmentations[m][0][1])
                    elif n[0] == 'diag':
                        captured_cells[n[1]].append(candidate_segmentations[m][1][1])
                    elif n[0] == 'all':
                        captured_cells[n[1]].append(candidate_segmentations[m][2][1])
            if len(captured_cells) > 0:
                captured_testinput.append(captured_cells)

        if testoutputs != None and any([len(x) > 0 for x in captured_testinput]) and all(x == token_to_colors[0] for x in token_to_colors):
            # ya, this is all we need in order to be able to predict
            preds = []
            for m in range(len(captured_testinput)):
                this_input = deepcopy(testinputs[m])
                for k, v in captured_testinput[m].items():
                    if all([x == v[0] for x in v]):
                        target_coordinates = v[0] # list of tuples of (value, coordinate)
                        start = token_to_colors[0][k[0]]
                        end = token_to_colors[0][k[1]]
                        for n in target_coordinates:
                            if n[0] == start:
                                this_input[n[1][0], n[1][1]] = end

                preds.append(this_input)

            if all([np.array_equal(x, y) for x, y in zip(preds, testoutputs)]):
                return True, preds
            else:
                return False
            #return preds
        else:
            return captured_testinput
    return None
