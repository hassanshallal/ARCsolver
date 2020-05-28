
from utils import *

# captured_astar(grid, x, y, bg, all_) # recursive done
# captured_astar_util(arr, n, m, bg, all_), done
# we will apply A-star in order to determine whether a cell is trapped with no access to edges or not
# having acces to an edge means not captured, so, captured is the oppposite
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
        return False

    elif all_ == 'diag':
        if ((x > 0 and y > 0 and captured_astar(grid, x-1, y-1, bg, 'diag'))
            or (x < grid.shape[0]-1 and y < grid.shape[1]-1 and captured_astar(grid, x+1, y+1, bg, 'diag'))
            or (x > 0 and y < grid.shape[1]-1 and captured_astar(grid, x-1, y+1, bg, 'diag'))
            or (x < grid.shape[0]-1 and y > 0 and captured_astar(grid, x+1, y-1, bg, 'diag'))):
            return True
        return False

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
        return False

# For reasons I don't have time to investigate, we need to wrap up the astar algorithm in order to get it to work
def captured_astar_util(arr, n, m, bg, all_):
    if captured_astar(arr, n, m, bg, all_):
        return False
    else:
        return True

# get_edge_set(maze)
# manufacture_dict_captured_whole()
# captured_situation_target_interior(test_points, arr, n, m, bg, order_, captured)
# captured_situation_target(maze, test_points) # return boolean list of three checks (ver_hot, diag, all)
# captured_situation_whole_interior(maze, arr, n, m, bg, order_, captured)
# captured_situation_whole(maze, bg) # return three dictionaries ver_hot, diag, all, each with two keys: 0: uncaptured coordinates and 1: captured coordinates

# get a list of points on the edges
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

# Once a system goes for evaluation, it starts with evaluation on training, if things
# are fine, it progresses to evaluation on testing, this is an internal decision at the moment.

# assess_captured_holistic_training(traininputs, testinputs)
def assess_captured_holistic_training(traininputs, testinputs):
    train_task_output_ver_hor = []
    train_task_output_diag = []
    train_task_output_all = []

    for n in range(len(traininputs)):
        ver_hor_, diag_, all_ = captured_situation_whole(traininputs[n])
        train_task_output_ver_hor.append(ver_hor_)
        train_task_output_diag.append(diag_)
        train_task_output_all.append(all_)

    test_task_output_ver_hor = []
    test_task_output_diag = []
    test_task_output_all = []

    for n in range(len(testinputs)):
        ver_hor_, diag_, all_ = captured_situation_whole(testinputs[n])
        test_task_output_ver_hor.append(ver_hor_)
        test_task_output_diag.append(diag_)
        test_task_output_all.append(all_)

    return [train_task_output_ver_hor, train_task_output_diag, train_task_output_all], [test_task_output_ver_hor, test_task_output_diag, test_task_output_all]

def assess_captured_holistic_testing(testinputs, bg = None):
    test_task_output_ver_hor = []
    test_task_output_diag = []
    test_task_output_all = []

    for n in range(len(testinputs)):
        ver_hor_, diag_, all_ = captured_situation_whole(testinputs[n], bg)
        test_task_output_ver_hor.append(ver_hor_)
        test_task_output_diag.append(diag_)
        test_task_output_all.append(all_)

    return [test_task_output_ver_hor, test_task_output_diag, test_task_output_all]

def retrieve_coords_from_assignments(target_tuple, asssignments_output):
    results = []
    for n in range(len(asssignments_output)): # len num_train
        for x in range(len(asssignments_output[n])): # assignment
            this_assignment = asssignments_output[n][x]
            for k, v in this_assignment.items():
                if target_tuple == k[1]:
                    results.append(v[0])
    return results

def check_objective_against_captured(asssignments_output, objective, bg, traininputs):
    objective = deepcopy(objective)
    for n in range(len(objective)):
        this_check = objective[n]
        if len(this_check) == 1:
            objective[n] = (this_check[0][0], this_check[0][1], 'direct')
        elif len(this_check) == 2:
            first_target_coords = retrieve_coords_from_assignments(this_check[0], asssignments_output)
            second_target_coords = retrieve_coords_from_assignments(this_check[1], asssignments_output)
            if len(first_target_coords) == len(second_target_coords):
                for m in range(len(first_target_coords)):
                    first_target_result = captured_situation_target(traininputs[m], first_target_coords[m], bg)
                    second_target_result = captured_situation_target(traininputs[m], second_target_coords[m], bg)
                    comparison = [x != y  for x, y in zip(first_target_result, second_target_result)]
                    if comparison[2]:
                        if first_target_result[2] and not second_target_result[2]:
                            cap = (this_check[0])
                            uncap = (this_check[1])
                        elif not first_target_result[2] and second_target_result[2]:
                            cap = (this_check[1])
                            uncap = (this_check[0])
                        objective[n] = (this_check[0], this_check[1], 'cap_all', uncap, cap)
                        continue
                    elif comparison[0] or comparison[1]:
                        if comparison[0]:
                            target, method = 0, 'cap_ver_hor'
                        else:
                            target, method = 1, 'cap_diag'
                        if first_target_result[target] and not second_target_result[target]:
                            cap = (this_check[0])
                            uncap = (this_check[1])
                        elif not first_target_result[target] and second_target_result[target]:
                            cap = (this_check[1])
                            uncap = (this_check[0])
                        objective[n] = (this_check[0], this_check[1], method, uncap, cap)
                        continue

    return objective

def assess_captured_target_training(asssignments_output, objective, bg, traininputs, testinputs):
    new_objective = check_objective_against_captured(asssignments_output, objective, bg, traininputs)
    if new_objective != objective:
        test_rep = assess_captured_holistic_testing(testinputs, bg)
    else:
        test_rep = []
    return new_objective, test_rep
