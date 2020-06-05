
# solved: 1, 97, 119, 186, 250, 293, 337, 345
# needs other first principles: 43, 84, 101, 124, 155, 159, 203, 366

from utils import *
from communicate import *

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

class Captured:
    def __init__(self, communication, running_objective, cur_test_preds):
        self.communication = communication
        self.running_objective = running_objective
        self.cur_test_preds = cur_test_preds

    # For reasons I don't have time to investigate, we need to wrap up the astar algorithm in order to get it to work
    def captured_astar_util(self, arr, n, m, bg, all_):
        if captured_astar(arr, n, m, bg, all_):
            return False
        else:
            return True

    def get_edge_set(self, maze):
        dim_0 = maze.shape[0]
        dim_1 = maze.shape[1]
        edge_list = []
        for n in range(dim_0):
            for m in range(dim_1):
                if n == 0 or n == dim_0-1 or m == 0 or m == dim_1-1:
                    edge_list.append([maze[n][m], (n, m)])
        return edge_list

    def manufacture_dict_captured_whole(self):
        captured = {}
        captured[1] = []
        captured[0] = []
        return captured

    def captured_situation_target_interior(self, test_points, arr, n, m, bg, order_, captured):
        if [n, m] in test_points:
            if arr[n][m] != bg:
                arr[n][m] = bg
            if self.captured_astar_util(arr, n, m, bg, order_):
                captured[1].append((n, m))
            else:
                captured[0].append((n, m))
        return captured

    def captured_situation_target(self, maze, test_points, bg):
        test_points = test_points.tolist()

        captured_ver_hor = self.manufacture_dict_captured_whole()
        captured_diag = self.manufacture_dict_captured_whole()
        captured_all = self.manufacture_dict_captured_whole()

        for n in range(1, maze.shape[0]-1):
            for m in range(1, maze.shape[1]-1):
                arr = maze.copy()
                captured_ver_hor = self.captured_situation_target_interior(test_points, arr, n, m, bg, 'ver_hor', captured_ver_hor)
                arr = maze.copy()
                captured_diag = self.captured_situation_target_interior(test_points, arr, n, m, bg, 'diag', captured_diag)
                arr = maze.copy()
                captured_all = self.captured_situation_target_interior(test_points, arr, n, m, bg, 'all', captured_all)

        return [len(captured_ver_hor[1]) == len(test_points), len(captured_diag[1]) == len(test_points), len(captured_all[1]) == len(test_points)]

    def captured_situation_whole_interior(self, maze, arr, n, m, bg, order_, captured):
        if arr[n][m] != bg: # in order to operate also on non-bg cells
            arr[n][m] = bg

        if self.captured_astar_util(arr, n, m, bg, order_):
            captured[1].append([maze[n][m], (n, m)])
        else:
            captured[0].append([maze[n][m], (n, m)])
        return captured

    def captured_situation_whole(self, maze, bg = None):
        if bg == None:
            bg = get_background(maze)
        edge_list = self.get_edge_set(maze)
        captured_ver_hor = self.manufacture_dict_captured_whole()
        captured_diag = self.manufacture_dict_captured_whole()
        captured_all = self.manufacture_dict_captured_whole()

        for n in range(1, maze.shape[0]-1):
            for m in range(1, maze.shape[1]-1):
                arr = maze.copy()
                captured_ver_hor = self.captured_situation_whole_interior(maze, arr, n, m, bg, 'ver_hor', captured_ver_hor)
                captured_ver_hor[0] = captured_ver_hor[0] + edge_list
                arr = maze.copy()
                captured_diag = self.captured_situation_whole_interior(maze, arr, n, m, bg, 'diag', captured_diag)
                captured_diag[0] = captured_diag[0] + edge_list
                arr = maze.copy()
                captured_all = self.captured_situation_whole_interior(maze, arr, n, m, bg, 'all', captured_all)
                captured_all[0] = captured_all[0] + edge_list

        return captured_ver_hor, captured_diag, captured_all

    # here is where the cargo from communicate is important
    def assess_captured_holistic_training(self):
        train_task_output_ver_hor = []
        train_task_output_diag = []
        train_task_output_all = []

        for n in range(len(self.communication.traininputs)):
            ver_hor_, diag_, all_ = self.captured_situation_whole(self.communication.traininputs[n])
            train_task_output_ver_hor.append(ver_hor_)
            train_task_output_diag.append(diag_)
            train_task_output_all.append(all_)

        test_task_output_ver_hor = []
        test_task_output_diag = []
        test_task_output_all = []

        for n in range(len(self.communication.testinputs)):
            ver_hor_, diag_, all_ = self.captured_situation_whole(self.communication.testinputs[n])
            test_task_output_ver_hor.append(ver_hor_)
            test_task_output_diag.append(diag_)
            test_task_output_all.append(all_)

        return [train_task_output_ver_hor, train_task_output_diag, train_task_output_all], [test_task_output_ver_hor, test_task_output_diag, test_task_output_all]

    def assess_captured_holistic_testing(self, bg = None):
        test_task_output_ver_hor = []
        test_task_output_diag = []
        test_task_output_all = []

        for n in range(len(self.communication.testinputs)):
            ver_hor_, diag_, all_ = self.captured_situation_whole(self.communication.testinputs[n], bg)
            test_task_output_ver_hor.append(ver_hor_)
            test_task_output_diag.append(diag_)
            test_task_output_all.append(all_)

        return [test_task_output_ver_hor, test_task_output_diag, test_task_output_all]

    def check_objective_against_captured(self, bg):
        copy_objective = deepcopy(self.running_objective)
        changed = False
        if not is_list_of_list_of_list(copy_objective):
            changed = True
            copy_objective = [copy_objective]
        for x in range(len(copy_objective)):
            for n in range(len(copy_objective[x])):
                this_check = copy_objective[x][n]
                if len(this_check) == 1:
                    copy_objective[x][n] = [this_check[0][0], this_check[0][1], 'direct']
                elif len(this_check) == 2:
                    first_target_coords = retrieve_coords_from_assignments(this_check[0], self.communication.asssignments_output)
                    second_target_coords = retrieve_coords_from_assignments(this_check[1], self.communication.asssignments_output)
                    if len(first_target_coords) == len(second_target_coords):
                        for m in range(len(first_target_coords)):
                            first_target_result = self.captured_situation_target(self.communication.traininputs[m], first_target_coords[m], bg)
                            second_target_result = self.captured_situation_target(self.communication.traininputs[m], second_target_coords[m], bg)
                            comparison = [x != y  for x, y in zip(first_target_result, second_target_result)]
                            if comparison[2]:
                                if first_target_result[2] and not second_target_result[2]:
                                    cap = (this_check[0])
                                    uncap = (this_check[1])
                                elif not first_target_result[2] and second_target_result[2]:
                                    cap = (this_check[1])
                                    uncap = (this_check[0])
                                copy_objective[x][n] = [this_check[0], this_check[1], 'cap_all', uncap, cap]
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
                                copy_objective[x][n] = [this_check[0], this_check[1], method, uncap, cap]
                                continue
        if changed:
            return copy_objective[0]
        else:
            return copy_objective

    def assess_captured_target_training(self, bg):
        # this objective concept may not work for tasks of so many objectives, task 60 is an example of this
        # This type of tasks require a different framework based on symmetry and finding symmetric indices which
        # by the way must be a pretty basic first principle operation requiring none of the long objective
        # based screenings

        if len(self.running_objective) > 5:
            return self.running_objective, []

        new_objective = self.check_objective_against_captured(bg)
        if new_objective != self.running_objective:
            test_rep = self.assess_captured_holistic_testing(bg)
        else:
            test_rep = []
        return new_objective, test_rep

    def validate_cap(self, overall_cur_invest_num_unique, overall_cur_invest_unique, order_, unique_train_outputs, b):
        if is_list_equal_1(overall_cur_invest_num_unique[order_]):
            if overall_cur_invest_unique[order_] == unique_train_outputs:
                test_num_unique, test_unique = self.communication.first_principles_diff_dim_signal_class(b)
                if is_list_equal_1(test_num_unique[order_]):
                    return True, test_unique
            else:
                return False, []
        else:
            return False, []


    # generalize infer_on_mechanism
    def infer_on_mechanism(self, signal_, direct = False): # signal_ = b above
        mechanisms = set()
        decided_change_tuples = set()

        # make sure you turn off first not last, this is an example of massaging an objective
        for n in range(len(self.running_objective)-1):
            if len(self.running_objective[n]) == 3 and len(self.running_objective[n+1]) == 3 and (self.running_objective[n][1] == 'bg' or self.running_objective[n+1][1] == 'bg'):
                temp = self.running_objective[n]
                self.running_objective[n] = self.running_objective[n+1]
                self.running_objective[n+1] = temp

        # this is a heuristic for red cases
        if self.communication.objective_status == 'red' or (self.communication.objective_status == 'obd' and all([len(x) == 5 or len(x) == 3 for x in self.running_objective])): # In case of 'obd' cases, the test expectation is not read to handle unsatisfiable objectives
            self.running_objective = self.communication.set_test_expectations(self.running_objective)

        for n in self.running_objective:
            if len(n) > 2:
                if n[2] == 'direct' and direct: # Major hyperparameter candidate
                    mechanisms.add(n[2])
                    self.cur_test_preds = self.communication.apply_direct_transformation((n[0], n[1]), self.cur_test_preds)
                    # in cur_test_preds, replace n[0] with n[1]
                elif n[2] in ['cap_all', 'cap_ver_hor', 'cap_diag']:
                    mechanisms.add(n[2])
                    if n[2] == 'cap_all':
                        this_signal = signal_[2]
                    elif n[2] == 'cap_ver_hor':
                        this_signal = signal_[0]
                    elif n[2] == 'cap_diag':
                        this_signal = signal_[1]

                    if n[3][0] == n[3][1]:
                        coords = [x[1] for x in this_signal]
                        if n[4] not in decided_change_tuples:
                            decided_change_tuples.add(n[4])
                            self.cur_test_preds = self.communication.objectively_build_a_prediction(n[4], coords, self.cur_test_preds)

                    elif n[4][0] == n[4][1]:
                        coords = [x[0] for x in this_signal]
                        if n[3] not in decided_change_tuples:
                            decided_change_tuples.add(n[3])
                            self.cur_test_preds = self.communication.objectively_build_a_prediction(n[3], coords, self.cur_test_preds)
                    else:
                        if n[3] not in decided_change_tuples and n[4] not in decided_change_tuples: # we can't repeat assignments, very imp precondition
                            uncap_coords = [x[0] for x in this_signal]
                            if n[3] not in decided_change_tuples:
                                decided_change_tuples.add(n[3])
                                self.cur_test_preds = self.communication.objectively_build_a_prediction(n[3], uncap_coords, self.cur_test_preds)
                            cap_coords = [x[1] for x in this_signal]
                            if n[4] not in decided_change_tuples:
                                decided_change_tuples.add(n[4])
                                self.cur_test_preds = self.communication.objectively_build_a_prediction(n[4], cap_coords, self.cur_test_preds)

        return sorted(list(mechanisms))

    # The following is the screening protocola which needs alot of cleaning
    # ya the screen dimensions is a scary method, needs a lot of work now gets: 222, 268, 288, 306
    # screen dimensions
    def screen_captured(self): # This is abstract
        if self.communication.objective_status == 'None':
            unique_train_outputs = [np.unique(x).tolist() for x in self.communication.trainoutputs]
            is_unique_train_outputs = [len(x) == 1 for x in unique_train_outputs]
            a, b = self.assess_captured_holistic_training()
            if not self.communication.first_principles_diff_dim_checks(a) or not all(is_unique_train_outputs):
                return 'unsolved', [], [], None
            else:
                if all(is_unique_train_outputs):
                    unique_train_outputs = [x[0] for x in unique_train_outputs]
                    overall_cur_invest_num_unique, overall_cur_invest_unique = self.communication.first_principles_diff_dim_signal_class(a)
                    m, n = self.validate_cap(overall_cur_invest_num_unique, overall_cur_invest_unique, 2, unique_train_outputs, b)
                    if m:
                        return 'solved', ['cap_all', 'one_unique_captured'], [[[l] for l in n[2]]], None
                    m, n = self.validate_cap(overall_cur_invest_num_unique, overall_cur_invest_unique, 0, unique_train_outputs, b)
                    if m:
                        return 'solved', ['cap_ver_hor', 'one_unique_captured'], [[[l] for l in n[0]]], None
                    m, n = self.validate_cap(overall_cur_invest_num_unique, overall_cur_invest_unique, 1, unique_train_outputs, b)
                    if m:
                        return 'solved', ['cap_diag', 'one_unique_captured'], [[l] for l in n[1]], None
                return 'unsolved', [], [], None

        else:
            a, b =  self.assess_captured_target_training(self.communication.bg)
            # we need to create outputs out of the modified objective and the b (testinputs as sets)
            is_solved = 'unsolved'
            mechanisms = []
            if len(b) > 0:
                self.running_objective = a
                objective_satisfiability = [len(x) == 3 or len(x) == 5 for x in a]

                if all(objective_satisfiability):
                    mechanisms = self.infer_on_mechanism(b, True)
                elif any(objective_satisfiability):
                    mechanisms = self.infer_on_mechanism(b)

                if all([np.array_equal(x, y) for x, y in zip(self.cur_test_preds, self.communication.testoutputs)]):
                    is_solved = 'solved'
                elif any(objective_satisfiability):
                    is_solved = 'partially solved'
                else:
                    is_solved = 'unsolved'

            return is_solved, mechanisms, self.cur_test_preds, self.running_objective


# captured_astar(grid, x, y, bg, all_) # recursive done
# captured_astar_util(arr, n, m, bg, all_), done
# we will apply A-star in order to determine whether a cell is trapped with no access to edges or not
# having acces to an edge means not captured, so, captured is the oppposite
# get_edge_set(maze)
# manufacture_dict_captured_whole()
# captured_situation_target_interior(test_points, arr, n, m, bg, order_, captured)
# captured_situation_target(maze, test_points) # return boolean list of three checks (ver_hot, diag, all)
# captured_situation_whole_interior(maze, arr, n, m, bg, order_, captured)
# captured_situation_whole(maze, bg) # return three dictionaries ver_hot, diag, all, each with two keys: 0: uncaptured coordinates and 1: captured coordinates
# get a list of points on the edges
# Once a system goes for evaluation, it starts with evaluation on training, if things
# are fine, it progresses to evaluation on testing, this is an internal decision at the moment.
# assess_captured_holistic_training(traininputs, testinputs)
