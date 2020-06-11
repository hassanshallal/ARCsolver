
from utils import *
from communicate import *
from cages import *


nbh = lambda arr, i, j: {
    (ip, jp) : arr[i+ip, j+jp]
        for ip, jp in product([1, -1, 0], repeat=2)
            if 0 <= i+ip < arr.shape[0] and 0 <= j+jp < arr.shape[1]
}

nbh_conditional = lambda arr, i, j: {
    (i+ip, j+jp) : arr[i+ip, j+jp]
            for ip, jp in product([1, -1, 0], repeat=2)
                if (0 <= i < arr.shape[0]) and (0 <= j < arr.shape[1]) and (0 <= i+ip < arr.shape[0]) and (0 <= j+jp < arr.shape[1])
}

def get_neighbour_shifts(l):
    shifts = []
    for ip, jp in itertools.product([*l], repeat=2):
        shifts.append((ip, jp))
    shifts = [x for x in shifts if x != (0, 0)]
    return shifts

def get_nb_location_space():
    nb_shifts = get_neighbour_shifts([1, -1, 0])

    all_combs = set()
    all_combs.add(tuple(nb_shifts))

    for n in nb_shifts:
        all_combs.add((n, ))

    for n in range(2, 8):
        the_list = list(itertools.combinations(get_neighbour_shifts([1, -1, 0]), n))
        for x in the_list:
            all_combs.add(x)

    l1 = list(all_combs)
    l1 = [[]] + l1
    for x in range(len(l1)):
        if len(l1[x]) > 0:
            l1[x] = sorted([list(y) for y in l1[x]])
    nb_space = sorted(l1, key = len)
    assert len(nb_space) == 256
    return nb_space

class Neighbors:
    def __init__(self, communication, running_objective, cur_test_preds):
        self.communication = communication

        # print(self.communication.dimension_status, self.communication.output_dim_preds)
        self.running_objective = running_objective
        self.cur_test_preds = cur_test_preds

        # control the space in regards of number, locarion, and type of neighbours
        self.nb_num_space = [x for x in range(9)]
        self.nb_location_space = get_nb_location_space() # don't worry about this now, just focus on [255] shifts
        # self.nb_types_space # will probably have to be defined on runtime

    def neighbored_astar(self, arr, i, j):
        # nb_shifts is an element at an index in the nb_location_space nb_location_space[255] is the holistic 8 nb space
        this_location_space = nbh(arr, i, j)
        if (0, 0) in this_location_space.keys():
            del this_location_space[(0, 0)]
        location_output = {}
        for k, v in this_location_space.items():
            if v not in location_output.keys():
                location_output[v] = []
            location_output[v].append(list(k))
        trimmed_location_output = {}
        for k, v in location_output.items():
            trimmed_location_output[k] = sorted(v)
        for k, v in trimmed_location_output.items():
            trimmed_location_output[k] = self.nb_location_space.index(v)

        this_point = nbh_conditional(arr, i, j)
        if (i, j) in this_point.keys():
            del this_point[(i, j)]
        output = {}
        for k, v in this_point.items():
            if v not in output.keys():
                output[v] = []
            output[v].append(list(k))
        trimmed_output = {k: len(v) for k, v in output.items()}
        return location_output, trimmed_location_output, output, trimmed_output

    def neighbored_situation_whole(self, maze):
        results = {}
        for n in range(1, maze.shape[0]-1):
            for m in range(1, maze.shape[1]-1):
                results[(n, m), maze[n][m]] = self.neighbored_astar(maze, n, m)
        return results

    def neighbored_situation_target(self, maze, test_points):
        results = {}
        test_points = test_points.tolist()
        for x in test_points:
            results.append(self.neighbored_astar(maze, x[0], x[1]))
        return results

    def assess_neighbored_holistic_testing(self):
        results = []
        for n in range(len(self.communication.testinputs)):
            results.append(self.neighbored_situation_whole(self.communication.testinputs[n]))
        return results

    def check_objective_against_neighbored(self, bg):
        copy_objective = deepcopy(self.running_objective)
        changed = False
        if not is_list_of_list_of_list(copy_objective):
            changed = True
            copy_objective = [copy_objective]
        for x in range(len(copy_objective)):
            for n in range(len(copy_objective[x])):
                this_check = copy_objective[x][n]
                if len(this_check) == 2:
                    first_target_coords = retrieve_coords_from_assignments(this_check[0], self.communication.asssignments_output)
                    second_target_coords = retrieve_coords_from_assignments(this_check[1], self.communication.asssignments_output)
                    if len(first_target_coords) == len(second_target_coords):
                        for m in range(len(first_target_coords)):
                            first_target_result = self.neighbored_situation_target(self.communication.traininputs[m], first_target_coords[m], bg)
                            second_target_result = self.neighbored_situation_target(self.communication.traininputs[m], second_target_coords[m], bg)
                            # comparison = [x != y  for x, y in zip(first_target_result, second_target_result)]
                            # if comparison[2]:
                            #     if first_target_result[2] and not second_target_result[2]:
                            #         cap = (this_check[0])
                            #         uncap = (this_check[1])
                            #     elif not first_target_result[2] and second_target_result[2]:
                            #         cap = (this_check[1])
                            #         uncap = (this_check[0])
                            #     copy_objective[x][n] = [this_check[0], this_check[1], 'cap_all', uncap, cap]
                            #     continue
                            # elif comparison[0] or comparison[1]:
                            #     if comparison[0]:
                            #         target, method = 0, 'cap_ver_hor'
                            #     else:
                            #         target, method = 1, 'cap_diag'
                            #     if first_target_result[target] and not second_target_result[target]:
                            #         cap = (this_check[0])
                            #         uncap = (this_check[1])
                            #     elif not first_target_result[target] and second_target_result[target]:
                            #         cap = (this_check[1])
                            #         uncap = (this_check[0])
                            #     copy_objective[x][n] = [this_check[0], this_check[1], method, uncap, cap]
                            #     continue
        if changed:
            return copy_objective[0]
        else:
            return copy_objective

    def assess_neighbored_target_training(self, bg):
        if len(self.running_objective) > 20:
            return self.running_objective, []

        new_objective = self.check_objective_against_neighbored(bg)
        test_rep = []
        if new_objective != self.running_objective:
            test_rep = self.assess_neighbored_holistic_testing(bg)

        return new_objective, test_rep

    def infer_on_mechanism(self, signal_, direct = False): # signal_ = b above
        mechanisms = set()
        decided_change_tuples = set()

        # make sure you turn off first not last, this is an example of massaging an objective
        for n in range(len(self.running_objective)-1):
            if len(self.running_objective[n]) == 3 and len(self.running_objective[n+1]) == 3 and (self.running_objective[n][1] != 'bg' and self.running_objective[n+1][1] == 'bg'):
                temp = self.running_objective[n]
                self.running_objective[n] = self.running_objective[n+1]
                self.running_objective[n+1] = temp

        # this is a heuristic for red cases
        if self.communication.objective_status == 'red' or (self.communication.objective_status == 'obd' and all([len(x) == 5 or len(x) == 3 for x in self.running_objective])):
            # In case of 'obd' cases, the test expectation is not ready to handle unsatisfiable objectives
            self.running_objective = self.communication.set_test_expectations(self.running_objective)

        for n in self.running_objective:
            if len(n) > 2:
                if n[2] == 'direct' and direct: # Major hyperparameter candidate
                    mechanisms.add(n[2])
                    self.cur_test_preds = self.communication.apply_direct_transformation((n[0], n[1]), self.cur_test_preds)
                    # in cur_test_preds, replace n[0] with n[1]
                # elif n[2] in ['cap_all', 'cap_ver_hor', 'cap_diag']:
                #     mechanisms.add(n[2])
                #     if n[2] == 'cap_all':
                #         this_signal = signal_[2]
                #     elif n[2] == 'cap_ver_hor':
                #         this_signal = signal_[0]
                #     elif n[2] == 'cap_diag':
                #         this_signal = signal_[1]
                #
                #     if n[3][0] == n[3][1]:
                #         coords = [x[1] for x in this_signal]
                #         if n[4] not in decided_change_tuples:
                #             decided_change_tuples.add(n[4])
                #             self.cur_test_preds = self.communication.objectively_build_a_prediction(n[4], coords, self.cur_test_preds)
                #
                #     elif n[4][0] == n[4][1]:
                #         coords = [x[0] for x in this_signal]
                #         if n[3] not in decided_change_tuples:
                #             decided_change_tuples.add(n[3])
                #             self.cur_test_preds = self.communication.objectively_build_a_prediction(n[3], coords, self.cur_test_preds)
                #     else:
                #         if n[3] not in decided_change_tuples and n[4] not in decided_change_tuples: # we can't repeat assignments, very imp precondition
                #             uncap_coords = [x[0] for x in this_signal]
                #             if n[3] not in decided_change_tuples:
                #                 decided_change_tuples.add(n[3])
                #                 self.cur_test_preds = self.communication.objectively_build_a_prediction(n[3], uncap_coords, self.cur_test_preds)
                #             cap_coords = [x[1] for x in this_signal]
                #             if n[4] not in decided_change_tuples:
                #                 decided_change_tuples.add(n[4])
                #                 self.cur_test_preds = self.communication.objectively_build_a_prediction(n[4], cap_coords, self.cur_test_preds)
        return sorted(list(mechanisms))

    def screen_neighbored(self):# This is abstract
        if self.communication.objective_status == None and self.communication.is_unique_output != None: # and is used rather than or in order to prevent random capturing modules
            pass
        else:
            is_solved = 'unsolved'
            mechanisms = []
            if type(self.communication.bg) == int:
                a, b =  self.assess_neighbored_target_training(self.communication.bg)
                if len(b) > 0: # this condition may not be enough
                    self.running_objective = a
                    objective_satisfiability = [len(x) == 3 or len(x) == 5 for x in a]
                    if all(objective_satisfiability):
                        mechanisms = self.infer_on_mechanism(b, True) # we shall trigger any direct at this point
                    elif any(objective_satisfiability):
                        mechanisms = self.infer_on_mechanism(b)
                    if all([np.array_equal(x, y) for x, y in zip(self.cur_test_preds, self.communication.testoutputs)]):
                        is_solved = 'solved'
                    elif any(objective_satisfiability):
                        is_solved = 'objective initiated'
                    else:
                        is_solved = 'unsolved'

            return is_solved, mechanisms, self.cur_test_preds, self.running_objective
