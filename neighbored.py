
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
    all_combs.add(tuple(nb_shifts)) # added 8

    for n in nb_shifts:
        all_combs.add((n, )) # added ones

    for n in range(2, 8): # between 2 and 8
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
        return location_output, trimmed_location_output, output, trimmed_output, len(trimmed_output)

    def neighbored_situation_whole(self, maze):
        results = {}
        for n in range(0, maze.shape[0]):
            for m in range(0, maze.shape[1]):
                results[(n, m), maze[n][m]] = self.neighbored_astar(maze, n, m)
        return results

    def neighbored_situation_target(self, maze, test_points):
        results = {}
        test_points = test_points.tolist()
        for x in test_points:
            results[tuple(x)] = self.neighbored_astar(maze, x[0], x[1])
        return results


    # the following two compare functions focus on only one nb based comparison.
    def get_set_of_lengths(self, results_dict):
        lens = set()
        lens_association = {}
        for k, v in results_dict.items():
            lens.add(v[4])
            if v[4] not in lens_association.keys():
                lens_association[v[4]] = []
            lens_association[v[4]].append((k, v[3]))
            print(v[4], k, v[3])

        return lens, lens_association

    def compare_len_trimmed_dicts_targeted(self, result1, result2):
        len_1 , lens_association_1 = self.get_set_of_lengths(result1)
        len_2, lens_association_2 = self.get_set_of_lengths(result2)
        return len_1 != len_2, (len_1, len_2), (lens_association_1, lens_association_2)

    def compare_len_trimmed_dicts_holistic(self, result, code = None):
        test_outputs = []
        for m in range(len(result)):
            len_1 , lens_association_1 = self.get_set_of_lengths(result[m])
            if code == None:
                test_outputs.append((len_1 , lens_association_1))
            else:
                first_group = sorted(list(code[0]))
                first_group_coords = []
                for n in first_group:
                    if n in lens_association_1.keys():
                        first_group_coords.append(lens_association_1[n])

                second_group = sorted(list(code[0]))
                second_group_coords = []
                for n in second_group:
                    if n in lens_association_1.keys():
                        second_group_coords.append(lens_association_1[n])

                test_outputs.append((first_group_coords, second_group_coords))
        return test_outputs


    def check_objective_against_neighbored(self): # return a new objective and a neighbor code
        copy_objective = deepcopy(self.running_objective)
        print('copy_objective a: ', copy_objective)
        changed = False
        if not is_list_of_list_of_list(copy_objective):
            changed = True
            copy_objective = [copy_objective]

        print('copy_objective b: ', copy_objective)
        y = tuple() # to be returned instead of a code
        for x in range(len(copy_objective)):
            for n in range(len(copy_objective[x])):
                this_check = copy_objective[x][n]
                print(this_check)
                if len(this_check) == 2:
                    first_target_coords = retrieve_coords_from_assignments(this_check[0], self.communication.asssignments_output)
                    second_target_coords = retrieve_coords_from_assignments(this_check[1], self.communication.asssignments_output)
                    if len(first_target_coords) == len(second_target_coords):
                        for m in range(len(first_target_coords)):
                            first_target_result = self.neighbored_situation_target(self.communication.traininputs[m], first_target_coords[m])
                            second_target_result = self.neighbored_situation_target(self.communication.traininputs[m], second_target_coords[m])
                            w, y, z = self.compare_len_trimmed_dicts_targeted(first_target_result, second_target_result)
                            if w:
                                print('caught nbed')
                                print('y is:', y)
                                if len(y[0]) > len(y[1]):
                                    print('exchange!')
                                    nbed = (this_check[0])
                                    unnbed =  (this_check[1])
                                    new_tuple = tuple((y[1], y[0]))
                                    y = new_tuple
                                    copy_objective[x][n] = [this_check[0], this_check[1], 'num_nbed', unnbed, nbed]
                                elif len(y[0]) < len(y[1]):
                                    print('no exchange!')
                                    nbed = (this_check[1])
                                    unnbed =  (this_check[0])
                                    copy_objective[x][n] = [this_check[0], this_check[1], 'num_nbed', unnbed, nbed]
        print('copy_objective after: ', copy_objective)
        if changed:
            return copy_objective[0], y
        else:
            return copy_objective, y

    def assess_neighbored_targeted_testing(self, candidate_code):
        results = []
        for n in range(len(self.communication.testinputs)):
            results.append(self.neighbored_situation_whole(self.communication.testinputs[n]))

        return self.compare_len_trimmed_dicts_holistic(results, candidate_code)


    def assess_neighbored_target_training(self):
        if len(self.running_objective) > 20:
            return self.running_objective, []

        new_objective, candidate_code = self.check_objective_against_neighbored()
        print('new_objective: ', new_objective)
        test_rep = []
        if new_objective != self.running_objective and candidate_code != tuple():
            test_rep = self.assess_neighbored_targeted_testing(candidate_code)

        return new_objective, test_rep

    def infer_on_mechanism(self, signal_, direct = False): # signal_ = b above
        print('entered infer_on_mechanism')
        mechanisms = set()
        decided_change_tuples = set()

        # make sure you turn off first not last, this is an example of massaging an objective
        for n in range(len(self.running_objective)-1):
            print('a in infer_on_mechanism')
            if len(self.running_objective[n]) == 3 and len(self.running_objective[n+1]) == 3 and (self.running_objective[n][1] != 'bg' and self.running_objective[n+1][1] == 'bg'):
                temp = self.running_objective[n]
                self.running_objective[n] = self.running_objective[n+1]
                self.running_objective[n+1] = temp

        # this is a heuristic for red cases
        if self.communication.objective_status == 'red' or (self.communication.objective_status == 'obd' and all([len(x) == 5 or len(x) == 3 for x in self.running_objective])):
            print('b in infer_on_mechanism')
            # In case of 'obd' cases, the test expectation is not ready to handle unsatisfiable objectives
            self.running_objective = self.communication.set_test_expectations(self.running_objective)

        for n in self.running_objective:
            print(self.running_objective)
            print('c in infer_on_mechanism')
            if len(n) > 2:
                print('d in infer_on_mechanism')
                if n[2] == 'direct' and direct: # Major hyperparameter candidate
                    print('e in infer_on_mechanism')
                    mechanisms.add(n[2])
                    self.cur_test_preds = self.communication.apply_direct_transformation((n[0], n[1]), self.cur_test_preds)
                elif n[2] == 'num_nbed':
                    print('f in infer_on_mechanism')
                    mechanisms.add(n[2])
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
                            unnbed_coords = [x[0] for x in this_signal]
                            if n[3] not in decided_change_tuples:
                                decided_change_tuples.add(n[3])
                                self.cur_test_preds = self.communication.objectively_build_a_prediction(n[3], uncap_coords, self.cur_test_preds)
                            nbed_coords = [x[1] for x in this_signal]
                            if n[4] not in decided_change_tuples:
                                decided_change_tuples.add(n[4])
                                self.cur_test_preds = self.communication.objectively_build_a_prediction(n[4], cap_coords, self.cur_test_preds)

            return sorted(list(mechanisms))

    def screen_neighbored(self):# This is abstract
        if self.communication.objective_status == None and self.communication.is_unique_output != None: # and is used rather than or in order to prevent random capturing modules
            pass
        else:
            is_solved = 'unsolved'
            mechanisms = []
            if type(self.communication.bg) == int:
                a, b =  self.assess_neighbored_target_training()
                if len(b) > 0 and type(b[0]) == list: # this condition may not be enough
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
