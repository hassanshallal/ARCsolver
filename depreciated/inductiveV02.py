# should be able to get started with this very soon. may be or may be not.

from utils import *
from communicate import *

from dimensionWork import *
from capturedV02 import *


class Inductive:
    def __init__(self, communication):
        self.communication = communication

        # These are important and can be retrieved anytime
        self.running_objective = deepcopy(self.communication.objective)
        self.cur_test_preds = deepcopy(self.communication.testinputs)
        self.solved, self.mechanisms = 'unsolved', []
        # cognify and analyze dimensions using dimensionWork
        self.dimensionWork = DimensionWork(self.communication, self.cur_test_preds)

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

    # The following are supposed to be generic, unfortunately  infer_on_mechanism is geared towards captured module
    def set_test_expectations(self):
        current_objective = deepcopy(self.running_objective)
        assignments_leads = deepcopy(self.communication.assignments_leads)
        test_token_to_color = deepcopy(self.communication.test_token_to_color)

        if is_all_nonbg_in_cur_obj(current_objective) and 'bg' in test_token_to_color.keys():
            del test_token_to_color['bg']

        if len(test_token_to_color) > len(current_objective):
            diff = set(test_token_to_color.keys()) - get_bare_assignment_leads(assignments_leads)
            for n in list(diff):
                current_objective.append(expand_obj(deepcopy(current_objective[0]), n))
            return current_objective
        if len(test_token_to_color) < len(current_objective):
            target = set(test_token_to_color.keys())
            new_objective = []
            for n in list(target):
                new_objective.append(find_target_obj(current_objective, n))
            return new_objective

        return current_objective # this is a more difficult case where we need to really convolve more cognition

    def objectively_build_a_prediction(self, change_tuple, coordinates):
        assert(len(coordinates) == len(self.cur_test_preds))

        # this is to cover colors to only show in the output and to get it from the training!
        # We tokenized testinputs based on the traininputs
        if change_tuple[1] not in self.communication.test_token_to_color.keys():
            for x in self.communication.token_to_colors:
                for k, v in x.items():
                    if change_tuple[1] == k:
                        self.communication.test_token_to_color[change_tuple[1]] = v

        for n in range(len(coordinates)):
            these_coords = coordinates[n]
            for m in these_coords:
                coord = m[1]
                cur = self.cur_test_preds[n][coord[0]][coord[1]]
                if change_tuple[0] in self.communication.test_token_to_color.keys() and cur == self.communication.test_token_to_color[change_tuple[0]]:
                    self.cur_test_preds[n][coord[0]][coord[1]] = self.communication.test_token_to_color[change_tuple[1]]

    def apply_direct_transformation(self, change_tuple):
        for x in range(len(self.cur_test_preds)):
            for n in range(self.cur_test_preds[x].shape[0]):
                for m in range(self.cur_test_preds[x].shape[1]):
                    cur = self.cur_test_preds[x][n][m]
                    if change_tuple[0] in self.communication.test_token_to_color.keys() and change_tuple[1] in self.communication.test_token_to_color.keys() and cur == self.communication.test_token_to_color[change_tuple[0]]:
                        self.cur_test_preds[x][n][m] = self.communication.test_token_to_color[change_tuple[1]]

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
            self.running_objective = self.set_test_expectations()

        for n in self.running_objective:
            if len(n) > 2:
                if n[2] == 'direct' and direct: # Major hyperparameter candidate
                    mechanisms.add(n[2])
                    self.apply_direct_transformation((n[0], n[1]))
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
                            self.objectively_build_a_prediction(n[4], coords)

                    elif n[4][0] == n[4][1]:
                        coords = [x[0] for x in this_signal]
                        if n[3] not in decided_change_tuples:
                            decided_change_tuples.add(n[3])
                            self.objectively_build_a_prediction(n[3], coords)
                    else:
                        if n[3] not in decided_change_tuples and n[4] not in decided_change_tuples: # we can't repeat assignments, very imp precondition
                            uncap_coords = [x[0] for x in this_signal]
                            if n[3] not in decided_change_tuples:
                                decided_change_tuples.add(n[3])
                                self.objectively_build_a_prediction(n[3], uncap_coords)
                            cap_coords = [x[1] for x in this_signal]
                            if n[4] not in decided_change_tuples:
                                decided_change_tuples.add(n[4])
                                self.objectively_build_a_prediction(n[4], cap_coords)

        return sorted(list(mechanisms))

    # The following is the screening protocola which needs alot of cleaning
    # ya the screen dimensions is a scary method, needs a lot of work now gets: 222, 268, 288, 306
    # screen dimensions
    def screen_captured(self): # This is abstract
        if self.communication.objective_status == 'None':
            unique_train_outputs = [np.unique(x).tolist() for x in self.communication.trainoutputs]
            is_unique_train_outputs = [len(x) == 1 for x in unique_train_outputs]
            a, b = assess_captured_holistic_training(self.communication.traininputs, self.communication.testinputs)
            if not self.communication.first_principles_diff_dim_checks(a) or not all(is_unique_train_outputs):
                return 'unsolved', [], []
            else:
                if all(is_unique_train_outputs):
                    unique_train_outputs = [x[0] for x in unique_train_outputs]
                    overall_cur_invest_num_unique, overall_cur_invest_unique = self.communication.first_principles_diff_dim_signal_class(a)
                    m, n = self.validate_cap(overall_cur_invest_num_unique, overall_cur_invest_unique, 2, unique_train_outputs, b)
                    if m:
                        return 'solved', ['cap_all', 'one_unique_captured'], [[[l] for l in n[2]]]
                    m, n = self.validate_cap(overall_cur_invest_num_unique, overall_cur_invest_unique, 0, unique_train_outputs, b)
                    if m:
                        return 'solved', ['cap_ver_hor', 'one_unique_captured'], [[[l] for l in n[0]]]
                    m, n = self.validate_cap(overall_cur_invest_num_unique, overall_cur_invest_unique, 1, unique_train_outputs, b)
                    if m:
                        return 'solved', ['cap_diag', 'one_unique_captured'], [[l] for l in n[1]]
                return 'unsolved', [], []

        else:
            a, b =  assess_captured_target_training(self.communication.asssignments_output, self.running_objective, self.communication.bg, self.communication.traininputs, self.communication.testinputs)
            # we need to create outputs out of the modified objective and the b (testinputs as sets)
            if len(b) > 0:
                self.running_objective = a
                objective_satisfiability = [len(x) == 3 or len(x) == 5 for x in a]
                if all(objective_satisfiability):
                    mechanisms = self.infer_on_mechanism(b, True)
                    if all([np.array_equal(x, y) for x, y in zip(self.cur_test_preds, self.communication.testoutputs)]):
                        return 'solved', mechanisms, self.cur_test_preds
                    elif any([np.array_equal(x, y) for x, y in zip(self.cur_test_preds, self.communication.testinputs)]):
                        return 'unsolved', [], []
                    else:
                        return 'partially solved', mechanisms, self.cur_test_preds
                elif any(objective_satisfiability):
                    mechanisms = self.infer_on_mechanism(b)
                    if any([np.array_equal(x, y) for x, y in zip(self.cur_test_preds, self.communication.testinputs)]):
                        return 'unsolved', [], []
                    else:
                        return 'partially solved', mechanisms, self.cur_test_preds
                else:
                    return 'unsolved', [], []
            else:
                return 'unsolved', [], []

    def inductive_strategy(self): # we will change this into a multilane highway and a find_path
    # routines on samples to decide whether to send a positive or a negative feedback so as to stop

        # try dimension related
        self.solved, self.mechanisms,  this_testpred = self.dimensionWork.screen_dimesnions()
        if self.solved == 'solved':
            self.cur_test_preds = this_testpred
            return

        # try flips related
        self.solved, self.mechanisms, this_testpred = self.dimensionWork.screen_flips()
        if self.solved == 'solved':
            self.cur_test_preds = this_testpred
            return

        # try captured related, we will pass and recieve a modified running objective or none
        self.solved, self.mechanisms,  this_test_pred = self.screen_captured()
        if self.solved == 'solved' or self.solved == 'partially solved':
            self.cur_test_preds = this_test_pred
            return

        return
