# should be able to get started with this very soon. may be or may be not.

from utils import *
from capturedV02 import *

class Inductive:
    def __init__(self, traininputs, trainoutputs, testinputs, objective_status, objective, asssignments_output, bg, token_to_colors, testoutputs = None):
        self.traininputs = traininputs
        self.trainoutputs = trainoutputs
        self.testinputs = testinputs
        self.objective_status = objective_status
        self.objective = objective
        self.asssignments_output = asssignments_output
        self.bg = bg
        self.token_to_colors = token_to_colors
        self.testoutputs = testoutputs

        # These are important and can be retrieved anytime
        self.running_objective = deepcopy(self.objective)
        self.cur_train_preds = deepcopy(self.traininputs)
        self.cur_test_preds = deepcopy(self.testinputs)

        # Induction
        self.solved, self.mechanisms, self.testpreds = self.inductive_strategy()

    def inductive_strategy(self): # method for an instance of induction:
        # output: x = solved/unsolved, y = mechanisms (functions to apply), testpreds (preds to plot)
        # flips:
        x, y, z = self.screen_flips()
        if x != 'unsolved':
            return x, y, z

        # screen captured:
        x, y, z = self.screen_captured()
        if x != 'unsolved':
            return x, y, z

        return 'unsolved', [], []

    def screen_flips(self):
        # check on training
        are_flips = [screen_flips_rotation(in_, out_) for in_, out_ in zip(self.traininputs, self.trainoutputs)] # this took care of train validation
        if all([x[0] != False for x in are_flips]) and all([x[0] == are_flips[0][0] for x in are_flips]):
            if are_flips[0][1] != None:
                self.cur_test_preds = [are_flips[0][0](x, are_flips[0][1]) for x in self.cur_test_preds]
            else:
                self.cur_test_preds = [are_flips[0][0](x) for x in self.cur_test_preds]

            if self.testoutputs:
                test_evaluated = all([np.array_equal(x, y) for x, y in zip(self.cur_test_preds, self.testoutputs)])
            else:
                test_evaluated = False

            if test_evaluated:
                return 'solved', [(are_flips[0][0], are_flips[0][1])], self.cur_test_preds
            elif not test_evaluated:
                return 'partially solved', [(are_flips[0][0], are_flips[0][1])], self.cur_test_preds
        else:
            return 'unsolved', [], []


    # this method confirms we have signals to proceed, if all points are captured or uncaptured then there is no signal
    # [train_task_output_ver_hor, train_task_output_diag, train_task_output_all],
    # [test_task_output_ver_hor, test_task_output_diag, test_task_output_all]
    def is_list_one_value(list_ex):
        return all([x == list_ex[0] for x in list_ex])

    def is_list_equal_1(self, list_ex):
        return all([x == 1 for x in list_ex])

    def first_principles_diff_dim_checks(self, *args):
        for n in args:
            ext_len = len(n)
            int_len = len(n[0])
            for x in range(ext_len):
                for y in range(int_len):
                    if len(n[x][y][0]) == 0 or len(n[x][y][1]) == 0:
                        return False
        return True

    def first_principles_diff_dim_signal_size(self, *args):
        zero_signal = []
        one_signal = []

        for n in args:
            ext_len = len(n)
            int_len = len(n[0])
            for x in range(ext_len):
                for y in range(int_len):
                    zero_signal.append(len(n[x][y][0]))
                    one_signal.append(len(n[x][y][1]))

        return zero_signal, one_signal

    def get_set_of_values_1(self, this_dict):
        values = set()
        target = this_dict[1]
        for n in range(len(target)):
            values.add(target[n][0])
        return values


    def first_principles_diff_dim_signal_class(self, results):
        overall_cur_invest_num_unique = []
        overall_cur_invest_unique = []
        for n in range(len(results)): # 3 for captured
            cur_invest = results[n]
            cur_invest_num_unique = []
            cur_invest_unique = []
            for x in range(len(cur_invest)): # num_train
                this_result = cur_invest[x]
                values_1 = self.get_set_of_values_1(this_result)
                cur_invest_num_unique.append(len(values_1))
                if len(values_1) == 1:
                    cur_invest_unique.append(values_1.pop())
                else:
                    cur_invest_unique.append(values_1)
            overall_cur_invest_num_unique.append(cur_invest_num_unique)
            overall_cur_invest_unique.append(cur_invest_unique)

        return overall_cur_invest_num_unique, overall_cur_invest_unique

    def validate_cap(self, overall_cur_invest_num_unique, overall_cur_invest_unique, order_, unique_train_outputs, b):
        if self.is_list_equal_1(overall_cur_invest_num_unique[order_]):
            if overall_cur_invest_unique[order_] == unique_train_outputs:
                test_num_unique, test_unique = self.first_principles_diff_dim_signal_class(b)
                if self.is_list_equal_1(test_num_unique[order_]):
                    return True, test_unique
            else:
                return False, []
        else:
            return False, []


    def screen_captured(self):
        if self.objective_status == 'None':
            unique_train_outputs = [np.unique(x).tolist() for x in self.trainoutputs]
            is_unique_train_outputs = [len(x) == 1 for x in unique_train_outputs]
            a, b = assess_captured_holistic_training(self.traininputs, self.testinputs)
            if not self.first_principles_diff_dim_checks(a) or not all(is_unique_train_outputs):
                return 'unsolved', [], []
            else:
                if all(is_unique_train_outputs):
                    unique_train_outputs = [x[0] for x in unique_train_outputs]
                    overall_cur_invest_num_unique, overall_cur_invest_unique = self.first_principles_diff_dim_signal_class(a)
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

        elif self.objective_status == 'obd' or  self.objective_status == 'red':
            a, b =  assess_captured_target_training(self.asssignments_output, self.running_objective, self.bg, self.traininputs, self.testinputs)
            # we need to create outputs out of the modified objective and the b (testinputs as sets)
            if len(b) > 0:
                self.running_objective = a
                objective_satisfiability = [len(x) == 3 or len(x) == 5 for x in a]
                if all(objective_satisfiability):
                    if all([x == self.token_to_colors[0] for x in self.token_to_colors]):
                        mechanisms = self.get_mechanism_preds(b)
                        if all([np.array_equal(x, y) for x, y in zip(self.cur_test_preds, self.testoutputs)]):
                            return 'solved', mechanisms, self.cur_test_preds
                        else:
                            return 'partially solved', mechanisms, self.cur_test_preds
                    else:
                        return 'unsolved', [], []

                elif any(objective_satisfiability):
                    if all([x == self.token_to_colors[0] for x in self.token_to_colors]):
                        mechanisms = self.get_mechanism_preds(b)
                        return 'partially solved', mechanisms, self.cur_test_preds
                    else:
                        return 'unsolved', [], []
                else:
                    return 'unsolved', [], []
            else:
                return 'unsolved', [], []
        elif self.objective_status == 'pred' or self.objective_status == 'irr':
            return 'unsolved', [], []

    def objectively_build_a_prediction(self, change_tuple, coordinates):
        assert(len(coordinates) == len(self.cur_test_preds))
        for n in range(len(coordinates)):
            these_coords = coordinates[n]
            for m in these_coords:
                coord = m[1]
                cur = self.cur_test_preds[n][coord[0]][coord[1]]
                if cur == self.token_to_colors[0][change_tuple[0]]:
                    self.cur_test_preds[n][coord[0]][coord[1]] = self.token_to_colors[0][change_tuple[1]]

    def apply_direct_transformation(self, change_tuple):
        for x in range(len(self.cur_test_preds)):
            for n in range(self.cur_test_preds[x].shape[0]):
                for m in range(self.cur_test_preds[x].shape[1]):
                    cur = self.cur_test_preds[x][n][m]
                    if cur == self.token_to_colors[0][change_tuple[0]]:
                        self.cur_test_preds[x][n][m] = self.token_to_colors[0][change_tuple[1]]
        return


    def get_mechanism_preds(self, signal_): # signal_ = b above
        mechanisms = set()
        for n in self.running_objective:
            if len(n) > 2:
                if n[2] == 'direct':
                    mechanisms.add(n[2])
                    self.apply_direct_transformation((n[0], n[1]))
                    # in cur_test_preds, replace n[0] with n[1]
                else:
                    mechanisms.add(n[2])
                    if n[2] == 'cap_all':
                        this_signal = signal_[2]
                    elif n[2] == 'cap_ver_hor':
                        this_signal = signal_[0]
                    elif n[2] == 'cap_diag':
                        this_signal = signal_[1]

                    if n[3][0] == n[3][1]:
                        coords = [x[1] for x in this_signal]
                        self.objectively_build_a_prediction(n[4], coords)
                    elif n[4][0] == n[4][1]:
                        coords = [x[0] for x in this_signal]
                        self.objectively_build_a_prediction(n[3], coords)
                    else:
                        uncap_coords = [x[0] for x in this_signal]
                        self.objectively_build_a_prediction(n[3], uncap_coords)
                        cap_coords = [x[1] for x in this_signal]
                        self.objectively_build_a_prediction(n[4], cap_coords)

        return sorted(list(mechanisms))
