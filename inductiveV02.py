# should be able to get started with this very soon. may be or may be not.

from utils import *
from capturedV02 import *

class Inductive:
    def __init__(self, traininputs, trainoutputs, testinputs, objective_status, objective, assignments_leads, asssignments_output, bg, token_to_colors, size_sorted_nonbg, int_anchor_vals, testoutputs = None):
        self.traininputs = traininputs
        self.trainoutputs = trainoutputs
        self.testinputs = testinputs
        self.objective_status = objective_status
        self.objective = objective
        self.assignments_leads = assignments_leads
        self.asssignments_output = asssignments_output
        self.bg = bg
        self.token_to_colors = token_to_colors
        self.size_sorted_nonbg = size_sorted_nonbg
        self.int_anchor_vals = int_anchor_vals
        self.testoutputs = testoutputs

        # These are important and can be retrieved anytime
        self.running_objective = deepcopy(self.objective)
        self.cur_train_preds = deepcopy(self.traininputs)
        self.cur_test_preds = deepcopy(self.testinputs)

        # work out your self.test_token_to_color: 227, 328 are example of a blind spot of this system with 'direct' strategy
        self.test_token_to_color = self.get_test_token_to_color()

        # Induction
        self.solved, self.mechanisms, self.testpreds = self.inductive_strategy()

    def get_test_token_to_color(self):
        if len(self.token_to_colors) > 0:
            if all([x == self.token_to_colors[0] for x in self.token_to_colors]):
                return self.token_to_colors[0]
            else:
                pr_tokens = get_priority_token(self.token_to_colors)
                testinputs_vals = [np.unique(n).tolist() for n in self.testinputs]
                final_test_nonbg = sorted(list(get_common_nonbg_inputs(testinputs_vals)))
                test_color_to_token = get_col_to_token_class_c_test(final_test_nonbg, pr_tokens, self.bg, self.size_sorted_nonbg, self.int_anchor_vals[0], self.testinputs[0])
                return get_token_to_color_class_C(test_color_to_token)
# if len(pr_tokens) > 0: # this is the only way to transfer
# else: return {}

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

        else:
            a, b =  assess_captured_target_training(self.asssignments_output, self.running_objective, self.bg, self.traininputs, self.testinputs)
            # we need to create outputs out of the modified objective and the b (testinputs as sets)
            if len(b) > 0:
                self.running_objective = a
                objective_satisfiability = [len(x) == 3 or len(x) == 5 for x in a]
                if all(objective_satisfiability):
                    mechanisms = self.get_mechanism_preds(b, True)
                    if all([np.array_equal(x, y) for x, y in zip(self.cur_test_preds, self.testoutputs)]):
                        return 'solved', mechanisms, self.cur_test_preds
                    elif any([np.array_equal(x, y) for x, y in zip(self.cur_test_preds, self.testinputs)]):
                        return 'unsolved', [], []
                    else:
                        return 'partially solved', mechanisms, self.cur_test_preds
                elif any(objective_satisfiability):
                    mechanisms = self.get_mechanism_preds(b)
                    if any([np.array_equal(x, y) for x, y in zip(self.cur_test_preds, self.testinputs)]):
                        return 'unsolved', [], []
                    else:
                        return 'partially solved', mechanisms, self.cur_test_preds
                else:
                    return 'unsolved', [], []
            else:
                return 'unsolved', [], []

    def objectively_build_a_prediction(self, change_tuple, coordinates):
        assert(len(coordinates) == len(self.cur_test_preds))
        # this is to cover colors to only show in the output and to get it from the training!
        if change_tuple[1] not in self.test_token_to_color.keys():
            for x in self.token_to_colors:
                for k, v in x.items():
                    if change_tuple[1] == k:
                        self.test_token_to_color[change_tuple[1]] = v

        for n in range(len(coordinates)):
            these_coords = coordinates[n]
            for m in these_coords:
                coord = m[1]
                cur = self.cur_test_preds[n][coord[0]][coord[1]]
                if change_tuple[0] in self.test_token_to_color.keys() and cur == self.test_token_to_color[change_tuple[0]]:
                    self.cur_test_preds[n][coord[0]][coord[1]] = self.test_token_to_color[change_tuple[1]]

    def apply_direct_transformation(self, change_tuple):
        for x in range(len(self.cur_test_preds)):
            for n in range(self.cur_test_preds[x].shape[0]):
                for m in range(self.cur_test_preds[x].shape[1]):
                    cur = self.cur_test_preds[x][n][m]
                    if change_tuple[0] in self.test_token_to_color.keys() and change_tuple[1] in self.test_token_to_color.keys() and cur == self.test_token_to_color[change_tuple[0]]:
                        self.cur_test_preds[x][n][m] = self.test_token_to_color[change_tuple[1]]


    def set_test_expectations(self):
        current_objective = deepcopy(self.running_objective)
        assignments_leads = deepcopy(self.assignments_leads)
        test_token_to_color = deepcopy(self.test_token_to_color)
        if is_all_nonbg_in_cur_obj(current_objective) and 'bg' in test_token_to_color.keys():
            del test_token_to_color['bg']
        if len(test_token_to_color) > len(current_objective):
            diff = set(test_token_to_color.keys()) - get_bare_assignment_leads(assignments_leads)
            for n in list(diff):
                current_objective.append(expand_obj(deepcopy(current_objective[0]), n))
            return current_objective
        elif len(test_token_to_color) < len(current_objective):
            target = set(test_token_to_color.keys())
            new_objective = []
            for n in list(target):
                new_objective.append(find_target_obj(current_objective, n))
            return new_objective
        else: # this is a more difficult case where we need to really convolve more cognition
            return self.running_objective

    def get_mechanism_preds(self, signal_, direct = False): # signal_ = b above
        mechanisms = set()
        decided_change_tuples = set()

        # make sure you turn off first not last, this is an example of massaging an objective
        for n in range(len(self.running_objective)-1):
            if len(self.running_objective[n]) == 3 and len(self.running_objective[n+1]) == 3 and (self.running_objective[n][1] == 'bg' or self.running_objective[n+1][1] == 'bg'):
                temp = self.running_objective[n]
                self.running_objective[n] = self.running_objective[n+1]
                self.running_objective[n+1] = temp

        # this is a heuristic for red cases
        if self.objective_status == 'red':
            self.running_objective = self.set_test_expectations()

        for n in self.running_objective:
            if len(n) > 2:
                if n[2] == 'direct' and direct:
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
