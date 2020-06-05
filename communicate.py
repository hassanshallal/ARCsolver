# this is a class that wraps important information in a single object to be passed to anyother class
from utils import *
class Communication:
    def __init__(self, traininputs, trainoutputs, testinputs, bg, objective_status, objective, assignments_leads, asssignments_output, token_to_colors, test_token_to_color, testoutputs = None):
        self.traininputs = traininputs
        self.trainoutputs = trainoutputs
        self.testinputs = testinputs
        self.bg = bg

        self.objective_status = objective_status
        self.objective = objective
        self.assignments_leads = assignments_leads
        self.asssignments_output = asssignments_output
        self.token_to_colors = token_to_colors
        self.test_token_to_color = test_token_to_color
        self.testoutputs = testoutputs

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

    def first_principles_diff_dim_signal_class(self, results):
        overall_cur_invest_num_unique = []
        overall_cur_invest_unique = []
        for n in range(len(results)): # 3 for captured
            cur_invest = results[n]
            cur_invest_num_unique = []
            cur_invest_unique = []
            for x in range(len(cur_invest)): # num_train
                this_result = cur_invest[x]
                values_1 = get_set_of_values_1(this_result)
                cur_invest_num_unique.append(len(values_1))
                if len(values_1) == 1:
                    cur_invest_unique.append(values_1.pop())
                else:
                    cur_invest_unique.append(values_1)
            overall_cur_invest_num_unique.append(cur_invest_num_unique)
            overall_cur_invest_unique.append(cur_invest_unique)

        return overall_cur_invest_num_unique, overall_cur_invest_unique

    # The following are supposed to be generic, unfortunately  infer_on_mechanism is geared towards captured module

    def set_test_expectations(self, current_objective):
        assignments_leads = deepcopy(self.assignments_leads)
        test_token_to_color = deepcopy(self.test_token_to_color)

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

    def objectively_build_a_prediction(self, change_tuple, coordinates, cur_test_pred):
        assert(len(coordinates) == len(cur_test_pred))

        # this is to cover colors to only show in the output and to get it from the training!
        # We tokenized testinputs based on the traininputs
        if change_tuple[1] not in self.test_token_to_color.keys():
            for x in self.token_to_colors:
                for k, v in x.items():
                    if change_tuple[1] == k:
                        self.test_token_to_color[change_tuple[1]] = v

        for n in range(len(coordinates)):
            these_coords = coordinates[n]
            for m in these_coords:
                coord = m[1]
                cur = cur_test_pred[n][coord[0]][coord[1]]
                if change_tuple[0] in self.test_token_to_color.keys() and cur == self.test_token_to_color[change_tuple[0]]:
                    cur_test_pred[n][coord[0]][coord[1]] = self.test_token_to_color[change_tuple[1]]
        return cur_test_pred

    def apply_direct_transformation(self, change_tuple, cur_test_pred):
        for x in range(len(cur_test_pred)):
            for n in range(cur_test_pred[x].shape[0]):
                for m in range(cur_test_pred[x].shape[1]):
                    cur = cur_test_pred[x][n][m]
                    if change_tuple[0] in self.test_token_to_color.keys() and change_tuple[1] in self.test_token_to_color.keys() and cur == self.test_token_to_color[change_tuple[0]]:
                        cur_test_pred[x][n][m] = self.test_token_to_color[change_tuple[1]]
        return cur_test_pred

    def carry_along(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        retrun
