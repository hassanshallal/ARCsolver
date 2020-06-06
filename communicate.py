# this is a class that wraps important information in a single object to be passed to anyother class
from utils import *

class Communication:
    def __init__(self, traininputs, trainoutputs, testinputs, bg, objective_status, objective, assignments_leads, asssignments_output, token_to_colors, test_token_to_color, is_unique_output, testoutputs = None):
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
        self.is_unique_output = is_unique_output
        self.testoutputs = testoutputs

        # The following data members are added along the way and used by other modules
        # The following two data members will be updated with the dimensionWORK

        # Hyperparameters for inductive or any of its modules can be passed here
        # self.turn_on_direct_premature = False
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
            new_objective = []
            for n in current_objective:
                if len(n) == 3 and n[0] in test_token_to_color.keys():
                    new_objective.append(n)

            target = set(test_token_to_color.keys())
            for n in list(target):
                is_found_obj = find_target_obj(current_objective, n)
                if is_found_obj != None:
                    new_objective.append(is_found_obj)

            return new_objective

        return current_objective # this is a more difficult case where we need to really convolve more cognition

    # creating output with objectives
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

    # Creating and populating output in cases with no objective:
    def unobjectively_build_a_prediction(self, dimension_status, output_dim_preds, **kwargs):
        if dimension_status == 'deduced':
            cur_output = np.zeros(output_dim_preds)
            for k, v in kwargs.items():
                if k == 'one_unique_captured':
                    cur_output += v
                    cur_output = cur_output.astype(int)
                return [y.tolist() for y in cur_output]
        return None

    # Transfer cargo
    def carry_along_args(self, *args):
        self.args = args
        return

    def carry_along_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            if k == 'dimension_status':
                self.dimension_status = v
            if k == 'output_dim_preds':
                self.output_dim_preds = v

        return
