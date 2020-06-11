# this is a class that wraps important information in a single object to be passed to anyother class
from utils import *

class Communication:
    def __init__(self, traininputs, trainoutputs, testinputs, bg, objective_status, objective, assignments_leads, asssignments_output, token_to_colors, test_token_to_color, testinputs_vals, testoutputs = None):
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
        self.testinputs_vals = testinputs_vals
        self.testoutputs = testoutputs

        # The following data members are added along the way and used by other modules
        self.is_unique_output = self.find_unique_output()
        self.freq_nonbg_traininputs = self.get_freq_nonbg_inputs(self.traininputs) # just the number of nonbg: utilized only in dimensionWork
        self.freq_nonbg_testinputs = self.get_freq_nonbg_inputs(self.testinputs) # just the number of nonbg: utilized only in dimensionWork
        self.frequency_counter_traininputs = [Counter(list(itertools.chain.from_iterable(x.tolist()))).most_common() for x in self.traininputs]
        self.frequency_counter_testinputs = [Counter(list(itertools.chain.from_iterable(x.tolist()))).most_common() for x in self.testinputs]


        # Hyperparameters for inductive or any of its modules can be passed here
        # self.turn_on_direct_premature = False
    def get_frequency_graph(self, frequency_counter):
        frequency_graph = []
        for x in frequency_counter:
            this_list = []
            for n in x:
                this_list.append(n[0])
                this_list.append(n[1])
            frequency_graph.append(this_list)
        return frequency_graph

    def find_unique_output(self):
        unique_train_outputs = [np.unique(x).tolist() for x in self.trainoutputs]
        is_unique_train_outputs = [len(x) == 1 for x in unique_train_outputs]
        if all(is_unique_train_outputs):
            return [x[0] for x in unique_train_outputs]
        return None

    def get_freq_nonbg_inputs(self, lists):
        if type(self.bg) == int:
            return [np.sum(x != self.bg) for x in lists]
        elif type(self.bg) == list:
            return [np.sum(x != y) for x, y in zip(lists, self.bg)]

    # The following are supposed to be generic, unfortunately  infer_on_mechanism is geared towards captured module

    def set_test_expectations(self, current_objective):
        assignments_leads = deepcopy(self.assignments_leads)
        test_token_to_color = deepcopy(self.test_token_to_color)

        if is_all_nonbg_in_cur_obj(current_objective) and 'bg' in test_token_to_color.keys():
            del test_token_to_color['bg']

        if len(test_token_to_color) > len(current_objective):
            diff = list(set(test_token_to_color.keys()) - get_bare_assignment_leads(assignments_leads))
            for n in diff:
                if self.test_token_to_color[n] in list(itertools.chain.from_iterable(self.testinputs_vals)):
                    this_extra = expand_obj(deepcopy(current_objective[0]), n)
                    current_objective.append(this_extra)
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

    # this method direct transform without reference to coordinates
    def apply_direct_transformation(self, change_tuple, cur_test_pred):
        for x in range(len(cur_test_pred)):
            for n in range(cur_test_pred[x].shape[0]):
                for m in range(cur_test_pred[x].shape[1]):
                    cur = cur_test_pred[x][n][m]
                    if change_tuple[0] in self.test_token_to_color.keys() and change_tuple[1] in self.test_token_to_color.keys() and cur == self.test_token_to_color[change_tuple[0]]:
                        cur_test_pred[x][n][m] = self.test_token_to_color[change_tuple[1]]
        return cur_test_pred

    # Creating and populating output in cases with no objective:
    def build_a_prediction(self, dimension_status, reference, **kwargs):
        if dimension_status == 'deduced':
            for k, v in kwargs.items():
                if k == 'add_to_zero':
                    cur_output = np.zeros(reference) # reference is dimensions
                    cur_output += v
                    cur_output = cur_output.astype(int)
                    return [y.tolist() for y in cur_output]
                elif k == 'direct_transform':
                    cur_output = reference # reference is a testinput array
                    this_objective_dict = create_an_objective_dict(self.objective)
                    for token, coords in v.items():
                        for coord in coords:
                            cur_output[coord[0], coord[1]] = self.test_token_to_color[this_objective_dict[token]]
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
