# this is a class that wraps important information in a single object to be passed to anyother class
from utils import *

class Communication:
    def __init__(self, cur_train_preds, trainoutputs, cur_test_preds, bg, objective_status, objective, assignments_leads, asssignments_output, token_to_colors, test_token_to_color, testinputs_vals, test_objective, testoutputs = None):
        self.cur_train_preds = cur_train_preds
        self.trainoutputs = trainoutputs
        self.cur_test_preds = cur_test_preds
        self.bg = bg

        self.objective_status = objective_status
        self.objective = objective
        self.assignments_leads = assignments_leads
        self.asssignments_output = asssignments_output
        self.token_to_colors = token_to_colors
        self.test_token_to_color = test_token_to_color
        self.testinputs_vals = testinputs_vals
        self.test_objective = test_objective
        self.testoutputs = testoutputs

        # The following data members are added along the way and used by other modules
        self.is_unique_output = self.find_unique_output()
        self.freq_nonbg_traininputs = self.get_freq_nonbg_inputs(self.cur_train_preds) # just the number of nonbg: utilized only in dimensionWork
        self.freq_nonbg_testinputs = self.get_freq_nonbg_inputs(self.cur_test_preds) # just the number of nonbg: utilized only in dimensionWork
        self.frequency_counter_traininputs = [Counter(list(itertools.chain.from_iterable(x.tolist()))).most_common() for x in self.cur_train_preds]
        self.frequency_counter_testinputs = [Counter(list(itertools.chain.from_iterable(x.tolist()))).most_common() for x in self.cur_test_preds]


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
            if k == 'is_similar_dim':
                self.is_similar_dim = v

        return
