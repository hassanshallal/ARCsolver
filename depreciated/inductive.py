from utils import *
from communicate import *

from dimensionWork import *
from captured import *
from neighbored import *
from cages import *

class Inductive:
    def __init__(self, communication):
        self.communication = communication

        # These are important and can be retrieved anytime
        self.running_objective = deepcopy(self.communication.test_objective)
        self.cur_test_preds = deepcopy(self.communication.testinputs)
        self.solved, self.mechanisms = 'unsolved', []

        # cognify and analyze dimensions using dimensionWork
        self.dimensionWork = DimensionWork(self.communication, self.cur_test_preds)
        self.dimension_status, self.output_dim_preds = self.dimensionWork.get_dimension_cognified()

        # We need to pass dimension_status, output_dim_preds to other modules
        self.communication.carry_along_kwargs(dimension_status = self.dimension_status, output_dim_preds = self.output_dim_preds)

        #self.cages = Cages(self.communication, self.running_objective)
        #self.captured = Captured(self.communication, self.running_objective, self.cur_test_preds)
        #self.neighbours = Neighbors(self.communication, self.running_objective, self.cur_test_preds)

    def inductive_strategy(self): # we will change this into a multilane highway and a find_path
    # routines on samples to decide whether to send a positive or a negative feedback so as to stop
        # try dimension related
        self.solved, mechanisms,  this_testpred = self.dimensionWork.screen_dimesnions()
        if self.solved == 'solved':
            self.mechanisms.append(mechanisms)
            self.cur_test_preds = this_testpred
            return
        # try flips related
        print("didn't catch screen_dimesnions")
        self.solved, mechanisms, this_testpred = self.dimensionWork.screen_flips()
        if self.solved == 'solved':
            self.mechanisms.append(mechanisms)
            self.cur_test_preds = this_testpred
            return
        print("didn't catch screen_flips")
        print('objective: ', self.communication.objective)
        print('objective_status: ', self.communication.objective_status)
        print('running_objective: ', self.running_objective)
        # go cages and be careful or othwrwise yo'll screw it up
        cages = Cages(self.communication)
        self.solved, mechanisms, this_testpred, self.running_objective = cages.screen_cages()
        if self.solved == 'solved':
            self.mechanisms.append(mechanisms)
            self.cur_test_preds = this_testpred
            return
        print("didn't catch screen_cages")
        # try captured related, we will pass and recieve a modified running objective or none
        captured = Captured(self.communication, self.cur_test_preds)
        self.solved, mechanisms,  this_test_pred, self.running_objective = captured.screen_captured()
        if self.solved == 'solved' or self.solved == 'partially solved':
            self.mechanisms.append(mechanisms)
            self.cur_test_preds = this_test_pred
            return
        print("didn't catch screen_captured")

        # try neighbored related, we will pass and recieve a modified running objective or none
        # neighbours = Neighbors(self.communication, self.running_objective, self.cur_test_preds)
        # self.solved, mechanisms,  this_test_pred, self.running_objective = neighbours.screen_neighbored()
        # if self.solved == 'solved' or self.solved == 'partially solved':
        #     self.mechanisms.append(mechanisms)
        #     self.cur_test_preds = this_test_pred
        #     return

        return

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
