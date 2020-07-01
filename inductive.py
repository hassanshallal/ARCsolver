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
        self.running_objective = deepcopy(self.communication.objective)
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
        self.solved, mechanisms, this_testpred = self.dimensionWork.screen_flips()
        if self.solved == 'solved':
            self.mechanisms.append(mechanisms)
            self.cur_test_preds = this_testpred
            return

        # go cages and be careful or othwrwise yo'll screw it up
        cages = Cages(self.communication)
        self.solved, mechanisms, this_testpred, self.running_objective = cages.screen_cages()
        if self.solved == 'solved':
            self.mechanisms.append(mechanisms)
            self.cur_test_preds = this_testpred
            return

        # try captured related, we will pass and recieve a modified running objective or none
        captured = Captured(self.communication, self.cur_test_preds)
        self.solved, mechanisms,  this_test_pred, self.running_objective = captured.screen_captured()
        if self.solved == 'solved' or self.solved == 'partially solved':
            self.mechanisms.append(mechanisms)
            self.cur_test_preds = this_test_pred
            return

        # try neighbored related, we will pass and recieve a modified running objective or none
        # neighbours = Neighbors(self.communication, self.running_objective, self.cur_test_preds)
        # self.solved, mechanisms,  this_test_pred, self.running_objective = neighbours.screen_neighbored()
        # if self.solved == 'solved' or self.solved == 'partially solved':
        #     self.mechanisms.append(mechanisms)
        #     self.cur_test_preds = this_test_pred
        #     return

        return
