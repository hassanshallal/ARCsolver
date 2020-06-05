# should be able to get started with this very soon. may be or may be not.

from utils import *
from communicate import *

from dimensionWork import *
from captured import *


class Inductive:
    def __init__(self, communication):
        self.communication = communication

        # These are important and can be retrieved anytime
        self.running_objective = deepcopy(self.communication.objective)
        self.cur_test_preds = deepcopy(self.communication.testinputs)
        self.solved, self.mechanisms = 'unsolved', []
        # cognify and analyze dimensions using dimensionWork
        self.dimensionWork = DimensionWork(self.communication, self.cur_test_preds)
        self.captured = Captured(self.communication, self.running_objective, self.cur_test_preds)


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
        self.solved, self.mechanisms,  this_test_pred, self.running_objective = self.captured.screen_captured()
        if self.solved == 'solved' or self.solved == 'partially solved':
            self.cur_test_preds = this_test_pred
            return

        return
