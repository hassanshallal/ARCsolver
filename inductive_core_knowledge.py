from utils import *
from dimensionWork import *


class Inductive:
    def __init__(self, communication):
        self.communication = communication

        # These are important and can be retrieved anytime
        self.running_objective = deepcopy(self.communication.test_objective)
        self.cur_train_preds = deepcopy(self.communication.traininputs)
        self.cur_test_preds = deepcopy(self.communication.testinputs)
        self.solved, self.mechanisms = 'unsolved', []

        self.dimensionWork = DimensionWork(self.communication, self.cur_test_preds)
        self.dimension_status, self.output_dim_preds = self.dimensionWork.get_dimension_cognified()


        
