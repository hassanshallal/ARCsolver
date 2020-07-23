from taskManager import *
from core_knowledge_utils import *

from communicate import *
from dimensionWork import *
from edges import *
from neighbors import *
from cages import *

class Inductive(TaskManager):
    def __init__(self, raw_task, forced_bg = None):
        super().__init__(raw_task, forced_bg = None)
        self.communication = Communication(self.cur_train_preds, self.trainoutputs, self.cur_test_preds, self.bg, self.objective_status, self.objective, self.assignments_leads, self.asssignments_output, self.token_to_colors, self.test_token_to_colors, self.testinputs_vals, self.test_objective, self.testoutputs)
        self.dimensionWork = DimensionWork(self.communication, self.cur_test_preds)
        self.dimension_status, self.output_dim_preds = self.dimensionWork.get_dimension_cognified()
        self.communication.carry_along_kwargs(output_dim_preds = self.output_dim_preds)
        self.communication.carry_along_kwargs(is_similar_dim = self.is_similar_dim)

        self.check_routine_list = [('check_value_maps', check_value_maps(self.couple_val_map))]
        self.apply_routine_list = [('apply_value_maps', apply_value_maps), ('apply_direct_transformation', apply_direct_transformation)]

        if self.is_similar_dim:
            self.cur_x_train, self.cur_y_train, self.cur_x_test = self.get_features()
            self.running_objective = [sets_obj_on_train(self.cur_x_train, self.cur_y_train, objec) for objec in self.objective]

        self.solved, self.mechanisms = 'unsolved', []
        self.current_situation, self.train_situation, self.train_screen, self.test_situation, self.test_screen = 'ineligible', [], [], [], []
        self.inductive_strategy()

    def try_apply_routines(self, what_to_try, apply_routine):
        current_situation = 'screening ' + what_to_try
        train_situation = deepcopy(self.cur_train_preds)
        train_screen = [False] * len(train_situation)
        test_situation = deepcopy(self.cur_test_preds)
        test_screen = ['nil'] * len(test_situation) # we always assume we don't have testoutputs and hence we can't compare
        color_to_tokens = deepcopy(self.color_to_tokens)
        objective = deepcopy(self.objective)
        token_to_colors = deepcopy(self.token_to_colors)

        test_color_to_tokens = deepcopy(self.test_color_to_tokens)
        test_objective = deepcopy(self.test_objective)
        test_token_to_colors = deepcopy(self.test_token_to_colors)

        if what_to_try == 'apply_value_maps':
            train_situation = [apply_routine(x, self.global_value_map) for x in train_situation]
        elif what_to_try == 'apply_direct_transformation':
            train_situation = [apply_routine(w, x, y, z) for w, x, y, z in zip(train_situation, color_to_tokens, objective, token_to_colors)]


        train_screen = [np.array_equal(x, y) for x, y in zip(train_situation, self.trainoutputs)]

        if all(train_screen):

            current_situation = 'passed_all_traininputs'
            if what_to_try == 'apply_value_maps':
                test_situation = [apply_routine(x, self.global_value_map) for x in test_situation]
            elif what_to_try == 'apply_direct_transformation':
                test_situation = [apply_routine(w, x, y, z) for w, x, y, z in zip(test_situation, test_color_to_tokens, test_objective, test_token_to_colors)]

            if len(self.testoutputs) > 0:
                test_screen = [np.array_equal(x, y) for x, y in zip(test_situation, self.testoutputs)]
                if all(test_screen):
                    current_situation = 'passed_all_testinputs'
                elif any(test_screen):
                    current_situation = 'passed_some_testinputs'
                else:
                    current_situation = 'unpassed_all_testinputs'

        elif any(train_screen):
            current_situation = 'passed_some_traininputs'

        else:
            current_situation = 'unpassed_all_traininputs'

        return current_situation, train_situation, train_screen, test_situation, test_screen

    def get_features(self):
        # First figure out the bg
        if self.global_bg:
            bg_train = [self.bg] * self.num_train
            bg_test = [self.bg] * self.num_test
        else:
            bg_train = self.bg
            bg_test = [get_background(x) for x in self.testinputs]

        # second gather prior knowledge
        train_inputs = [build_prior_knowledge(x, y, z) for x, y, z in zip(self.cur_train_preds, bg_train, self.color_to_tokens)]
        test_inputs = [build_prior_knowledge(x, y, z) for x, y, z in zip(self.cur_test_preds, bg_test, self.test_color_to_tokens)]
        train_outputs = [apply_transform_map(x, y) for x, y in zip(self.trainoutputs, self.color_to_tokens)]

        if all([x.shape[1] == y.shape[0] and x.shape[2] == y.shape[1] for x, y in zip(train_inputs, train_outputs)]):
            # Third: featurize
            x_train = []
            y_train = []
            for x, y in zip(train_inputs, train_outputs):
                features, target = featurize_prior_knowledge_train(x, y)
                x_train.append(features)
                y_train.append(target)
            x_test =  [featurize_prior_knowledge_test(x) for x in test_inputs]


            # Fourth: stack train cases
            cur_x_train = x_train[0]
            for n in range(1, len(x_train)):
                cur_x_train = np.vstack((cur_x_train, x_train[n]))

            # Fifth: find your task classes, codify them
            cur_y_train = y_train[0]
            for n in range(1, len(y_train)):
                cur_y_train = np.hstack((cur_y_train, y_train[n]))

            # Fourth: stack testcases
            cur_x_test = x_test[0]
            for n in range(1, len(x_test)):
                cur_x_test = np.vstack((cur_x_test, x_test[n]))

            return cur_x_train, cur_y_train, cur_x_test
        else:
            return train_inputs, train_outputs, test_inputs




    def inductive_strategy(self): # we will change this into a multilane highway and a find_path
    # routines on samples to decide whether to send a positive or a negative feedback so as to stop
        # try dimension related
        self.solved, mechanisms,  this_testpred = self.dimensionWork.screen_dimesnions()
        if self.solved == 'solved':
            self.current_situation = 'passed_all_testinputs'
            self.mechanisms.append(mechanisms)
            self.cur_test_preds = this_testpred
            self.test_situation = this_testpred
            return

        # try flips related
        self.current_situation, self.train_situation, self.train_screen, self.test_situation, self.test_screen, mechnsim = self.dimensionWork.screen_flips()
        if self.current_situation == 'passed_all_testinputs':
            self.solved = 'solved'
            self.mechanisms.append(mechnsim)
            self.cur_test_preds = self.test_situation
            return

        # go cages and be careful or othwrwise yo'll screw it up
        #print('attemting to have a cage instance')
        cages = Cages(self.communication)
        #print('cages:', attrs(cages))
        self.solved, mechanisms, this_testpred = cages.screen_cages(self.dimension_status)
        if self.solved == 'solved':
            self.current_situation = 'passed_all_testinputs'
            self.mechanisms.append(mechanisms)
            self.cur_test_preds = this_testpred
            self.test_situation = this_testpred
            return

        self.current_situation, self.train_situation, self.train_screen, self.test_situation, self.test_screen = self.screen_case()
        return

    def screen_case(self):
        # logic: this must be screen_a_thing which can be an array or an object in an array
        # We need a system that is aware here man
        # We can't use if conditional, the system must naturally process the task
        # and apply different strategies regardless of dimension, bg, etc
        # this logic is very narrow and doesn't serve our purposes

        if self.dimension_status =='deduced':
            if self.is_similar_dim:
                for check_routine in self.check_routine_list:
                    try_it, what_to_apply = check_routine[1]
                    if try_it:
                        indices = [i for i, tupl in enumerate(self.apply_routine_list) if tupl[0] == what_to_apply]
                        if len(indices) == 1:
                            indices = indices.pop()
                            current_situation, train_situation, train_screen, test_situation, test_screen = self.try_apply_routines(what_to_apply, self.apply_routine_list[indices][1])
                            if current_situation == 'passed_all_testinputs':
                                self.solved = 'solved'
                                self.mechanisms.append(self.apply_routine_list[indices][1])
                            return current_situation, train_situation, train_screen, test_situation, test_screen
                        else:
                            return 'unavailable_apply', [], [], [], []
                    else:
                        return 'ineligible', [], [], [], []
            else: # we need to find out
                return 'different_dimension', [], [], [], []
        else:
            return 'unknown_dimensions', [], [], [], []
