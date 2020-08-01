# minimal at this point, test github commit
from taskManager import *
from core_knowledge_utils import *

from fragmented_core_knowledge import *

from edges import *
from neighbors import *


class Inductive(TaskManager):
    def __init__(self, raw_task, forced_bg = None):
        super().__init__(raw_task, forced_bg = None)

        self.fragmented_core_knowledge = Fragmented(self.cur_train_preds, self.trainoutputs, self.cur_test_preds, self.bg, self.testoutputs)
        # self.dimensionWork = DimensionWork(self.communication, self.cur_test_preds)
        self.dimension_status, self.train_output_dim_preds, self.test_output_dim_preds = self.fragmented_core_knowledge.get_dimension_cognified()

        self.fragmented_core_knowledge.carry_along_kwargs(dimension_status = self.dimension_status)
        self.fragmented_core_knowledge.carry_along_kwargs(train_output_dim_preds = self.train_output_dim_preds)
        self.fragmented_core_knowledge.carry_along_kwargs(test_output_dim_preds = self.test_output_dim_preds)
        self.fragmented_core_knowledge.carry_along_kwargs(is_similar_dim = self.is_similar_dim)


        self.unique_train_outputs = [np.unique(x).tolist() for x in self.trainoutputs]
        # logical Factors
        self.is_unique_train_outputs = all([len(x) == 1 for x in self.unique_train_outputs])
        #print('is_unique_train_outputs:', self.is_unique_train_outputs)
        if self.is_unique_train_outputs:
            self.bg_in_unique_train_outputs = self.bg in self.unique_train_outputs
        else:
            self.bg_in_unique_train_outputs = None


        # get prior knowledge
        self.cur_x_train, self.cur_y_train, self.cur_x_test = self.get_features()
        self.running_objective = [self.sets_obj_on_train(x_train, y_train, objec) for x_train, y_train, objec in zip(self.cur_x_train, self.cur_y_train, self.objective)]
        self.is_complete_objective, self.global_objective = self.globalize_objective()

        self.check_routine_list = [('check_value_maps', check_value_maps(self.couple_val_map)), ('check_flips', check_flips(self.cur_train_preds, self.trainoutputs)), ('check_unique_output', check_unique_output(self.cur_train_preds, self.trainoutputs, self.bg_in_unique_train_outputs, self.bg))]
        self.apply_routine_list = [('apply_value_maps', apply_value_maps, 'ineligible'), ('apply_direct_transformation', apply_direct_transformation, 'ineligible'), ('apply_flips', screen_flips_rotation, 'ineligible'), ('apply_unique_output_frequency', apply_unique_output_frequency, 'ineligible')]

        self.solved, self.mechanisms = 'unsolved', []
        self.current_situation, self.train_situation, self.train_screen, self.test_situation, self.test_screen = 'ineligible', [], [], [], []
        self.inductive_strategy()

    def try_apply_routines(self, what_to_try, apply_routine, pass_info):
        current_situation = 'screening ' + what_to_try
        train_situation = deepcopy(self.cur_train_preds)
        train_options = [(None, None)] * len(train_situation)
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
            train_situation = [apply_routine(x, self.global_value_map, pass_info) for x in train_situation]
        elif what_to_try == 'apply_direct_transformation':
            train_situation = [apply_routine(w, x, y, z, pass_info) for w, x, y, z in zip(train_situation, color_to_tokens, objective, token_to_colors)]
        elif what_to_try == 'apply_flips':
            if pass_info['routine'][1] == None:
                routines = [pass_info['routine'][0]] * len(self.traininputs)
                train_situation = [x(y) for x, y in zip(routines, train_situation)]
            elif pass_info['routine'][1] != None:
                routines = [pass_info['routine']] * len(self.traininputs)
                train_situation = [x[0](y, x[1]) for x, y in zip(routines, train_situation)]
        elif what_to_try == 'apply_unique_output_frequency':
            train_situation = apply_unique_output_frequency(self.cur_train_preds, self.train_output_dim_preds, pass_info)

        train_screen = [np.array_equal(x, y) for x, y in zip(train_situation, self.trainoutputs)]

        if all(train_screen):
            current_situation = 'passed_all_traininputs'
            if what_to_try == 'apply_value_maps':
                test_situation = [apply_routine(x, self.global_value_map, pass_info) for x in test_situation]
            elif what_to_try == 'apply_direct_transformation':
                test_situation = [apply_routine(w, x, y, z, pass_info) for w, x, y, z in zip(test_situation, test_color_to_tokens, test_objective, test_token_to_colors)]
            elif what_to_try == 'apply_flips':
                if pass_info['routine'][1] == None:
                    routines = [pass_info['routine'][0]] * len(self.testinputs)
                    test_situation = [x(y) for x, y in zip(routines, test_situation)]
                elif pass_info['routine'][1] != None:
                    routines = [pass_info['routine']] * len(self.testinputs)
                    test_situation = [x[0](y, x[1]) for x, y in zip(routines, test_situation)]
            elif what_to_try == 'apply_unique_output_frequency':
                test_situation = apply_unique_output_frequency(self.cur_test_preds, self.test_output_dim_preds, pass_info)

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

    def get_prior_knowledge(self):
        # First figure out the bg
        if self.global_bg:
            bg_train = [self.bg] * self.num_train
            bg_test = [self.bg] * self.num_test
        else:
            bg_train = self.bg
            bg_test = [get_background(x) for x in self.testinputs]

        # second gather prior knowledge
        traininputs_prkn = [build_prior_knowledge(x, y, z) for x, y, z in zip(self.cur_train_preds, bg_train, self.color_to_tokens)]
        testinputs_prkn = [build_prior_knowledge(x, y, z) for x, y, z in zip(self.cur_test_preds, bg_test, self.test_color_to_tokens)]
        trainoutputs_tokenized = [apply_transform_map(x, y) for x, y in zip(self.trainoutputs, self.color_to_tokens)]
        return traininputs_prkn, trainoutputs_tokenized, testinputs_prkn


    def get_features(self):
        traininputs_prkn, trainoutputs_tokenized, testinputs_prkn  = self.get_prior_knowledge()
        return traininputs_prkn, trainoutputs_tokenized, testinputs_prkn

    def sets_obj_on_train(self, cur_x_train, cur_y_train, this_objective):
        objective = deepcopy(this_objective)
        #print('objective: ', objective)
        sets_dict = {}
        this_target = 'nil'
        if cur_y_train.shape[0] == cur_x_train.shape[1] and cur_y_train.shape[1] == cur_x_train.shape[2]:
            for obj in objective:
                if len(obj) == 2:
                    sets_dict[obj[0]] = [set() for index in range(1, cur_x_train.shape[0])]
                    sets_dict[obj[1]] = [set() for index in range(1, cur_x_train.shape[0])]
        if self.is_unique_train_outputs:
            for obj in objective:
                if len(obj) == 1 and len(obj[0]) == 3 and obj[0][1] != 'nil' and obj[0][2] == 'direct':
                    this_target = obj[0][1]
                    sets_dict[this_target] = [set() for index in range(1, cur_x_train.shape[0])]
                    sets_dict['other'] = [set() for index in range(1, cur_x_train.shape[0])]

        #print('sets_dict before:', sets_dict)
        if len(sets_dict) > 0:
            for x in range(cur_x_train.shape[1]):
                for y in range(cur_x_train.shape[2]):
                    for z in range(1, cur_x_train.shape[0]):
                        if this_target == 'nil':
                            first = cur_x_train[0][x][y]
                            second = cur_y_train[x][y]
                            if (first, second) in sets_dict.keys():
                                sets_dict[(first, second)][z-1].add(cur_x_train[z][x][y])
                        elif this_target != 'nil':
                            if cur_x_train[0][x][y] == this_target:
                                sets_dict[this_target][z-1].add(cur_x_train[z][x][y])
                            else:
                                sets_dict['other'][z-1].add(cur_x_train[z][x][y])

            #print('sets_dict after: ', sets_dict)
            # for cases where output has similar dimension to input
            for obj in objective:
                if len(obj) == 2:
                    is_opprtunity = [len(x.intersection(y)) == 0 for x, y in zip(sets_dict[obj[0]], sets_dict[obj[1]])]
                    
                    if any(is_opprtunity):
                        columns = [i+1 for i in range(len(is_opprtunity)) if is_opprtunity[i]]                       
                        if len(columns) > 0:
                            obj.append(columns)
                            for opp in columns:
                                obj.append((opp, sets_dict[obj[0]][opp-1], sets_dict[obj[1]][opp-1]))

            # for cases with unique output
            for obj in objective:
                if len(obj) == 1 and len(obj[0]) == 3 and obj[0][1] != 'nil' and obj[0][2] == 'direct':
                    if self.is_unique_train_outputs:
                        this_target = obj[0][1]
                        sets_dict['any_opp'] = [sets_dict[this_target][x] - sets_dict['other'][x] for x in range(len(sets_dict[this_target]))]
                        sets_dict['is_opp'] = [len(x) > 0 for x in sets_dict['any_opp']]
                        if any(sets_dict['is_opp']):
                            columns = [i+1 for i in range(len(sets_dict['is_opp'])) if sets_dict['is_opp'][i]]
                            obj.append(columns)
                            for opp in obj[1]:
                                obj.append((opp, sets_dict['any_opp'][opp-1]))
                                
        return objective

    
    def globalize_objective(self):
        # validate and compile on a train level
        len_list = [len(x) for x in self.running_objective]
        lead_index = len_list.index(max(len_list))

        lead_objective = self.running_objective[lead_index]
        for n in range(len(self.running_objective)):       
            if n != lead_index:
                current_sub = self.running_objective[n]
                for x in range(len(current_sub)):
                    for y in range(len(lead_objective)):
                        lead_objective[y] = merge_sub_objectives(current_sub[x], lead_objective[y])
                        
        # Use the information in the test so as to further prepare the global_objective
        lead_objective = enforce_global_objective(self.test_objective, lead_objective)
        return all([is_complete_sub_objective(x) for x in lead_objective]), lead_objective

    def inductive_strategy(self): # we will change this into a multilane highway and a find_path
    # routines on samples to decide whether to send a positive or a negative feedback so as to stop
        # try dimension related
        self.solved, mechanisms,  this_testpred = self.fragmented_core_knowledge.screen_dimesnions()
        if self.solved == 'solved':
            self.current_situation = 'passed_all_testinputs'
            self.mechanisms.append(mechanisms)
            self.cur_test_preds = this_testpred
            self.test_situation = this_testpred
            return

        self.current_situation, self.train_situation, self.train_screen, self.test_situation, self.test_screen = self.screen_case()
        return

    def screen_case(self):
        if self.dimension_status =='deduced':
            for check_routine in self.check_routine_list:
                try_it, what_to_apply, pass_info = check_routine[1]
                if try_it:
                    indices = [i for i, tupl in enumerate(self.apply_routine_list) if tupl[0] == what_to_apply]
                    if len(indices) == 1:
                        indices = indices.pop()
                        current_situation, train_situation, train_screen, test_situation, test_screen = self.try_apply_routines(what_to_apply, self.apply_routine_list[indices][1], pass_info)
                        if current_situation in ['passed_all_traininputs', 'passed_some_traininputs', 'passed_all_testinputs', 'passed_some_testinputs']:
                            self.apply_routine_list[indices] = (self.apply_routine_list[indices][0], self.apply_routine_list[indices][1], current_situation)
                        if current_situation == 'passed_all_testinputs':
                            self.solved = 'solved'
                            self.mechanisms.append(self.apply_routine_list[indices][1])
                            return current_situation, train_situation, train_screen, test_situation, test_screen
            return 'ineligible', [], [], [], []
        else:
            return 'ineligible', [], [], [], []
