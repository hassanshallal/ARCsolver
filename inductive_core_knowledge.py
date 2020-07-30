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
        self.running_test_objective = self.validate_running_objective()

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
                    #print('is_opprtunity: ', is_opprtunity)
                    if any(is_opprtunity):
                        columns = [i+1 for i in range(len(is_opprtunity)) if is_opprtunity[i]]
                        #print('columns: ', columns)
                        obj.append(columns)
                        for opp in obj[2]:
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

# validate running_objective: complete: each sub_objective of each training case has found a working strategy
#                             consistent: there is a common mechanism for each sub_objective among all train retrieve_coords_from_assignments
#                             if complete and consistent:
#                                1) create a global running_objective (complete with consistently tackled objectives)
#                                2) massage test_objective using the global running_objective
# apply running_objective: input prior_knowledge, cur_preds, output: applied_cur_preds (for train or for test)

    def is_complete_sub_objective(self, sub_objective):
            return len(sub_objective) != 2

    # def trickle_down_lead_objective(lead_objective, other_objectives)
    # comnine with completeness and apply globally on obd, red, irr (abstraction baby abstaction)
    # basically it finds intersections of solution lists and confirms thier options_sets are similar
    # This is a level of consistency that should work for red and even obd objective_status cases which are no consistent
    # because of redundancy in the stable arm of the objective. This is abstraction in action, simple, quick, and clean.
    

    # intersection [6], [6] : solution lists
    # union({15} + {8, 15}) AND union({0}, {0})
    # loose arm has more options whereas tight arm has usually one option
    # two objectives with common signal may even differ in the component arms: 128 irr!

    # [[('bg', 'bg'), ('bg', 'nonbg0pr'), [6], (6, {15}, {0})]]
    # [[('bg', 'bg'), ('bg', 'nonbg0pr'), [6], (6, {8, 15}, {0})]]

    # [('nonbg0pr', 'nonbg0pr'), ('nonbg0pr', 'nonbg2pr'), [6, 7], (6, {15}, {0}), (7, {2}, {1})]
    # [('nonbg0pr', 'nonbg0pr'), ('nonbg0pr', 'nonbg2pr'), [6, 7], (6, {15}, {0}), (7, {2}, {1})]
    # [('nonbg0pr', 'nonbg0pr'), ('nonbg0pr', 'nonbg2pr'), [6], (6, {4, 15}, {0})]

    # [('nonbg3', 'nonbg3'), ('nonbg3', 'nonbg2pr'), [6], (6, {8, 3, 4, 15}, {0})]
    # [('nonbg3', 'nonbg3'), ('nonbg3', 'nonbg2pr'), [6], (6, {8, 3, 4, 15}, {0})]

    # [('bg', 'nonbg0pr'), ('bg', 'nonbg1pr'), [6], (6, {0}, {5, 15})]
    # [('bg', 'nonbg0pr'), ('bg', 'nonbg1pr'), [6], (6, {0}, {11, 12})]

    #  [('nonbg0', 'nil', 'direct')], [('nonbg1', 'nonbg1', 'direct'), [1, 4, 5], (1, {8}), (4, {14}), (5, {1})]
    #  [('nonbg0', 'nil', 'direct')], [('nonbg1', 'nonbg1', 'direct'), [1, 4, 5], (1, {7}), (4, {16}), (5, {1})]
    #  [('nonbg0', 'nil', 'direct')], [('nonbg1', 'nonbg1', 'direct'), [1, 4, 5, 6], (1, {4}), (4, {22}), (5, {1}), (6, {2})]

    # Notice the (5, {2}) that is common in all obectives of 128 which is irr
    # [[('nonbg-1pr', 'nonbg-1pr', 'direct'), [1, 4, 5, 7], (1, {4}), (4, {3}), (5, {2}), (7, {5})
    # [('nonbg-1pr', 'nonbg3', 'direct'), [1, 4, 5, 6], (1, {9}), (4, {3}), (5, {2}), (6, {8})]

    #  [[('bg', 'nil', 'direct')], [('nonbg0', 'nonbg0', 'direct'), [1, 4, 5], (1, {1}), (4, {2}), (5, {0})]]
    #  [[('bg', 'nil', 'direct')], [('nonbg0', 'nonbg0', 'direct'), [1, 4, 5, 6],  (1, {2}), (4, {3}), (5, {0}), (6, {15})]]
    
    def test_consistency(self):
        len_list = [len(x) for x in self.running_objective]
        lead_index = len_list.index(max(len_list))
        lead_objective = self.running_objective[lead_index]
        #print('lead_objective: ', lead_objective)
        consistency_measure = [None] * len(self.running_objective)
        consistency_measure[lead_index] = True
        #print(consistency_measure)
        for n in range(len(self.running_objective)):
            if n != lead_index:
                #print('test_objective: ', this_task.running_objective[n])
                this_case_consistency = [None] * len(self.running_objective[n])
                #print('this_case_consistency: ', this_case_consistency)
                for m in range(len(self.running_objective[n])):
                    if self.running_objective[n][m] not in lead_objective: # replace with a less stringent condition and modify your lead objective on the fly
                        this_case_consistency[m] = False
                    else:
                        this_case_consistency[m] = True
                    #print('this_case_consistency: ', this_case_consistency)
                consistency_measure[n] = all(this_case_consistency)
            #print(consistency_measure)
        return all(consistency_measure), lead_objective

    def validate_running_objective(self):
        if self.objective_status == 'obd' and self.dimension_status == 'deduced' and self.running_objective != self.objective:
            # test for completeness and for consistency of self.running_objective
            # In this case, we just want completeness to apply, in other cases, we'll need consistency in addition to completeness
            cur_lead = self.running_objective[0]
            is_complete = all([self.is_complete_sub_objective(x) for x in cur_lead])

            self.running_test_objective = [self.running_objective[0]] * len(self.cur_test_preds)
            return

        elif self.objective_status == 'red' and self.dimension_status == 'deduced' and self.running_objective != self.objective:
            # here we need all training cases to be completeness
            task_completeness = []
            for case in self.running_objective:
                task_completeness.append(all([self.is_complete_sub_objective(x) for x in case]))

            if all(task_completeness):
                # test consistency of similar objectives among different cases and this is done by
                # making sure their candidate solution lists have intersection and that the sets belonging
                # to one solution ar indeed either similar or have an intersection
                is_consistent, lead_objective = self.test_consistency()
                if is_consistent:
                    self.running_test_objective = [lead_objective] * len(self.cur_test_preds)
                    return
        self.running_test_objective = self.test_objective
        return

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
