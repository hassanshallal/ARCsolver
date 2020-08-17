from untokenized import *

class ARCsolver:

    def __init__(self, raw_task):
        self.solved, self.mechanisms = 'unsolved', []
        self.current_situation, self.train_situation, self.train_screen, self.test_situation, self.test_screen = 'ineligible', [], [], [], []
        self.applied_checkpoint_results = {}

        self.raw_task = raw_task
        self.num_train = len(raw_task['train'])
        self.num_test = len(raw_task['test'])
        self.traininputs, self.trainoutputs = get_training(raw_task)
        self.testinputs, self.testoutputs = get_testing(raw_task) # For tasks where there is no output, testoutputs is an empty list
        self.cur_train_preds, self.cur_test_preds = deepcopy(self.traininputs), deepcopy(self.testinputs)

        # check and apply for flips, here, you have all you need to do so.
        checkpoint0 = self.checkpoint(self.check_flips(self.cur_train_preds, self.trainoutputs))
        if checkpoint0 == 'halt':
            return

        # we get transformation_graph_list
        self.transformation_graph_list = [sorted(list(get_value_graphs(x,y))) for x, y in zip(self.traininputs, self.trainoutputs)]

        checkpoint1 = self.checkpoint(self.check_transforms(self.transformation_graph_list))
        if checkpoint1 == 'halt':
            return

        # prepare relevant info for your tests conditionally on presence of testoutputs
        self.freqs_traininputs = [get_sorted_frequency_situation(x) for x in self.traininputs]
        self.freqs_trainoutputs = [get_sorted_frequency_situation(x) for x in self.trainoutputs]
        self.freqs_testinputs = [get_sorted_frequency_situation(x) for x in self.testinputs]

        # assess bg and apply deductive routines to the task
        self.global_bg, self.traininputs_bg, self.trainoutputs_bg, self.testinputs_bg = self.assess_bg_situation()
        self.legitimate_bg_cutoff = 0.52
        self.is_bg_not_component = self.is_bg_or_comp()
        # dimensions initial processing
        self.traininputs_shapes = [list(x.shape) for x in self.traininputs]
        self.trainoutputs_shapes = [list(x.shape) for x in self.trainoutputs]
        self.testinputs_shapes = [list(x.shape) for x in self.testinputs]
        # we have the main traget is the testoutput, dimesnion and frequency of test components are subtargets
        if len(self.testoutputs) > 0:
            self.testoutputs_shapes = [list(x.shape) for x in self.testoutputs]
        else:
            self.testoutputs_shapes = []

        # overaall num of cells
        self.num_cells_traininputs = [x[0] * x[1] for x in self.traininputs_shapes]
        self.num_cells_trainoutputs = [x[0] * x[1] for x in self.trainoutputs_shapes]

        # dimesnion relationship extraction
        self.is_same_dim_couple = all([x == y for x, y in zip(self.traininputs_shapes, self.trainoutputs_shapes)])
        self.is_sim_out_shapes = all([x.shape == self.trainoutputs[0].shape for x in self.trainoutputs])
        self.int_div = [get_int_div(x, y) for x, y in zip(self.traininputs_shapes, self.trainoutputs_shapes)]
        self.is_sim_int_div = all([x == self.int_div[0] for x in self.int_div])
        self.is_sim_internal_int_div = all([x[0] == x[1] for x in self.int_div])

        self.what_relation = [list_comparator(x, y) for x, y in zip(self.traininputs_shapes, self.trainoutputs_shapes)]

        # logical Factors
        self.unique_train_outputs = [np.unique(x).tolist() for x in self.trainoutputs]
        self.is_unique_train_outputs = all([len(x) == 1 for x in self.unique_train_outputs])
        # self.bg_in_unique_train_outputs = self.bg in self.unique_train_outputs

        self.row_dim_ouputs = [x[0] for x in self.trainoutputs_shapes]
        self.col_dim_ouputs = [x[1] for x in self.trainoutputs_shapes]
        self.is_one_row = any([x == 1 for x in self.row_dim_ouputs])
        self.is_one_col = any([x == 1 for x in self.col_dim_ouputs])
        self.is_one_row_or_column = self.is_one_row or self.is_one_col

        # objectness
        self.traininput_objects = sort_objects_dims([extract_color_continious(x, y, self.global_bg, self.is_bg_not_component) for x, y in zip(self.traininputs, self.traininputs_bg)])
        self.testinput_objects = sort_objects_dims([extract_color_continious(x, y, self.global_bg, self.is_bg_not_component) for x, y in zip(self.testinputs, self.testinputs_bg)])
        self.trainoutput_objects = sort_objects_dims([extract_color_continious(x, y, self.global_bg, self.is_bg_not_component) for x, y in zip(self.trainoutputs, self.trainoutputs_bg)])

        self.traininput_spatial_objects = [extract_spatial_continious(x) for x in deepcopy(self.traininput_objects)]
        self.testinput_spatial_objects = [extract_spatial_continious(x) for x in deepcopy(self.testinput_objects)]
        self.trainoutput_spatial_objects = [extract_spatial_continious(x) for x in deepcopy(self.trainoutput_objects)]

        self.traininput_objects_dims = get_dims_objects(self.traininput_objects)
        self.testinput_objects_dims = get_dims_objects(self.testinput_objects)

        self.trainoutput_dims_in_traininputs_objects = all([x in y for x, y in zip(self.trainoutputs_shapes, self.traininput_objects_dims)])

        # we start predicting dimensions
        self.dimension_status, self.train_output_dim_preds, self.test_output_dim_preds = self.cognify_dimensions()

        if self.dimension_status != 'deduced':
            return

        checkpoint2 = self.checkpoint(self.check_expansions_contractions())
        if checkpoint2 == 'halt':
            return


        # tokenize, collect transformation and tokenized_transformation graphs (there are decisions made on the fly here)
        self.col_to_token_inputs = [get_freq_dict_in_(x[0], y) for x, y in zip(self.freqs_traininputs, self.traininputs_bg)]
        self.col_to_token_testinputs = [get_freq_dict_in_(x[0], y) for x, y in zip(self.freqs_testinputs, self.testinputs_bg)]
        self.col_to_token_inputs, self.col_to_token_testinputs= self.massage_input_dicts()
        self.col_to_token_outputs = [get_freq_dict_out_(x[0], y) for x, y in zip(self.freqs_trainoutputs, self.col_to_token_inputs)]
        self.col_to_token_inputs = [supplement_dict(x, y) for x, y in zip(self.col_to_token_inputs, self.col_to_token_outputs)]


        self.token_to_col_inputs = [reverse_dict(x) for x in self.col_to_token_inputs]
        self.token_to_col_outputs = [reverse_dict(x) for x in self.col_to_token_outputs]

        self.token_to_col_inputs_keys = [sorted(list(x.keys())) for x in self.token_to_col_inputs]
        self.token_to_col_outputs_keys = [sorted(list(x.keys())) for x in self.token_to_col_outputs]

        self.token_to_col_inputs_keys_length = [len(x) for x in self.token_to_col_inputs_keys]
        self.token_to_col_outputs_keys_length = [len(x) for x in self.token_to_col_outputs_keys]

        self.max_input_dict_ind = self.token_to_col_inputs_keys_length.index(max(self.token_to_col_inputs_keys_length))
        self.max_output_dict_ind = self.token_to_col_outputs_keys_length.index(max(self.token_to_col_outputs_keys_length))
        self.col_to_token_testinputs = [supplement_dict(x, y) for x, y in zip(self.col_to_token_testinputs, [self.col_to_token_outputs[self.max_output_dict_ind]] * self.num_test)]
        self.token_to_col_testinputs = [reverse_dict(x) for x in self.col_to_token_testinputs]

        self.same_input_tokens = all([sorted(list(x.keys())) == sorted(list(self.token_to_col_inputs[0].keys())) for x in self.token_to_col_inputs])
        self.same_output_tokens = all([sorted(list(x.keys())) == sorted(list(self.token_to_col_outputs[0].keys())) for x in self.token_to_col_outputs])
        self.obd_input = all([set(x).issubset(set(self.token_to_col_inputs_keys[self.max_input_dict_ind])) for x in self.token_to_col_inputs_keys])
        self.obd_output = all([set(x).issubset(set(self.token_to_col_outputs_keys[self.max_output_dict_ind])) for x in self.token_to_col_outputs_keys])
        self.obd_input_len = all([len(x) == len(self.token_to_col_inputs_keys[0]) for x in self.token_to_col_inputs_keys])
        self.obd_output_len = all([len(x) == len(self.token_to_col_outputs_keys[0]) for x in self.token_to_col_outputs_keys])

        self.tokenized_transformation_graph_list = [sorted(tokenize_transformation_graph(x, y, z)) for x, y, z in zip(self.transformation_graph_list, self.col_to_token_inputs, self.col_to_token_outputs)]
        checkpoint3 = self.checkpoint(self.check_transforms(self.tokenized_transformation_graph_list))
        if checkpoint3 == 'halt':
            return

        # We need to get token_to_col and col_to_token for test_input
        self.cur_x_train, self.cur_x_test = self.get_prior_knowledge()

    def assess_bg_situation(self):
        traininputs_bg = [x[0][0] for x in self.freqs_traininputs]
        trainoutputs_bg = [x[0][0] for x in self.freqs_trainoutputs]

        is_same_couple_bg = all([x == y for x, y in zip(traininputs_bg, trainoutputs_bg)])

        testinputs_bg = [x[0][0] for x in self.freqs_testinputs]

        inputs_bg_set = set(traininputs_bg + testinputs_bg)
        outputs_bg_set = set(trainoutputs_bg)
        overall_set = inputs_bg_set.union(outputs_bg_set)

        if len(overall_set) == 1:
            bg = overall_set.pop()
            train = [bg] * self.num_train
            test = [bg] * self.num_test
            return True, train, train, test
        elif len(inputs_bg_set) == 1 and list(inputs_bg_set)[0] == 0 and len(outputs_bg_set) > 1:
            bg = inputs_bg_set.pop()
            train = [bg] * self.num_train
            test = [bg] * self.num_test
            return True, train, train, test
        elif len(inputs_bg_set) == 1 and list(inputs_bg_set)[0] != 0 and len(outputs_bg_set) == 1 and list(outputs_bg_set)[0] == 0:
            bg = outputs_bg_set.pop()
            train = [bg] * self.num_train
            test = [bg] * self.num_test
            return True, train, train, test
        elif len(inputs_bg_set) == 1 and list(inputs_bg_set)[0] == 0 and len(outputs_bg_set) == 1 and list(outputs_bg_set)[0] != 0:
            input_bg = inputs_bg_set.pop()
            output_bg = outputs_bg_set.pop()

            in_train = [input_bg] * self.num_train
            # out_train = [output_bg] * self.num_train
            in_test = [input_bg] * self.num_test
            return True, in_train, in_train, in_test
        else:
            overall_list = traininputs_bg + testinputs_bg
            assessment = get_sorted_frequency_situation(fix_dim(overall_list))
            if len(assessment[0]) > 1 and (assessment[0][0] == 0 or (assessment[0][0] != 0 and assessment[0][1] == 0 and assessment[1][0] == assessment[1][1])):
                bg = 0
                train = [bg] * self.num_train
                test = [bg] * self.num_test
                return True, train, train, test
            else:
                cond1 = all([0 in x[0] for x in self.freqs_traininputs])
                cond2 = all([0 in x[0] for x in self.freqs_testinputs])
                if cond1 and cond2:
                    bg = 0
                    train = [bg] * self.num_train
                    test = [bg] * self.num_test
                    return True, train, train, test
                else:
                    return False, traininputs_bg, traininputs_bg, testinputs_bg

    def is_bg_or_comp(self):
        all_counts = [x[1] for x in self.freqs_traininputs + self.freqs_testinputs]
        bg_percent = [x[0]/np.sum(x) for x in all_counts]
        is_legit = all([x > self.legitimate_bg_cutoff for x in bg_percent])
        return is_legit

    def cognify_dimensions(self):
        if self.is_same_dim_couple:
            train_dim_preds = self.traininputs_shapes
            test_dim_preds = self.testinputs_shapes
            if all([x == y for x, y in zip(self.trainoutputs_shapes, train_dim_preds)]):
                return 'deduced',  train_dim_preds, test_dim_preds # 262, leaving 142 for free play
        if self.is_sim_out_shapes:
            output_shape = self.trainoutputs_shapes[0]
            train_dim_preds = [output_shape] * self.num_train
            test_dim_preds = [output_shape] * self.num_test
            if all([x == y for x, y in zip(self.trainoutputs_shapes, train_dim_preds)]):
                return 'deduced',  train_dim_preds, test_dim_preds # 76
        if self.is_sim_int_div:
            train_dim_preds = modify_dimensiosn(self.traininputs_shapes, self.int_div[0])
            test_dim_preds = modify_dimensiosn(self.testinputs_shapes, self.int_div[0])
            if all([x == y for x, y in zip(self.trainoutputs_shapes, train_dim_preds)]):
                return 'deduced',  train_dim_preds, test_dim_preds # 7

        if self.trainoutput_dims_in_traininputs_objects:
            indices = [x.index(y) for x, y in zip(self.traininput_objects_dims, self.trainoutputs_shapes)]
            same_index = all(x == indices[0] for x in indices)

            from_max_index = [len(y) - x for x, y in zip(indices, self.traininput_objects_dims)]
            same_from_max_index = all(x == from_max_index[0] for x in from_max_index)
            if same_index:
                train_dim_preds = [x[indices[0]] for x in self.traininput_objects_dims]
                test_dim_preds = [x[indices[0]] for x in self.testinput_objects_dims]
                if all([x == y for x, y in zip(self.trainoutputs_shapes, train_dim_preds)]):
                    return 'deduced',  train_dim_preds, test_dim_preds # 9
            elif same_from_max_index:
                train_dim_preds = [x[len(x) - from_max_index[0]] for x in self.traininput_objects_dims]
                test_dim_preds = [x[len(x) - from_max_index[0]] for x in self.testinput_objects_dims]
                if all([x == y for x, y in zip(self.trainoutputs_shapes, train_dim_preds)]):
                    return 'deduced',  train_dim_preds, test_dim_preds # 1
            # else: 200:, 208  objects determined by two horizonal lines OR 4 corners!
            #
        # comp_shape, minimum_objects_inputs, maximum_objects_inputs: 8 cases
        comp_shape = screen_basic_math(self.traininputs_shapes, self.trainoutputs_shapes)
        if comp_shape[0]:
            train_dim_preds = apply_basic_math(comp_shape[1], comp_shape[2], self.traininputs_shapes)
            test_dim_preds = apply_basic_math(comp_shape[1], comp_shape[2], self.testinputs_shapes)
            if all([x == y for x, y in zip(self.trainoutputs_shapes, train_dim_preds)]):
                return 'deduced',  train_dim_preds, test_dim_preds

        if all([len(x) > 0 for x in self.traininput_objects_dims]):
            minimum_objects_inputs = [x[0] for x in self.traininput_objects_dims]
            comp_minimum_objects = screen_basic_math(minimum_objects_inputs, self.trainoutputs_shapes)
            if comp_minimum_objects[0]:
                test_minimum_objects_inputs = [x[0] for x in self.testinput_objects_dims]
                train_dim_preds = apply_basic_math(comp_minimum_objects[1], comp_minimum_objects[2], minimum_objects_inputs)
                test_dim_preds = apply_basic_math(comp_minimum_objects[1], comp_minimum_objects[2], test_minimum_objects_inputs)
                if all([x == y for x, y in zip(self.trainoutputs_shapes, train_dim_preds)]):
                    return 'deduced',  train_dim_preds, test_dim_preds

            maximum_objects_inputs = [x[len(x) - 1] for x in self.traininput_objects_dims]
            comp_maximum_objects = screen_basic_math(maximum_objects_inputs, self.trainoutputs_shapes)
            if comp_maximum_objects[0]:
                test_maximum_objects_inputs = [x[len(x) - 1] for x in self.testinput_objects_dims]
                train_dim_preds = apply_basic_math(comp_maximum_objects[1], comp_maximum_objects[2], maximum_objects_inputs)
                test_dim_preds = apply_basic_math(comp_maximum_objects[1], comp_maximum_objects[2], test_maximum_objects_inputs)
                if all([x == y for x, y in zip(self.trainoutputs_shapes, train_dim_preds)]):
                    return 'deduced',  train_dim_preds, test_dim_preds

        return 'undeduced', [], []

    def get_dimension_cognified(self):
        return self.dimension_status, self.train_output_dim_preds, self.test_output_dim_preds

    def massage_input_dicts(self):
        original = deepcopy(self.col_to_token_inputs + self.col_to_token_testinputs)
        common = set.intersection(*[set(x.keys()) for x in original])
        common_list = sorted(list(common))
        modified_dicts = []
        for original_dict in original:
            new_dict = {}
            extra_keys = set(original_dict.keys()) - common
            start = 0
            for n in common_list:
                if n == -1:
                    new_dict[n] = 'nil'
                else:
                    new_dict[n] = 'c' + str(start)
                    start += 1

            for n in extra_keys:
                if original_dict[n] not in new_dict.values():
                    new_dict[n] = original_dict[n]
                else:
                    new_dict[n] = 'c' + str(start)

            modified_dicts.append(new_dict)

        return modified_dicts[0:len(self.col_to_token_inputs)], modified_dicts[len(self.col_to_token_inputs):]

    def get_prior_knowledge(self):
        # First figure out the bg
        bg_train = self.traininputs_bg
        bg_test = self.testinputs_bg

        # second gather prior knowledge
        traininputs_prkn = [build_prior_knowledge_adv(w, x, y, z) for w, x, y, z in zip(self.cur_train_preds, self.col_to_token_inputs, self.freqs_traininputs, self.traininputs_bg)]
        testinputs_prkn = [build_prior_knowledge_adv(w, x, y, z) for w, x, y, z in zip(self.cur_test_preds, self.col_to_token_testinputs, self.freqs_testinputs, self.testinputs_bg)]

        return traininputs_prkn, testinputs_prkn

    def checkpoint(self, what_to_check):
        # check and apply for flips, here, you have all you need to do so.
        # print('a')
        to_try, what_to_try, pass_info = what_to_check
        #print(to_try, what_to_try, pass_info)
        if to_try:
            # print('c')
            current_situation, train_situation, train_screen, test_situation, test_screen = self.try_apply_routines(what_to_try, pass_info)
            self.applied_checkpoint_results[str(what_to_check)] = (current_situation, train_situation, train_screen, test_situation, test_screen)

            # print(current_situation)
            if current_situation == 'passed_all_testinputs':
                # print('d')
                self.solved = 'solved'
                self.mechanisms.append((what_to_try, pass_info))
                self.current_situation, self.train_situation, self.train_screen, self.test_situation, self.test_screen =  current_situation, train_situation, train_screen, test_situation, test_screen
                return 'halt'
            return 'continue'
        return  'continue'

    def try_apply_routines(self, apply_routine, pass_info): # apply routine must work on in_ and pass_info
        # print('e')
        # print('apply_routine:', apply_routine)
        # print('pass_info:', pass_info)
        current_situation = 'screening ' + str(apply_routine)
        train_situation = deepcopy(self.cur_train_preds)
        train_options = [(None, None)] * len(train_situation)
        train_screen = [False] * len(train_situation)
        test_situation = deepcopy(self.cur_test_preds)
        test_screen = ['nil'] * len(test_situation) # we always assume we don't have testoutputs and hence we can't compare

        # assess train_situation
        if apply_routine == self.apply_simple_tokenized_transforms:
            train_situation = [apply_routine(x, y, pass_info, z) for x, y, z in zip(train_situation, self.col_to_token_inputs, self.token_to_col_inputs)]
        else:
            train_situation = [apply_routine(x, pass_info) for x in train_situation]

        train_screen = [np.array_equal(x, y) for x, y in zip(train_situation, self.trainoutputs)]
        # print('f')
        if all(train_screen):
            current_situation = 'passed_all_traininputs'
            if apply_routine == self.apply_simple_tokenized_transforms:
                test_situation = [apply_routine(x, y, pass_info, z) for x, y, z in zip(test_situation, self.col_to_token_testinputs, self.token_to_col_testinputs)]
            else:
                test_situation = [apply_routine(x, pass_info) for x in test_situation]

            if len(self.testoutputs) > 0:
                test_screen = [np.array_equal(x, y) for x, y in zip(test_situation, self.testoutputs)]
                if all(test_screen):
                    current_situation = 'passed_all_testinputs'
                elif all(test_screen) == False and any(test_screen):
                    current_situation = 'passed_some_testinputs'
                else:
                    current_situation = 'unpassed_all_testinputs'
        elif all(train_screen) == False and any(train_screen):
            current_situation = 'passed_some_traininputs'

        else:
            current_situation = 'unpassed_all_traininputs'

        return current_situation, train_situation, train_screen, test_situation, test_screen

    def check_flips(self, cur_train_preds, trainoutput):
        to_pass = {}
        x, y = screen_flips_rotation(cur_train_preds[0], trainoutput[0])
        train_options = [screen_flips_rotation(x, y) for x, y in zip(cur_train_preds, trainoutput)]
        if all([x != (None, None) and x == train_options[0] for x in train_options]):
            to_pass['routine'] = train_options[0]
            return True, self.apply_flips, to_pass
        else:
            return False, '', to_pass

    def apply_flips(self, in_, pass_info, **kawrgs):
        if pass_info['routine'][1] == None:
            return pass_info['routine'][0](in_)
        elif pass_info['routine'][1] != None:
            return pass_info['routine'][0](in_, pass_info['routine'][1])

    def check_transforms(self, transformation_graph_list):
        to_pass = {}
        set_of_col_changes = set.union(*[set(x) for x in transformation_graph_list])
        if len(set([x[0] for x in set_of_col_changes])) == len(set_of_col_changes):
            for transform in set_of_col_changes:
                to_pass[transform[0]] = transform[1]

            if type(list(to_pass.keys())[0]) == int:
                return True, self.apply_simple_transforms, to_pass
            elif type(list(to_pass.keys())[0]) == str:
                return True, self.apply_simple_tokenized_transforms, to_pass
        else:
            return False, '', to_pass

    def apply_simple_transforms(self, in_, pass_info):
        return apply_transform_map(in_, pass_info)

    def apply_simple_tokenized_transforms(self, in_, col_to_token, pass_info, token_to_col):
        a = apply_transform_map(in_, col_to_token)
        b = apply_transform_map(a, pass_info)
        return apply_transform_map(b, token_to_col)

    def check_expansions_contractions(self, start = self.traininputs):
        to_pass = {}
        out_in_rel = [convolve_for_a_match(x, y, z) for x, y, z in zip(start, self.trainoutputs, self.what_relation)]
        is_out_in_rel = all([x[0] for x in out_in_rel])
        is_same = all([x[2] == out_in_rel[0][2] for x in out_in_rel])
        if is_out_in_rel and is_same or self.is_unique_train_outputs:
            to_pass['scenario'] = out_in_rel[0][2]
            to_pass['arrangements'] = [[j[0] for j in i[1]] for i in out_in_rel]
            to_pass['starts'] = [[j[1] for j in i[1]] for i in out_in_rel]
            to_pass['is_unique_train_outputs'] = self.is_unique_train_outputs =
            return True, self.apply_expansions_contractions, to_pass
        return False, '', to_pass

    def apply_expansions_contractions(self, in_, pass_info):

        pass
