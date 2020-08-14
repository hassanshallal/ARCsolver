from untokenized import *

class ARCsolver:

    def __init__(self, raw_task):
        self.raw_task = raw_task
        self.num_train = len(raw_task['train'])
        self.num_test = len(raw_task['test'])
        self.traininputs, self.trainoutputs = get_training(raw_task)
        self.testinputs, self.testoutputs = get_testing(raw_task) # For tasks where there is no output, testoutputs is an empty list

        # prepare relevant info for your tests conditionally on presence of testoutputs
        self.freqs_traininputs = [get_sorted_frequency_situation(x) for x in self.traininputs]
        self.freqs_trainoutputs = [get_sorted_frequency_situation(x) for x in self.trainoutputs]
        self.freqs_testinputs = [get_sorted_frequency_situation(x) for x in self.testinputs]

        # assess bg and apply deductive routines to the task
        self.traininputs_bg = [x[0][0] for x in self.freqs_traininputs]
        self.trainoutputs_bg = [x[0][0] for x in self.freqs_trainoutputs]
        self.testinputs_bg = [x[0][0]  for x in self.freqs_testinputs]
        self.global_bg, self.bg = self.assess_bg_situation()

        # dimensions initial processing
        self.traininputs_shapes = [list(x.shape) for x in self.traininputs]
        self.trainoutputs_shapes = [list(x.shape) for x in self.trainoutputs]
        self.testinputs_shapes = [list(x.shape) for x in self.testinputs]

        # overaall num of cells
        self.num_cells_traininputs = [x[0] * x[1] for x in self.traininputs_shapes]
        self.num_cells_trainoutputs = [x[0] * x[1] for x in self.trainoutputs_shapes]

        # dimesnion relationship extraction
        self.is_same_dim_couple = all([x == y for x, y in zip(self.traininputs_shapes, self.trainoutputs_shapes)])
        self.is_sim_out_shapes = all([x.shape == self.trainoutputs[0].shape for x in self.trainoutputs])
        self.int_div = [get_int_div(x, y) for x, y in zip(self.traininputs_shapes, self.trainoutputs_shapes)]
        self.is_sim_int_div = all([x == self.int_div[0] for x in self.int_div])
        self.is_sim_internal_int_div = all([x[0] == x[1] for x in self.int_div])

        self.traininput_objects = sort_objects_dims([extract_color_continious(x, y) for x, y in zip(self.traininputs, self.traininputs_bg)])
        self.testinput_objects = sort_objects_dims([extract_color_continious(x,  y) for x, y in zip(self.testinputs, self.testinputs_bg)])
        self.trainoutput_objects = sort_objects_dims([extract_color_continious(x, y) for x, y in zip(self.trainoutputs, self.trainoutputs_bg)])

        self.traininput_spatial_objects = [extract_spatial_continious(x) for x in deepcopy(self.traininput_objects)]
        self.testinput_spatial_objects = [extract_spatial_continious(x) for x in deepcopy(self.testinput_objects)]
        self.trainoutput_spatial_objects = [extract_spatial_continious(x) for x in deepcopy(self.trainoutput_objects)]

        self.traininput_objects_dims = get_dims_objects(self.traininput_objects)
        self.testinput_objects_dims = get_dims_objects(self.testinput_objects)

        self.trainoutput_dims_in_traininputs_objects = all([x in y for x, y in zip(self.trainoutputs_shapes, self.traininput_objects_dims)])

        self.is_one_object_in_input = all([len(x) == 1 for x in self.traininput_objects])
        self.is_one_object_in_output = all([len(x) == 1 for x in self.trainoutput_objects])
        self.one_object_situation = self.is_one_object_in_input and self.is_one_object_in_output
        self.multiple_object_situation = not self.is_one_object_in_input and self.is_one_object_in_output

        self.what_relation = [list_comparator(x, y) for x, y in zip(self.traininputs_shapes, self.trainoutputs_shapes)]
        self.out_in_rel = [convolve_for_a_match(x, y, z) for x, y, z in zip(self.traininputs, self.trainoutputs, self.what_relation)]
        self.is_out_in_rel = all([x[0] for x in self.out_in_rel])

        # logical Factors
        self.unique_train_outputs = [np.unique(x).tolist() for x in self.trainoutputs]
        self.is_unique_train_outputs = all([len(x) == 1 for x in self.unique_train_outputs])
        self.bg_in_unique_train_outputs = self.bg in self.unique_train_outputs

        self.row_dim_ouputs = [x[0] for x in self.trainoutputs_shapes]
        self.col_dim_ouputs = [x[1] for x in self.trainoutputs_shapes]
        self.is_one_row = any([x == 1 for x in self.row_dim_ouputs])
        self.is_one_col = any([x == 1 for x in self.col_dim_ouputs])
        self.is_one_row_or_column = self.is_one_row or self.is_one_col
        #all([(x[0] == 1 or x[1] == 1) and ((x[0] == 1) != (x[1] == 1)) for x in self.trainoutputs_shapes])

        # we have the main traget is the testoutput, dimesnion and frequency of test components are subtargets
        if len(self.testoutputs) > 0:
            self.freqs_testoutputs = [get_sorted_frequency_situation(x) for x in self.testoutputs]
            self.testoutputs_shapes = [list(x.shape) for x in self.testoutputs]
        else:
            self.freqs_testoutputs, self.testoutputs_shapes = [], []

        # we start predicting dimensions
        self.dimension_status, self.train_output_dim_preds, self.test_output_dim_preds = self.cognify_dimensions()

    def assess_bg_situation(self):
        traininputs_bg_set = set(self.traininputs_bg)
        trainoutputs_bg_set = set(self.trainoutputs_bg)
        if 0 in traininputs_bg_set and 0 in trainoutputs_bg_set:
            global_bg = True
            bg = 0
        elif len(traininputs_bg_set) == 1:
            global_bg = True
            bg = traininputs_bg_set.pop()
            bg = int(bg) #it is coming as numpy.int64 not int
        else:
            global_bg = False
            bg = self.traininputs_bg
        return global_bg, bg

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
