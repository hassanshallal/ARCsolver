from utils import *
from communicate import *

class DimensionWork:
    def __init__(self, communication, cur_test_preds):
        # This is a holistic induction class, only takes communication and cur_test_preds
        self.communication = communication
        self.cur_train_preds = self.communication.cur_train_preds
        self.cur_test_preds = self.communication.cur_test_preds

        # data members to understand dimensions situation
        self.traininput_shapes = [list(x.shape) for x in self.cur_train_preds]
        self.trainoutputs_shapes = [list(x.shape) for x in self.communication.trainoutputs]
        self.testinput_shapes = [list(x.shape) for x in self.cur_test_preds]

        self.is_sim_in_shapes = all([x.shape == self.cur_train_preds[0].shape for x in self.cur_train_preds])
        self.is_sim_out_shapes = all([x.shape == self.communication.trainoutputs[0].shape for x in self.communication.trainoutputs])

        self.is_same_ndim_couple = all([len(x) == len(y) for x, y in zip(self.traininput_shapes, self.trainoutputs_shapes)])
        self.is_same_dim_couple = all([x == y for x, y in zip(self.traininput_shapes, self.trainoutputs_shapes)])
        self.what_relation = [list_comparator(x, y) for x, y in zip(self.traininput_shapes, self.trainoutputs_shapes)]
        self.same_relation = all([x == self.what_relation[0] for x in self.what_relation]) and self.what_relation[0] != None
        self.unidirctional = all([list_modulo(x, y) for x, y in zip(self.traininput_shapes, self.trainoutputs_shapes)])
        self.in_mult_fact = [get_int_div(x, y) for x, y in zip(self.traininput_shapes, self.trainoutputs_shapes)]
        self.is_sim_int_div = all([x == self.in_mult_fact[0] for x in self.in_mult_fact])
        self.is_sim_internal_int_div = all([x[0] == x[1] for x in self.in_mult_fact])

        self.is_one_row_output = all([x[0] == 1 and x[1] >= 1 for x in self.trainoutputs_shapes])
        self.row_dim_ouputs = [x[0] for x in self.trainoutputs_shapes]
        self.is_one_column_output = all([x[1] == 1 and x[0] >= 1 for x in self.trainoutputs_shapes])
        self.col_dim_ouputs = [x[1] for x in self.trainoutputs_shapes]

        
        self.in_objects = [extract_objects(find_objects(x)) for x in self.cur_train_preds]
        self.out_objects = [extract_objects(find_objects(x)) for x in self.communication.trainoutputs]
        self.cond1 = all([len(x) == 1 for x in self.out_objects])
        self.cond2 = all([len(self.out_objects[n]) > 0 and self.out_objects[n][0][0] == 0 and self.out_objects[n][0][0] == self.out_objects[n][0][2] for n in range(len(self.out_objects))])
        self.cond3 = all([len(self.out_objects[n]) > 0 and is_out_obj_in_in_objs(self.out_objects[n][0], self.in_objects[n]) for n in range(len(self.out_objects))])
        self.cond4 = all([x == [[0, 1, 0, 1]] or x == [] for x in self.out_objects])
        self.cond5 = all([len(x) == len(y) for x, y in zip(self.in_objects, self.out_objects)])
        self.cond6 = all([direct_obj_movement_detection(x, y) for x, y in zip(self.in_objects, self.out_objects)])
        self.cond7 = all([one_obj_move(x, y) for x, y in zip(self.in_objects, self.out_objects)])
        self.cond8 = all([len(x) < len(y) for x, y in zip(self.in_objects, self.out_objects)])
        self.cond9 = all([len(x) == 0 for x in self.in_objects])
        self.cond10 = all([sorted(x) == sorted(y) for x, y in zip(self.in_objects, self.out_objects)])


        self.dimension_status, self.output_dim_preds = self.cognify_dimensions()
        # print(self.traininput_shapes, self.trainoutputs_shapes, self.is_sim_in_shapes, self.is_sim_out_shapes, self.is_same_ndim_couple, self.is_same_dim_couple, self.what_relation, self.same_relation, self.unidirctional, self.in_mult_fact, self.is_sim_int_div, self.is_sim_internal_int_div, self.is_one_row_output, self.row_dim_ouputs, self.is_one_column_output, self.col_dim_ouputs)
    def match_row_or_col(self, iuo):
        fg = self.communication.get_frequency_graph(self.communication.frequency_counter_traininputs)
        if all([x in y for x, y in zip(iuo, fg)]):
            indices = [y.index(x) for x, y in zip(iuo, fg)]
            if all([x == indices[0] for x in indices]): # we don't need this condition here for dimension and indices[0] % 2 == 0
                return True, indices[0]
            else:
                return False, None
        return False, None

    def cognify_dimensions(self):
        if self.is_same_ndim_couple and self.is_same_dim_couple:
            return 'deduced', [tuple(x.shape) for x in self.cur_test_preds]

        elif self.is_same_ndim_couple and not self.is_same_dim_couple:
            if self.same_relation and (self.what_relation[0] != '==' and self.what_relation[0] != None):
                if self.unidirctional and self.is_sim_int_div:
                    return 'deduced', [tuple(x) for x in modify_dimensiosn(self.testinput_shapes, self.in_mult_fact[0])]
                elif self.unidirctional and not self.is_sim_int_div:
                    if self.is_sim_out_shapes:
                        return 'deduced', [tuple(self.trainoutputs_shapes[0])] * len(self.cur_test_preds)
                    else:
                        if self.is_sim_internal_int_div:
                            return 'partially deduced', [tuple(x) for x in self.testinput_shapes]

            if self.is_one_row_output:
                m, n = self.match_row_or_col(self.col_dim_ouputs)
                if m:
                    test_fg = self.communication.get_frequency_graph(self.communication.frequency_counter_testinputs)
                    return 'deduced', [(1, x[n]) for x in test_fg]
            if self.is_one_column_output:
                m, n = self.match_row_or_col(self.row_dim_ouputs)
                if m:
                    test_fg = self.communication.get_frequency_graph(self.communication.frequency_counter_testinputs)
                    return 'deduced', [(x[n], 1) for x in test_fg]

            if self.is_one_row_output and self.col_dim_ouputs == self.communication.freq_nonbg_traininputs:
                return 'deduced', [(1, x) for x in self.communication.freq_nonbg_testinputs]
            if self.is_one_column_output and self.row_dim_ouputs == self.communication.freq_nonbg_traininputs:
                return 'deduced', [(x, 1) for x in self.communication.freq_nonbg_testinputs]
        elif self.cond1 and self.cond2 and self.cond3:
            # output an object [30, 35, 48, 87]: one out object and it is one of the in objects:
            test_in_objects = [extract_objects(ndimage.find_objects(x)) for x in self.cur_test_preds]
            return 'object_deduced', test_in_objects

        return 'undeduced', [tuple(x) for x in self.testinput_shapes]

    def get_dimension_cognified(self):
        return self.dimension_status, self.output_dim_preds

    def assess_num_unique_nonbg(self, test_list):
        traininputs_vals = [np.unique(n).tolist() for n in self.cur_train_preds]
        testinputs_vals = [np.unique(n).tolist() for n in self.cur_test_preds]

        if type(self.communication.bg) == int:
            [y.remove(self.communication.bg) for y in traininputs_vals] # we need to remove bg here
            len_unique_traininputs_nonbg = [len(y) for y in traininputs_vals]
            if len_unique_traininputs_nonbg == test_list:
                [y.remove(self.communication.bg) for y in testinputs_vals]
                return True, [len(y) for y in testinputs_vals]
        return False, []

    def screen_dimesnions(self):
        if  self.dimension_status == 'undeduced':
            return 'unsolved', [], self.cur_test_preds
        elif self.is_same_ndim_couple and not self.is_same_dim_couple and self.same_relation and self.unidirctional:
            if self.is_sim_int_div:
                #if self.what_relation[0] == '>':
                if int(self.in_mult_fact[0][0]) != 0 and int(self.in_mult_fact[0][1]) != 0:
                    is_expanded = all([np.all(np.repeat(np.repeat(x, int(y[0]), 0), int(y[1]), 1) == z) for x, y, z in zip(self.cur_train_preds, self.in_mult_fact, self.communication.trainoutputs)])
                else:
                    is_expanded =  False

                #if self.what_relation[0] == '<':
                if int(1/self.in_mult_fact[0][0]) != 0 and int(1/self.in_mult_fact[0][1]) != 0:
                    is_contracted = all([np.all(x[::int(1/y[0]),::int(1/y[1])] == z) for x, y, z in zip(self.cur_train_preds, self.in_mult_fact, self.communication.trainoutputs)])
                else:
                    is_contracted = False

                if is_expanded or is_contracted:
                    if len(self.cur_test_preds) == 1:
                        this_in_mult_fact = [self.in_mult_fact[0]]
                    elif len(self.cur_test_preds) > 1:
                        this_in_mult_fact = self.in_mult_fact[0] * len(self.cur_test_preds)

                    if is_expanded:
                        cur_preds = [np.repeat(np.repeat(x, int(y[0]), 0), int(y[1]), 1) for x, y in zip(self.cur_test_preds, this_in_mult_fact)]
                    elif is_contracted:
                        cur_preds = [x[::int(1/y[0]),::int(1/y[1])] for x, y in zip(self.cur_test_preds, this_in_mult_fact)]

                    if len(self.communication.testoutputs) == len(self.communication.cur_test_preds):
                        test_evaluated = all([np.array_equal(x, y) for x, y in zip(cur_preds, self.communication.testoutputs)])
                    elif len(self.communication.testoutputs) == 0:
                        test_evaluated = False

                    if test_evaluated:
                        return 'solved', ['is_exp_cont', self.in_mult_fact[0]], cur_preds
                    else:
                        return 'objective initiated', ['is_exp_cont', self.in_mult_fact[0]], self.cur_test_preds

            elif not self.is_sim_out_shapes and self.is_sim_internal_int_div:
                candidate_factors = [x[0] for x in self.in_mult_fact]
                if all([int(x) != 0 for x in candidate_factors]):
                    is_expanded = all([np.all(np.repeat(np.repeat(x, int(y), 0), int(y), 1) == z) for x, y, z in zip(self.cur_train_preds, candidate_factors, self.communication.trainoutputs)])
                else:
                    is_expanded =  False
                if all([int(1/x) != 0 for x in candidate_factors]):
                    is_contracted = all([np.all(x[::int(1/y),::int(1/y)] == z) for x, y, z in zip(self.cur_train_preds, candidate_factors, self.communication.trainoutputs)])
                else:
                    is_contracted =  False
                if is_expanded or is_contracted:
                    r1, r2 = self.assess_num_unique_nonbg(candidate_factors)
                    if r1:
                        if is_expanded:
                            cur_preds = [np.repeat(np.repeat(x, y, 0), y, 1) for x, y in zip(self.cur_test_preds, [r2] * len(self.cur_test_preds))]
                        elif is_contracted:
                            cur_preds = [x[::int(1/y),::int(1/y)] for x, y in zip(self.cur_test_preds, [r2] * len(self.cur_test_preds))]
                        if len(self.communication.testoutputs) == len(self.communication.cur_test_preds):
                            test_evaluated = all([np.array_equal(x, y) for x, y in zip(cur_preds, self.communication.testoutputs)])
                        elif len(self.communication.testoutputs) == 0:
                            test_evaluated = False

                        if test_evaluated:
                            return 'solved', ['is_exp_cont_freq_unique_nonbg', candidate_factors], cur_preds
                        else:
                            return 'objective initiated', ['is_exp_cont_freq_unique_nonbg', candidate_factors], self.cur_test_preds

        return 'unsolved', [], self.cur_test_preds

    # current_situation = 'screening ' + what_to_try
    # current_situation = 'passed_all_traininputs'
    # current_situation = 'passed_all_testinputs'
    # current_situation = 'passed_some_testinputs'
    # current_situation = 'unpassed_all_testinputs'
    # current_situation = 'passed_some_traininputs'
    # current_situation = 'unpassed_all_traininputs'
    # current_situation, train_situation, train_screen, test_situation, test_screen
