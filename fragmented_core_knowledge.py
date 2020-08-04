from core_knowledge_utils import *

class Fragmented:
    def __init__(self, cur_train_preds, trainoutputs, cur_test_preds, bg, testoutputs = None):
        self.cur_train_preds = cur_train_preds
        self.trainoutputs = trainoutputs
        self.cur_test_preds = cur_test_preds
        self.bg = bg
        self.testoutputs = testoutputs


        # data members to understand dimensions situation
        self.freq_nonbg_traininputs = self.get_freq_nonbg_inputs(self.cur_train_preds) # just the number of nonbg: utilized only in dimensionWork
        self.freq_nonbg_testinputs = self.get_freq_nonbg_inputs(self.cur_test_preds) # just the number of nonbg: utilized only in dimensionWork

        self.traininput_shapes = [list(x.shape) for x in self.cur_train_preds]
        self.trainoutputs_shapes = [list(x.shape) for x in self.trainoutputs]
        self.testinput_shapes = [list(x.shape) for x in self.cur_test_preds]
        self.row_dim_ouputs = [x[0] for x in self.trainoutputs_shapes]
        self.col_dim_ouputs = [x[1] for x in self.trainoutputs_shapes]

        self.what_relation = [list_comparator(x, y) for x, y in zip(self.traininput_shapes, self.trainoutputs_shapes)]
        self.int_div = [get_int_div(x, y) for x, y in zip(self.traininput_shapes, self.trainoutputs_shapes)]

        self.traininput_objects = [extract_objects(x) for x in self.cur_train_preds]
        self.trainoutput_objects = [extract_objects(x) for x in self.trainoutputs]
        self.testinput_objects = [extract_objects(x) for x in self.cur_test_preds]

        self.unique_train_outputs = [np.unique(x).tolist() for x in self.trainoutputs]

        # logical Factors
        self.is_unique_train_outputs = all([len(x) == 1 for x in self.unique_train_outputs])
        if self.is_unique_train_outputs:
            self.bg_in_unique_train_outputs = self.bg in self.unique_train_outputs
        else:
            self.bg_in_unique_train_outputs = None

        self.is_sim_in_shapes = all([x.shape == self.cur_train_preds[0].shape for x in self.cur_train_preds])
        self.is_sim_out_shapes = all([x.shape == self.trainoutputs[0].shape for x in self.trainoutputs])
        self.is_same_ndim_couple = all([len(x) == len(y) for x, y in zip(self.traininput_shapes, self.trainoutputs_shapes)])
        self.is_same_dim_couple = all([x == y for x, y in zip(self.traininput_shapes, self.trainoutputs_shapes)])
        self.same_relation = all([x == self.what_relation[0] for x in self.what_relation]) and self.what_relation[0] != None
        self.unidirctional = all([list_modulo(x, y) for x, y in zip(self.traininput_shapes, self.trainoutputs_shapes)])
        self.is_sim_int_div = all([x == self.int_div[0] for x in self.int_div])
        self.is_sim_internal_int_div = all([x[0] == x[1] for x in self.int_div])
        self.is_one_row_output = all([x[0] == 1 and x[1] >= 1 for x in self.trainoutputs_shapes])
        self.is_one_column_output = all([x[1] == 1 and x[0] >= 1 for x in self.trainoutputs_shapes])
        self.cond1 = all([len(x) == 1 for x in self.trainoutput_objects])
        self.cond2 = all([len(self.trainoutput_objects[n]) > 0 and self.trainoutput_objects[n][0][0] == 0 and self.trainoutput_objects[n][0][0] == self.trainoutput_objects[n][0][2] for n in range(len(self.trainoutput_objects))])
        self.cond3 = all([len(self.trainoutput_objects[n]) > 0 and is_out_obj_in_in_objs(self.trainoutput_objects[n][0], self.traininput_objects[n]) for n in range(len(self.trainoutput_objects))])
        self.cond4 = all([x == [[0, 1, 0, 1]] or x == [] for x in self.trainoutput_objects])
        self.cond5 = all([len(x) == len(y) for x, y in zip(self.traininput_objects, self.trainoutput_objects)])
        self.cond6 = all([direct_obj_movement_detection(x, y) for x, y in zip(self.traininput_objects, self.trainoutput_objects)])
        self.cond7 = all([one_obj_move(x, y) for x, y in zip(self.traininput_objects, self.trainoutput_objects)])
        self.cond8 = all([len(x) < len(y) for x, y in zip(self.traininput_objects, self.trainoutput_objects)])
        self.cond9 = all([len(x) == 0 for x in self.traininput_objects])
        self.cond10 = all([sorted(x) == sorted(y) for x, y in zip(self.traininput_objects, self.trainoutput_objects)])
        self.dimension_status, self.train_output_dim_preds, self.test_output_dim_preds = self.cognify_dimensions()

    # Recieve cargo
    def carry_along_args(self, *args):
        self.args = args
        return

    def carry_along_kwargs(self, **kwargs):
        for k, v in kwargs.items():
            if k == 'dimension_status':
                self.dimension_status = v
            if k == 'train_output_dim_preds':
                self.train_output_dim_preds = v
            if k == 'test_output_dim_preds':
                self.test_output_dim_preds = v
            if k == 'is_similar_dim':
                self.is_similar_dim = v
        return

    def get_freq_nonbg_inputs(self, lists):
        if type(self.bg) == int:
            return [np.sum(x != self.bg) for x in lists]
        elif type(self.bg) == list:
            return [np.sum(x != y) for x, y in zip(lists, self.bg)]

    def match_row_or_col(self, iuo):
        fg = get_frequency_graph([Counter(list(itertools.chain.from_iterable(x.tolist()))).most_common() for x in self.cur_train_preds], self.bg_in_unique_train_outputs, self.bg)
        if all([x in y for x, y in zip(iuo, fg)]):
            indices = [y.index(x) for x, y in zip(iuo, fg)]
            if all([x == indices[0] for x in indices]): # we don't need this condition here for dimension and indices[0] % 2 == 0
                return True, indices[0]
            else:
                return False, None
        return False, None

    def cognify_dimensions(self):
        if self.is_same_ndim_couple and self.is_same_dim_couple:
            #print('d_c_1')
            return 'deduced', [tuple(x.shape) for x in self.cur_train_preds], [tuple(x.shape) for x in self.cur_test_preds]

        elif self.is_same_ndim_couple and not self.is_same_dim_couple:
            if self.same_relation and (self.what_relation[0] != '==' and self.what_relation[0] != None):
                if self.unidirctional and self.is_sim_int_div:
                    #print('d_c_2')
                    return 'deduced', [tuple(x) for x in modify_dimensiosn(self.traininput_shapes, self.int_div[0])], [tuple(x) for x in modify_dimensiosn(self.testinput_shapes, self.int_div[0])]
                elif self.unidirctional and not self.is_sim_int_div:
                    if self.is_sim_out_shapes:
                        #print('d_c_3')
                        return 'deduced', [tuple(self.trainoutputs_shapes[0])] * len(self.cur_train_preds) , [tuple(self.trainoutputs_shapes[0])] * len(self.cur_test_preds)
                    else:
                        if self.is_sim_internal_int_div:
                            #print('d_c_4')
                            return 'partially deduced', [tuple(x) for x in self.traininput_shapes] , [tuple(x) for x in self.testinput_shapes]

            if self.is_one_row_output:
                m, n = self.match_row_or_col(self.col_dim_ouputs)
                if m:
                    train_fg = get_frequency_graph([Counter(list(itertools.chain.from_iterable(x.tolist()))).most_common() for x in self.cur_train_preds], self.bg_in_unique_train_outputs, self.bg)
                    test_fg = get_frequency_graph([Counter(list(itertools.chain.from_iterable(x.tolist()))).most_common() for x in self.cur_test_preds], self.bg_in_unique_train_outputs, self.bg)
                    #print('d_c_5')
                    cur_train_prop_dim = []
                    for x in train_fg:
                        if len(x) > 1:
                            cur_train_prop_dim.append((1, x[n]))
                        elif len(x) == 1:
                            cur_train_prop_dim.append((1, x[0]))

                    cur_test_prop_dim = []
                    for x in test_fg:
                        if len(x) > 1:
                            cur_test_prop_dim.append((1, x[n]))
                        elif len(x) == 1:
                            cur_test_prop_dim.append((1, x[0]))

                    return 'deduced', cur_train_prop_dim, cur_test_prop_dim

            if self.is_one_column_output:
                m, n = self.match_row_or_col(self.row_dim_ouputs)
                if m:
                    train_fg = get_frequency_graph([Counter(list(itertools.chain.from_iterable(x.tolist()))).most_common() for x in self.cur_train_preds], self.bg_in_unique_train_outputs, self.bg)
                    test_fg = get_frequency_graph([Counter(list(itertools.chain.from_iterable(x.tolist()))).most_common() for x in self.cur_test_preds], self.bg_in_unique_train_outputs, self.bg)
                    #print('d_c_6')

                    cur_train_prop_dim = []
                    for x in train_fg:
                        if len(x) > 1:
                            cur_train_prop_dim.append((x[n], 1))
                        elif len(x) == 1:
                            cur_train_prop_dim.append((x[0], 1))

                    cur_test_prop_dim = []
                    for x in test_fg:
                        if len(x) > 1:
                            cur_test_prop_dim.append((x[n], 1))
                        elif len(x) == 1:
                            cur_test_prop_dim.append((x[0], 1))

                    return 'deduced', cur_train_prop_dim, cur_test_prop_dim


            if self.is_one_row_output and self.col_dim_ouputs == self.freq_nonbg_traininputs:
                #print('d_c_7')
                return 'deduced', [(1, x) for x in self.freq_nonbg_traininputs] , [(1, x) for x in self.freq_nonbg_testinputs]
            if self.is_one_column_output and self.row_dim_ouputs == self.freq_nonbg_traininputs:
                #print('d_c_8')
                return 'deduced', [(x, 1) for x in self.freq_nonbg_traininputs], [(x, 1) for x in self.freq_nonbg_testinputs]

        elif self.cond1 and self.cond2 and self.cond3:
            #print('d_c_9')
            # output an object [30, 35, 48, 87]: one out object and it is one of the in objects:
            train_traininput_objects = [extract_objects(x) for x in self.cur_train_preds]
            test_traininput_objects = [extract_objects(x) for x in self.cur_test_preds]
            return 'object_deduced', train_traininput_objects, test_traininput_objects
        #print('d_c_10')
        return 'undeduced', [tuple(x) for x in self.traininput_shapes], [tuple(x) for x in self.testinput_shapes]

    def get_dimension_cognified(self):
        return self.dimension_status, self.train_output_dim_preds, self.test_output_dim_preds

    def assess_num_unique_nonbg(self, test_list):
        traininputs_vals = [np.unique(n).tolist() for n in self.cur_train_preds]
        testinputs_vals = [np.unique(n).tolist() for n in self.cur_test_preds]

        if type(self.bg) == int:
            [y.remove(self.bg) for y in traininputs_vals] # we need to remove bg here
            len_unique_traininputs_nonbg = [len(y) for y in traininputs_vals]
            if len_unique_traininputs_nonbg == test_list:
                [y.remove(self.bg) for y in testinputs_vals]
                return True, [len(y) for y in testinputs_vals]
        return False, []

    def screen_dimesnions(self):
        if  self.dimension_status == 'undeduced':
            return 'unsolved', [], self.cur_test_preds
        elif self.is_same_ndim_couple and not self.is_same_dim_couple and self.same_relation and self.unidirctional:
            if self.is_sim_int_div:
                #if self.what_relation[0] == '>':
                if int(self.int_div[0][0]) != 0 and int(self.int_div[0][1]) != 0:
                    is_expanded = all([np.all(np.repeat(np.repeat(x, int(y[0]), 0), int(y[1]), 1) == z) for x, y, z in zip(self.cur_train_preds, self.int_div, self.trainoutputs)])
                else:
                    is_expanded =  False

                #if self.what_relation[0] == '<':
                if int(1/self.int_div[0][0]) != 0 and int(1/self.int_div[0][1]) != 0:
                    is_contracted = all([np.all(x[::int(1/y[0]),::int(1/y[1])] == z) for x, y, z in zip(self.cur_train_preds, self.int_div, self.trainoutputs)])
                else:
                    is_contracted = False

                if is_expanded or is_contracted:
                    if len(self.cur_test_preds) == 1:
                        this_int_div = [self.int_div[0]]
                    elif len(self.cur_test_preds) > 1:
                        this_int_div = self.int_div[0] * len(self.cur_test_preds)

                    if is_expanded:
                        cur_preds = [np.repeat(np.repeat(x, int(y[0]), 0), int(y[1]), 1) for x, y in zip(self.cur_test_preds, this_int_div)]
                    elif is_contracted:
                        cur_preds = [x[::int(1/y[0]),::int(1/y[1])] for x, y in zip(self.cur_test_preds, this_int_div)]

                    if len(self.testoutputs) == len(self.cur_test_preds):
                        test_evaluated = all([np.array_equal(x, y) for x, y in zip(cur_preds, self.testoutputs)])
                    elif len(self.testoutputs) == 0:
                        test_evaluated = False

                    if test_evaluated:
                        return 'solved', ['is_exp_cont', self.int_div[0]], cur_preds
                    else:
                        return 'objective initiated', ['is_exp_cont', self.int_div[0]], self.cur_test_preds

            elif not self.is_sim_out_shapes and self.is_sim_internal_int_div:
                candidate_factors = [x[0] for x in self.int_div]
                if all([int(x) != 0 for x in candidate_factors]):
                    is_expanded = all([np.all(np.repeat(np.repeat(x, int(y), 0), int(y), 1) == z) for x, y, z in zip(self.cur_train_preds, candidate_factors, self.trainoutputs)])
                else:
                    is_expanded =  False
                if all([int(1/x) != 0 for x in candidate_factors]):
                    is_contracted = all([np.all(x[::int(1/y),::int(1/y)] == z) for x, y, z in zip(self.cur_train_preds, candidate_factors, self.trainoutputs)])
                else:
                    is_contracted =  False
                if is_expanded or is_contracted:
                    r1, r2 = self.assess_num_unique_nonbg(candidate_factors)
                    if r1:
                        if is_expanded:
                            cur_preds = [np.repeat(np.repeat(x, y, 0), y, 1) for x, y in zip(self.cur_test_preds, [r2] * len(self.cur_test_preds))]
                        elif is_contracted:
                            cur_preds = [x[::int(1/y),::int(1/y)] for x, y in zip(self.cur_test_preds, [r2] * len(self.cur_test_preds))]
                        if len(self.testoutputs) == len(self.cur_test_preds):
                            test_evaluated = all([np.array_equal(x, y) for x, y in zip(cur_preds, self.testoutputs)])
                        elif len(self.testoutputs) == 0:
                            test_evaluated = False

                        if test_evaluated:
                            return 'solved', ['is_exp_cont_freq_unique_nonbg', candidate_factors], cur_preds
                        else:
                            return 'objective initiated', ['is_exp_cont_freq_unique_nonbg', candidate_factors], self.cur_test_preds

        return 'unsolved', [], self.cur_test_preds
