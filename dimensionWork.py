
from utils import *
from communicate import *

class DimensionWork:
    def __init__(self, communication, cur_test_preds):
        # This is a holistic induction class, only takes communication and cur_test_preds
        self.communication = communication
        self.cur_test_preds = cur_test_preds

        # data members to understand dimensions situation
        self.traininput_shapes = [list(x.shape) for x in self.communication.traininputs]
        self.trainoutputs_shapes = [list(x.shape) for x in self.communication.trainoutputs]
        self.testinput_shapes = [list(x.shape) for x in self.communication.testinputs]

        self.is_sim_in_shapes = all([x.shape == self.communication.traininputs[0].shape for x in self.communication.traininputs])
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


        self.dimension_status, self.output_dim_preds = self.cognify_dimensions()

    def cognify_dimensions(self):
        if self.is_same_ndim_couple and self.is_same_dim_couple:
            return 'deduced', [tuple(x.shape) for x in self.communication.testinputs]

        elif self.is_same_ndim_couple and not self.is_same_dim_couple:
            if self.same_relation and (self.what_relation[0] == '>' or self.what_relation[0] == '<'):
                if self.unidirctional and self.is_sim_int_div:
                    return 'deduced', [tuple(x) for x in modify_dimensiosn(self.testinput_shapes, self.in_mult_fact[0])]
                elif self.unidirctional and not self.is_sim_int_div:
                    if self.is_sim_out_shapes:
                        return 'deduced', [tuple(self.trainoutputs_shapes[0])] * len(self.communication.testinputs)
                    else:
                        if self.is_sim_internal_int_div:
                            return 'partially deduced', [tuple(x) for x in self.testinput_shapes]

            if self.is_one_row_output and self.col_dim_ouputs == self.communication.freq_nonbg_traininputs:
                return 'deduced', [(1, x) for x in self.communication.freq_nonbg_testinputs]
            if self.is_one_column_output and self.row_dim_ouputs == self.communication.freq_nonbg_traininputs:
                return 'deduced', [(x, 1) for x in self.communication.freq_nonbg_testinputs]

        return 'undeduced', [tuple(x) for x in self.testinput_shapes]

    def get_dimension_cognified(self):
        return self.dimension_status, self.output_dim_preds

    def assess_num_unique_nonbg(self, test_list):
        traininputs_vals = [np.unique(n).tolist() for n in self.communication.traininputs]
        testinputs_vals = [np.unique(n).tolist() for n in self.communication.testinputs]

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
                if int(self.in_mult_fact[0][0]) != 0 and int(self.in_mult_fact[0][1]) != 0:
                    is_expanded = all([np.all(np.repeat(np.repeat(x, int(y[0]), 0), int(y[1]), 1) == z) for x, y, z in zip(self.communication.traininputs, self.in_mult_fact, self.communication.trainoutputs)])
                else:
                    is_expanded =  False

                if int(1/self.in_mult_fact[0][0]) != 0 and int(1/self.in_mult_fact[0][1]) != 0:
                    is_contracted = all([np.all(x[::int(1/y[0]),::int(1/y[1])] == z) for x, y, z in zip(self.communication.traininputs, self.in_mult_fact, self.communication.trainoutputs)])
                else:
                    is_contracted = False
                if is_expanded or is_contracted:
                    if len(self.communication.testinputs) == 1:
                        this_in_mult_fact = [self.in_mult_fact[0]]
                    elif len(self.communication.testinputs) > 1:
                        this_in_mult_fact = self.in_mult_fact[0] * len(self.communication.testinputs)

                    if is_expanded:
                        cur_preds = [np.repeat(np.repeat(x, int(y[0]), 0), int(y[1]), 1) for x, y in zip(self.communication.testinputs, this_in_mult_fact)]
                    elif is_contracted:
                        cur_preds = [x[::int(1/y[0]),::int(1/y[1])] for x, y in zip(self.communication.testinputs, this_in_mult_fact)]

                    if self.communication.testoutputs:
                        test_evaluated = all([np.array_equal(x, y) for x, y in zip(cur_preds, self.communication.testoutputs)])
                    else:
                        test_evaluated = False

                    if test_evaluated:
                        return 'solved', ['is_exp_cont', self.in_mult_fact[0]], cur_preds
                    else:
                        return 'objective initiated', ['is_exp_cont', self.in_mult_fact[0]], self.cur_test_preds

            elif not self.is_sim_out_shapes and self.is_sim_internal_int_div:
                candidate_factors = [x[0] for x in self.in_mult_fact]
                if all([int(x) != 0 for x in candidate_factors]):
                    is_expanded = all([np.all(np.repeat(np.repeat(x, int(y), 0), int(y), 1) == z) for x, y, z in zip(self.communication.traininputs, candidate_factors, self.communication.trainoutputs)])
                else:
                    is_expanded =  False
                if all([int(1/x) != 0 for x in candidate_factors]):
                    is_contracted = all([np.all(x[::int(1/y),::int(1/y)] == z) for x, y, z in zip(self.communication.traininputs, candidate_factors, self.communication.trainoutputs)])
                else:
                    is_contracted =  False
                if is_expanded or is_contracted:
                    r1, r2 = self.assess_num_unique_nonbg(candidate_factors)
                    if r1:
                        if is_expanded:
                            cur_preds = [np.repeat(np.repeat(x, y, 0), y, 1) for x, y in zip(self.communication.testinputs, [r2] * len(self.communication.testinputs))]
                        elif is_contracted:
                            cur_preds = [x[::int(1/y),::int(1/y)] for x, y in zip(self.communication.testinputs, [r2] * len(self.communication.testinputs))]
                        if self.communication.testoutputs:
                            test_evaluated = all([np.array_equal(x, y) for x, y in zip(cur_preds, self.communication.testoutputs)])
                        else:
                            test_evaluated = False

                        if test_evaluated:
                            return 'solved', ['is_exp_cont_freq_unique_nonbg', candidate_factors], cur_preds
                        else:
                            return 'objective initiated', ['is_exp_cont_freq_unique_nonbg', candidate_factors], self.cur_test_preds

        return 'unsolved', [], self.cur_test_preds

    def screen_flips(self):
        # check on training
        are_flips = [screen_flips_rotation(in_, out_) for in_, out_ in zip(self.communication.traininputs, self.communication.trainoutputs)] # this took care of train validation
        if all([x[0] != False for x in are_flips]) and all([x[0] == are_flips[0][0] for x in are_flips]):
            if are_flips[0][1] != None:
                cur_preds = [are_flips[0][0](x, are_flips[0][1]) for x in self.cur_test_preds]
            else:
                cur_preds = [are_flips[0][0](x) for x in self.cur_test_preds]

            if self.communication.testoutputs:
                test_evaluated = all([np.array_equal(x, y) for x, y in zip(cur_preds, self.communication.testoutputs)])
            else:
                test_evaluated = False

            if test_evaluated:
                return 'solved', [(are_flips[0][0], are_flips[0][1])], cur_preds
            elif not test_evaluated:
                return 'objective initiated', [(are_flips[0][0], are_flips[0][1])], self.cur_test_preds
        else:
            return 'unsolved', [], self.cur_test_preds
