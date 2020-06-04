# this is a class that wraps important information in a single object to be passed to anyother class
from utils import *
class Communication:
    def __init__(self, traininputs, trainoutputs, testinputs, bg, objective_status, objective, assignments_leads, asssignments_output, token_to_colors, test_token_to_color, testoutputs = None):
        self.traininputs = traininputs
        self.trainoutputs = trainoutputs
        self.testinputs = testinputs
        self.bg = bg

        self.objective_status = objective_status
        self.objective = objective
        self.assignments_leads = assignments_leads
        self.asssignments_output = asssignments_output
        self.token_to_colors = token_to_colors
        self.test_token_to_color = test_token_to_color
        self.testoutputs = testoutputs

    def first_principles_diff_dim_checks(self, *args):
        for n in args:
            ext_len = len(n)
            int_len = len(n[0])
            for x in range(ext_len):
                for y in range(int_len):
                    if len(n[x][y][0]) == 0 or len(n[x][y][1]) == 0:
                        return False
        return True

    def first_principles_diff_dim_signal_size(self, *args):
        zero_signal = []
        one_signal = []

        for n in args:
            ext_len = len(n)
            int_len = len(n[0])
            for x in range(ext_len):
                for y in range(int_len):
                    zero_signal.append(len(n[x][y][0]))
                    one_signal.append(len(n[x][y][1]))

        return zero_signal, one_signal

    def first_principles_diff_dim_signal_class(self, results):
        overall_cur_invest_num_unique = []
        overall_cur_invest_unique = []
        for n in range(len(results)): # 3 for captured
            cur_invest = results[n]
            cur_invest_num_unique = []
            cur_invest_unique = []
            for x in range(len(cur_invest)): # num_train
                this_result = cur_invest[x]
                values_1 = get_set_of_values_1(this_result)
                cur_invest_num_unique.append(len(values_1))
                if len(values_1) == 1:
                    cur_invest_unique.append(values_1.pop())
                else:
                    cur_invest_unique.append(values_1)
            overall_cur_invest_num_unique.append(cur_invest_num_unique)
            overall_cur_invest_unique.append(cur_invest_unique)

        return overall_cur_invest_num_unique, overall_cur_invest_unique

    def carry_along(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        retrun
