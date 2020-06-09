# we are trying to use this file mainly for unobjectified tasks


from utils import *

class Cages:
    def __init__(self, communication, running_objective):
        self.communication = communication
        self.running_objective = running_objective
        self.cur_test_preds = deepcopy(self.communication.testinputs)

    def check_objective_against_direct(self):
        copy_objective = deepcopy(self.running_objective)
        for n in range(len(copy_objective)):
            this_check = copy_objective[n]
            if len(this_check) == 1:
                copy_objective[n] = [this_check[0][0], this_check[0][1], 'direct']
            elif len(this_check) == 2:
                if this_check[0][0] != this_check[1][0]:
                    copy_objective[n][0] = [this_check[0][0], this_check[0][1], 'direct']
                    copy_objective[n][1] = [this_check[1][0], this_check[1][1], 'direct']
        if is_list_of_list_of_list(copy_objective):
            return copy_objective[0]
        return copy_objective

    def retrieve_coords_of_assignemnt_leads(self):
        # print(self.communication.assignments_leads)
        # print(self.communication.test_token_to_color)
        coords = []
        ass_leads = list(get_bare_assignment_leads(self.communication.assignments_leads))
        ass_leads_retokenized = [self.communication.test_token_to_color[x] for x in ass_leads]
        for n in range(len(self.communication.testinputs)):
            this_test_coords = {}
            this_test = self.communication.testinputs[n]
            for x in range(len(ass_leads)):
                this_ass_lead = ass_leads[x]
                this_ass_lead_coords = retrieve_coords_nonbg_from_arr(ass_leads_retokenized[x], this_test)
                this_test_coords[this_ass_lead] = this_ass_lead_coords
            coords.append(this_test_coords)
        return coords

    def assess_direct(self):
        if len(self.running_objective) > 20:
            return self.running_objective, []
        new_objective = self.check_objective_against_direct()
        test_rep = []
        if new_objective != self.running_objective:
            test_rep = self.retrieve_coords_of_assignemnt_leads()
        return new_objective, test_rep

    def screen_direct_trnsformations(self):
        is_solved = 'unsolved'
        mechanisms = []
        a, b =  self.assess_direct()
        if len(b) > 0:
            # this next feedback mechanism will need to be better for cases 202
            if self.communication.objective_status == 'red':
                # In case of 'obd' cases, the test expectation is not ready to handle unsatisfiable objectives
                a = self.communication.set_test_expectations(a)
            self.running_objective = a
            objective_satisfiability = [len(x) == 3 for x in a]
            if all(objective_satisfiability):
                this_output = [self.communication.build_a_prediction(self.communication.dimension_status, k, direct_transform = l) for k, l in zip(self.communication.testinputs, b)]
                if all([np.array_equal(x, y) for x, y in zip(this_output, self.communication.testoutputs)]):
                    is_solved = 'solved'
                    mechanisms.append('cages')
                    mechanisms.append('direct_transform')
                    self.cur_test_preds = this_output
        return is_solved, mechanisms, self.cur_test_preds, self.running_objective

    def screen_cages(self):
        if self.communication.dimension_status == 'deduced':
            test_bg = [get_background(x) for x in self.communication.testinputs]
            # output is bg (no global bg): 128
            if self.communication.is_unique_output == self.communication.bg:
                this_output = [self.communication.build_a_prediction(self.communication.dimension_status, k, add_to_zero = l) for k, l in zip(self.communication.output_dim_preds, test_bg)]
                return 'solved', ['cages', 'output_is_bg'], this_output, self.running_objective

            # output is the only nonbg in the input: 338
            elif self.communication.is_unique_output != None and all([sum([y != self.communication.bg  for y in np.unique(x).tolist()]) == 1 for x in self.communication.traininputs]):
                print('hello from here')
                test_nonbg = [np.unique(x).tolist()[1] for x in self.communication.testinputs]
                this_output = [self.communication.build_a_prediction(self.communication.dimension_status, k, add_to_zero = l) for k, l in zip(self.communication.output_dim_preds, test_nonbg)]
                return 'solved', ['cages', 'output_is_one_nonbg'], this_output, self.running_objective
            # the next is direct transformation on tasks with no single bg and with known expected dimension: last in row
            elif type(self.communication.bg) != int and self.communication.objective_status != 'irr':
                is_solved, mechanisms, this_output, this_running_objective = self.screen_direct_trnsformations()
                if is_solved == 'solved':
                    return is_solved, mechanisms, this_output, this_running_objective

        return 'unsolved', [], self.communication.testinputs, self.running_objective
