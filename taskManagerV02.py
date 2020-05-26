# Be subtle and abstract
# lists are the container we will use to handle different couples, etc
from utils import *

class TaskManager: # works on a task by task level, there are checks and balances
    # initiation defines several data members and uses class methods so as to output a problem graph
    def __init__(self, raw_task):
        self.raw_task = raw_task
        self.num_train = len(raw_task['train'])
        self.num_test = len(raw_task['test'])

        self.traininputs, self.trainoutputs = get_training(raw_task) # this will return two lists for inputs and outputs
        self.testinputs, self.testoutputs = get_testing(raw_task)

        # gather general information about the dimensionality from training
        self.is_similar_dim, self.input_dims, self.output_dims = explore_dimensions(self.traininputs, self.trainoutputs)
        self.traininputs_vals = [np.unique(n).tolist() for n in self.traininputs]
        self.trainoutputs_vals = [np.unique(n).tolist() for n in self.trainoutputs]
        self.couple_val_map = [get_value_map(n, m) for n, m in zip(self.traininputs, self.trainoutputs)]
        self.global_value_map = get_global_value_map(self.couple_val_map)
        self.couple_similars = [get_similars(n, m) for n, m in zip(self.traininputs, self.trainoutputs)]
        self.global_similars = get_global_value_map(self.couple_similars)

        # prepare relevant info for your tests conditionally on presence of testoutputs
        self.testinputs_vals = [np.unique(n).tolist() for n in self.testinputs]

        self.traininputs_bg = [get_background(n) for n in self.traininputs]
        self.testinputs_bg = [get_background(n) for n in self.testinputs]


        traininputs_bg_set = set(self.traininputs_bg)
        if len(traininputs_bg_set) == 1:
            self.global_bg = True
            self.bg = traininputs_bg_set.pop()
            self.bg = int(self.bg) #it is coming as numpy.int64 not int
        else:
            self.global_bg = False
            self.bg = list(traininputs_bg_set)


        # Move into deduction saver 0
        if self.is_similar_dim and type(self.bg) == int: #self.global_input_bg != None:
            situation  = [process_diff(n, m, self.bg) for n, m in zip(self.traininputs, self.trainoutputs)]
            #print('situation is: ', situation)
            situation_set  = set([str(tuple(n)) for n in situation])
            if len(situation_set) == 1:
                self.deductive_coder0 = situation[0]
            else:
                self.deductive_coder0 = None
        else:
            self.deductive_coder0 = None

        # Move into deduction saver 1
        if self.is_similar_dim and type(self.bg) == int: #self.global_input_bg != None:
            situation_spatial  = [process_diff_spatial(n, m, self.bg) for n, m in zip(self.traininputs, self.trainoutputs)]
            #print('situation_spatial is: ', situation_spatial)
            situation_set  = set([str(tuple(n)) for n in situation_spatial])
            if len(situation_set) == 1:
                self.deductive_coder1 = situation_spatial[0]
            else:
                self.deductive_coder1 = None
        else:
            self.deductive_coder1 = None

    # Tokenizer: alright, every task is different, but there is a global pattern in all the tasks
        self.priority_nonbg_input = sorted(list(get_common_nonbg_inputs(self.traininputs_vals)))
        self.priority_nonbg_output = sorted(list(get_common_nonbg_inputs(self.trainoutputs_vals)))
        self.color_to_tokens, self.token_to_colors, self.problem_statements =  self.tokenize()
        self.problem_graph = self.express_problem_graph()
        self.assignments_leads, self.asssignments_output = self.generate_set_assignemtns()
        self.objective, self.objective_status = self.get_objectives()

    # self.color_to_tokens, self.token_to_colors, self.problem_statements, self.problem_graph, self.asssignments_output
    # we have the above for each training couple
    # extract better objectives


    # Methods
    # This method tokenize a task
    def tokenize(self):
        if self.is_similar_dim and self.global_bg: #self.tokenizer_class == 'C':
            color_to_tokens = [get_col_to_token_class_C(x, y, z, self.bg, self.priority_nonbg_input + self.priority_nonbg_output) for x, y, z in zip(self.traininputs_vals, self.trainoutputs_vals, self.couple_val_map)]
            token_to_colors = [get_token_to_color_class_C(x) for x in color_to_tokens]
            problem_statements = [get_problem_statement_class_C(x, y) for x, y in zip(color_to_tokens, self.couple_val_map)]

            same_color_to_token = all(x == color_to_tokens[0] for x in color_to_tokens)
            #same_problem_statement = all(x == problem_statements[0] for x in problem_statements)

            if same_color_to_token:
                #print('Found ONE code: ')
                return color_to_tokens, token_to_colors, problem_statements
            else:
                #print('Found MULTIPLE codes: ')
                return color_to_tokens, token_to_colors, problem_statements
        else:
            #print('coming soon in a different taste!')
            return {}, {}, []

    # This method provide a holistic problem graph of the task
    def from_problem_statement_to_a_problem_graph(self, problem_statement, color_to_tokens, token_to_colors, couple_similar, traininput, trainoutput):
        problem_graph = deepcopy(problem_statement)
        for n in couple_similar.keys():
            if n in color_to_tokens.keys():
                problem_graph.append((color_to_tokens[n], color_to_tokens[n]))
            else:
                if n in traininput and n in trainoutput:
                    problem_graph.append((n, n)) # this ciuld be a relevant anchor nonbg
        return problem_graph

    # apply from_problem_statement_to_a_problem_graph to the task
    def express_problem_graph(self):
        if len(self.problem_statements) > 0 and type(self.problem_statements[0]) == list:
            return [self.from_problem_statement_to_a_problem_graph(problem_statement, color_to_tokens, token_to_colors, couple_similar, traininput, trainoutput) for problem_statement, color_to_tokens, token_to_colors, couple_similar, traininput, trainoutput in zip(self.problem_statements, self.color_to_tokens, self.token_to_colors, self.couple_similars, self.traininputs, self.trainoutputs)]
        else:
            return ['ARCsolver doesn can not express a problem graph yet.']

    def generate_set_assignemtns(self):
        if len(self.problem_graph) > 0 and type(self.problem_graph[0]) == list:
            results = [generate_set_assignemtns_per_graph(problem_graph, token_to_colors, traininput, trainoutput) for problem_graph, token_to_colors, traininput, trainoutput in zip(self.problem_graph, self.token_to_colors, self.traininputs, self.trainoutputs)]
            assignments_leads = []
            asssignments_output = []
            for n in range(len(results)):
                assignments_leads.append(results[n][0])
                asssignments_output.append(results[n][1])
            return assignments_leads, asssignments_output
        else:
            return ['ARCsolver doesn can not generate sets out of the problem graph yet.'], ['This requires a different mindset!']


    # what are the pairs to be resolved
    def get_objectives(self):
        if type(self.problem_graph[0]) != str:
            copy_problem_graph = []
            for n in self.problem_graph:
                copy_problem_graph.append(tuple(sorted([y for y in n if type(y[0]) == str])))

            copy_problem_graph = list(set(copy_problem_graph))
            if len(copy_problem_graph) == 1:
                return get_this_objective(copy_problem_graph), 'obd' # one by default
            elif len(copy_problem_graph) > 1:
                current_boss = []
                copy_problem_graph = sorted(copy_problem_graph, key=len, reverse=False)
                for n in range(len(copy_problem_graph)-1):
                    if set(copy_problem_graph[n]).issubset(set(copy_problem_graph[n + 1])):
                        if len(current_boss) > 0 and current_boss[len(current_boss) - 1] != copy_problem_graph[n+1]:
                            current_boss.append(copy_problem_graph[n+1])
                        elif len(current_boss) == 0:
                            current_boss.append(copy_problem_graph[n+1])

                if len(current_boss) == 0:
                    return copy_problem_graph, 'irr'# this is another level of difficulty I guess, irreducible
                elif len(current_boss)  == 1:
                    return get_this_objective(current_boss), 'red' # we had a total reduction here, just account for variability
                elif len(current_boss)  > 1 and len(current_boss) < len(self.problem_graph):
                    return [get_this_objective(x) for x in current_boss], 'pred' # a case must have been a subset of another case for sure, partially reduced

        else:
            return ['ARCsolver can not generate an objective for this task yet.'], 'None'

    def inductive_strategy(self):
        

    # simple print utilities
    def brief_task(self):
        # print('is_similar_dim: ', self.is_similar_dim,)
        # print('input_dims info: ', self.input_dims)
        # print('output_dims info: ', self.output_dims)
        # print('traininputs_vals: ', self.traininputs_vals)
        # print('trainoutputs_vals: ', self.trainoutputs_vals)
        # print('couple_val_map: ', self.couple_val_map)
        # print('global_bg: ', self.global_bg)
        # print('bg: ', self.bg)
        # print('deductive_coder0: ', self.deductive_coder0)
        # print('deductive_coder1: ', self.deductive_coder1)
        # print('global_value_map: ', self.global_value_map)
        # print('global_similars: ', self.global_similars)
        # print("=========")
        # print('information about tokenization:')
        # print('color_to_tokens: ', self.color_to_tokens)
        # print('token_to_colors: ', self.token_to_colors)
        # print('problem_statements: ', self.problem_statements)
        # print('length problem_graph: ', len(self.problem_graph))
        # print('problem_graph: ', self.problem_graph)
        #
        # if type(self.asssignments_output[0]) == list:
        #     for m in range(len(self.asssignments_output)):
        #         for n in self.asssignments_output[m]:
        #             for k, v in n.items():
        #                 print(k, ' : ', str([len(l) for l in v]))
        #             print("end of assignment.")
        #         print("end of an option :).")
        # print(self.traininputs)
        print(self.objective)
        print(self.objective_status)
        print("=========")
