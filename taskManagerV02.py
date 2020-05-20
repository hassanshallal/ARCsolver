# Be subtle and abstract
# lists are the container we will use to handle different couples, etc
from utils import *

class TaskManager: # works on a task by task level, there are checks and balances
    # initiation must take care of global descriptors of a task
    def __init__(self, raw_task, phase = 'Training'):
        self.raw_task = raw_task
        self.num_train = len(raw_task['train'])
        self.phase = phase

        if self.phase == 'Training': # both training and evaluation
            self.traininputs, self.trainoutputs = get_training(raw_task) # this will return two lists for inputs and outputs
        else:
            raise NameError("Current option is only 'Training'")

        # gather general information about the dimensionality
        self.is_similar_dim, self.input_dims, self.output_dims = explore_dimensions(self.traininputs, self.trainoutputs)

        # get list of lists of values in inputs, outputs
        self.traininputs_vals = [np.unique(n).tolist() for n in self.traininputs]
        self.trainoutputs_vals = [np.unique(n).tolist() for n in self.trainoutputs]
        self.couple_val_map = [get_value_map(n, m) for n, m in zip(self.traininputs, self.trainoutputs)] # needs further work
        self.global_value_map = get_global_value_map(self.couple_val_map)
        self.couple_similars = [get_similars(n, m) for n, m in zip(self.traininputs, self.trainoutputs)] # needs further work
        self.global_similars = get_global_value_map(self.couple_similars)
        #print('global_value_map is: ', self.global_value_map)

        self.traininputs_bg = [get_background(n) for n in self.traininputs]
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
    # We have a global tokenizer and a couple tokenizer
        #self.tokenizer_class = self.class_of_tokenizer()
        #if self.tokenizer_class == 'C':
        self.color_to_tokens, self.token_to_colors, self.problem_statements =  self.tokenize()
        self.problem_graph = self.express_problem_graph()
        self.assignments_leads, self.asssignments_output = self.generate_set_assignemtns()


    # Methods
    def tokenize(self):
        if self.is_similar_dim and self.global_bg: #self.tokenizer_class == 'C':
            color_to_tokens = [get_col_to_token_class_C(x, y, z, self.bg) for x, y, z in zip(self.traininputs_vals, self.trainoutputs_vals, self.couple_val_map)]
            token_to_colors = [get_token_to_color_class_C(x) for x in color_to_tokens]
            problem_statements = [get_problem_statement_class_C(x, y) for x, y in zip(color_to_tokens, self.couple_val_map)]

            #same_color_to_token = all(x == color_to_tokens[0] for x in color_to_tokens)
            #same_problem_statement = all(x == problem_statements[0] for x in problem_statements)

            #if same_problem_statement:
                #print('Found ONE code: ', problem_statements[0])
                #return color_to_tokens[0], token_to_colors[0], problem_statements[0]
            #else:
            consensus_color_to_tokens = max(list(color_to_tokens), key = color_to_tokens.count)
            consensus_token_to_colors = max(list(token_to_colors), key = token_to_colors.count)
            consensus_problem_statements = max(list(problem_statements), key = problem_statements.count)
            return consensus_color_to_tokens, consensus_token_to_colors, consensus_problem_statements
        else:
            print('coming soon in a different taste!')
            return {}, {}, []

    def express_problem_graph(self):
        if len(self.problem_statements) > 0 and type(self.problem_statements[0]) == tuple:
            problem_graph = deepcopy(self.problem_statements)
            for n in self.global_similars.keys():
                if n in self.color_to_tokens.keys():
                    problem_graph.append((self.color_to_tokens[n], self.color_to_tokens[n]))
                else:
                    if all([n in m for m in self.traininputs]) and all([n in m for m in self.trainoutputs]):
                        problem_graph.append((n, n)) # this may be a relevant anchor nonbg
                    elif get_num_nonbg(self.token_to_colors) == 1 and is_only_one_nonbg_per_couple(self.traininputs_vals, self.trainoutputs_vals, self.bg):
                        if type(self.token_to_colors['nonbg0']) == int:
                            self.token_to_colors['nonbg0'] = [self.token_to_colors['nonbg0']]
                        self.token_to_colors['nonbg0'].append(n)
            return problem_graph
        else:
            return ['ARCsolver doesn not have a clear problem statment yet.']

    def get_coordinates_of_tuple(self, x): # self method
        coordinates = []
        if type(x) == tuple:
            for n in range(len(self.traininputs)):
                in_ = self.traininputs[n]
                out_ = self.trainoutputs[n]
                target_indices = np.argwhere((in_ == x[0]) & (out_ == x[1]))
                coordinates.append(target_indices)
        elif type(x) == list:
            for n in range(len(self.traininputs)):
                in_ = self.traininputs[n]
                out_ = self.trainoutputs[n]
                for m in x:
                    if m[0] in in_ and m[1] in out_:
                        target_indices = np.argwhere((in_ == m[0]) & (out_ == m[1]))
                        coordinates.append(target_indices)
        return coordinates


    def retokenize(self, x):
        if x[0] in self.token_to_colors.keys() and x[1] in self.token_to_colors.keys():
            if type(self.token_to_colors[x[0]]) == int and type(self.token_to_colors[x[1]]) == int:
                return(self.token_to_colors[x[0]], self.token_to_colors[x[1]])
            elif type(self.token_to_colors[x[0]]) == list or type(self.token_to_colors[x[1]]) == list:
                output_list = []
                if type(self.token_to_colors[x[0]]) == list and type(self.token_to_colors[x[1]]) != list:
                    target = self.token_to_colors[x[0]]
                    off_target = self.token_to_colors[x[1]]
                elif type(self.token_to_colors[x[0]]) != list and type(self.token_to_colors[x[1]]) == list:
                    off_target = self.token_to_colors[x[0]]
                    target = self.token_to_colors[x[1]]
                else:
                    target = self.token_to_colors[x[0]]
                    off_target = self.token_to_colors[x[1]]
                for n in range(len(target)):
                    if type(off_target) == int:
                        output_list.append((target[n], off_target))
                    elif type(off_target) == list:
                        output_list.append((target[n], off_target[n]))
                return output_list
        else:
            return 'Issue with retokenization.'

    def generate_set_assignemtns(self):
         # specificaaly pick indices and tuples by category: int_anchor, token_anchor, difference
        int_anchors_indices = [i for i, val in enumerate(self.problem_graph) if val[0] not in self.token_to_colors.keys()]
        int_anchors = [self.problem_graph[x] for x in int_anchors_indices]

        token_anchors_indices = [i for i, val in enumerate(self.problem_graph) if val[0] == val[1] and type(val[0]) == str]
        token_anchors = [self.problem_graph[x] for x in token_anchors_indices]

        differences_indices = [i for i, val in enumerate(self.problem_graph) if val[0] != val[1] and type(val[0]) == str]
        differences = [self.problem_graph[x] for x in differences_indices]
        # set generation with cooridinates in all couples
        assignments_leads = sorted(list(set([x[0] for x in differences])))
        num_assignemnts  = len(assignments_leads)
        asssignments_output = []

        for n in range(num_assignemnts):
            this_assignemnt = {}
            for x in int_anchors:
                this_assignemnt[x] = self.get_coordinates_of_tuple(x)
            for x in token_anchors:
                if x[0] == assignments_leads[n]:
                    this_assignemnt[x] = self.get_coordinates_of_tuple(self.retokenize(x))
            for x in differences:
                if x[0] == assignments_leads[n]:
                    this_assignemnt[x] = self.get_coordinates_of_tuple(self.retokenize(x))
            asssignments_output.append(this_assignemnt)

        return assignments_leads, asssignments_output


    # simple print utilities
    def brief_task(self):
        print('is_similar_dim: ', self.is_similar_dim,)
        print('input_dims info: ', self.input_dims)
        print('output_dims info: ', self.output_dims)
        print('traininputs_vals: ', self.traininputs_vals)
        print('trainoutputs_vals: ', self.trainoutputs_vals)
        print('couple_val_map: ', self.couple_val_map)
        print('global_bg: ', self.global_bg)
        print('bg: ', self.bg)
        print('deductive_coder0: ', self.deductive_coder0)
        print('deductive_coder1: ', self.deductive_coder1)
        print('global_value_map: ', self.global_value_map)
        print('global_similars: ', self.global_similars)
        print("=========")
        print('information about tokenization:')
        #print('tokenization class: ', self.tokenizer_class)
        print('color_to_tokens: ', self.color_to_tokens)
        print('token_to_colors: ', self.token_to_colors)
        print('problem_statements: ', self.problem_statements)
        print('problem_graph: ', self.problem_graph)
        for n in self.asssignments_output:
            for k, v in n.items():
                print(k, ' : ', str([len(m) for m in v]))
            print("end of assignment.")
        print("=========")
