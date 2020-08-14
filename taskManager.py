from utils import *

def flip_test_train(task, num):
    modified_task = deepcopy(task)
    temp = deepcopy(modified_task['train'][num])
    test = deepcopy(modified_task['test'])[0]
    modified_task['train'][num] = test
    modified_task['test'][0] = temp
    return modified_task

def confirmer(task):
    reference_task = TaskManager(task)

    training = task['train']
    num_train = len(training)
    scope = []
    scope.append(TaskManager(task).inductive.solved)
    for n in range(num_train):
        modified_task = flip_test_train(task, n)
        managed_mod_task = TaskManager(modified_task)
        if managed_mod_task.inductive.solved != 'solved' and type(reference_task.bg) == int:
            managed_mod_task = TaskManager(modified_task, reference_task.bg)
        scope.append(managed_mod_task.inductive.solved)

    return all([x == 'solved' for x in scope])

class TaskManager: # works on a task by task level, there are checks and balances
    # initiation defines several data members and uses class methods so as to output a problem graph
    def __init__(self, raw_task, forced_bg = None):
        self.raw_task = raw_task
        self.num_train = len(raw_task['train'])
        self.num_test = len(raw_task['test'])

        self.traininputs, self.trainoutputs = get_training(raw_task) # this will return two lists for inputs and outputs
        self.testinputs, self.testoutputs = get_testing(raw_task)
        self.cur_train_preds, self.cur_test_preds = deepcopy(self.traininputs), deepcopy(self.testinputs)
        # gather general information about the dimensionality from training
        self.is_similar_dim, self.input_dims, self.output_dims = explore_dimensions(self.traininputs, self.trainoutputs)

        self.traininputs_vals = [np.unique(n).tolist() for n in self.traininputs]
        self.trainoutputs_vals = [np.unique(n).tolist() for n in self.trainoutputs]
        self.testinputs_vals = [np.unique(n).tolist() for n in self.testinputs]

        self.couple_val_map = [get_value_map(n, m, x, y) for n, m, x, y in zip(self.traininputs, self.trainoutputs, self.traininputs_vals, self.trainoutputs_vals)]
        self.global_value_map = get_global_value_map(self.couple_val_map)
        self.couple_similars = [get_similars(n, m) for n, m in zip(self.traininputs, self.trainoutputs)]
        self.global_similars = get_global_value_map(self.couple_similars)

        # prepare relevant info for your tests conditionally on presence of testoutputs
        self.traininputs_bg = [get_background(n) for n in self.traininputs]
        self.trainoutputs_bg = [get_background(n) for n in self.trainoutputs]
        self.testinputs_bg = [get_background(n) for n in self.testinputs]

        # assess bg and apply deductive routines to the task
        self.global_bg, self.bg = self.assess_bg_situation()
        if forced_bg != None:
            self.global_bg = True
            self.bg = forced_bg

        # deductive
        self.deductive_coder0, self.deductive_coder1 = self.expose_deductive()

        # Tokenizer: alright, every task is different, but there is a global pattern in all the tasks
        all_test_input_vals = list(itertools.chain.from_iterable(self.testinputs_vals))
        self.priority_nonbg_input = sorted(list(get_common_nonbg(self.traininputs_vals)))
        self.priority_nonbg_input = [x for x in self.priority_nonbg_input if x in all_test_input_vals and x in self.global_value_map.keys()]
        self.priority_nonbg_output = sorted(list(get_common_nonbg(self.trainoutputs_vals)))
        list_global_value_map_values = list(itertools.chain.from_iterable(list(self.global_value_map.values())))
        self.priority_nonbg_output = [x for x in self.priority_nonbg_output if x in list_global_value_map_values]

        self.color_to_tokens, self.token_to_colors, self.problem_statements =  self.tokenize()

        # problem description
        self.problem_graph = self.express_problem_graph()
        self.assignments_leads, self.int_anchor_vals, self.asssignments_output = self.generate_set_assignemtns()
        self.objective, self.objective_status = self.get_objectives()

        self.size_sorted_nonbg = self.get_size_sorted_nonbg()
        self.test_token_to_colors = self.get_test_token_to_color() # work out your self.test_token_to_color: 227, 328 are example of a blind spot of this system with 'direct' strategy
        self.test_color_to_tokens = [get_token_to_color(x) for x in self.test_token_to_colors]
        self.test_objective = self.set_test_expectations()

    # Methods
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


    def expose_deductive(self):
        if self.is_similar_dim and type(self.bg) == int:
            situation  = [process_diff(n, m, self.bg) for n, m in zip(self.traininputs, self.trainoutputs)]
            situation_set  = set([str(tuple(n)) for n in situation])
            situation_spatial  = [process_diff_spatial(n, m, self.bg) for n, m in zip(self.traininputs, self.trainoutputs)]
            situation_set_spatial  = set([str(tuple(n)) for n in situation_spatial])

            if len(situation_set) == 1:
                deductive_coder0 = situation[0]
            else:
                deductive_coder0 = None
            if len(situation_set_spatial) == 1:
                deductive_coder1 = situation_spatial[0]
            else:
                deductive_coder1 = None
            return deductive_coder0, deductive_coder1
        else:
            return None, None

    # This method tokenize a task based on training, it provides color_to_token, token_to_color, and problem_statements
    def tokenize(self):
        color_to_tokens = [get_col_to_token(w, x, y, z, self.bg, self.priority_nonbg_input + self.priority_nonbg_output) for w, x, y, z in zip(self.traininputs, self.traininputs_vals, self.trainoutputs_vals, self.couple_val_map)]
        token_to_colors = [get_token_to_color(x) for x in color_to_tokens]
        problem_statements = [get_problem_statement(x, y) for x, y in zip(color_to_tokens, self.couple_val_map)]
        return color_to_tokens, token_to_colors, problem_statements

    # steps to generate an objective from problem_statements (problem_statement --> problem_graph --> objective )
    def from_problem_statement_to_a_problem_graph(self, problem_statement, color_to_tokens, token_to_colors, couple_similar, traininput, trainoutput):
        problem_graph = deepcopy(problem_statement)
        for n in couple_similar.keys():
            if n in color_to_tokens.keys():
                problem_graph.append((color_to_tokens[n], color_to_tokens[n]))
            else:
                if n in traininput and n in trainoutput:
                    problem_graph.append((n, n)) # this could be a relevant anchor nonbg
        return problem_graph

    def express_problem_graph(self):
        if len(self.problem_statements) > 0 and type(self.problem_statements[0]) == list:
            return [self.from_problem_statement_to_a_problem_graph(problem_statement, color_to_tokens, token_to_colors, couple_similar, traininput, trainoutput) for problem_statement, color_to_tokens, token_to_colors, couple_similar, traininput, trainoutput in zip(self.problem_statements, self.color_to_tokens, self.token_to_colors, self.couple_similars, self.traininputs, self.trainoutputs)]
        else:
            return ['ARCsolver doesn can not express a problem graph yet.']

    def get_objectives(self):
        if type(self.problem_graph[0]) != str:
            objectives = [get_this_objective(x, self.assignments_leads, self.is_similar_dim) for x in self.problem_graph]
            if all([x == objectives[0] for x in objectives]):
                return objectives, 'obd'
            else:
                copy_problem_graph = []
                for n in self.problem_graph:
                    copy_problem_graph.append(tuple(sorted([y for y in n if type(y[0]) == str])))
                copy_problem_graph = sorted(list(set(copy_problem_graph)), key = len)
                current_boss = set()
                for n in range(len(copy_problem_graph)-1):
                    if set(copy_problem_graph[n]).issubset(set(copy_problem_graph[n + 1])):
                        if len(current_boss) > 0:
                            if copy_problem_graph[n] in current_boss:
                                current_boss.remove(copy_problem_graph[n])
                            if copy_problem_graph[n+1] not in current_boss:
                                current_boss.add(copy_problem_graph[n+1])
                        elif len(current_boss) == 0:
                            current_boss.add(copy_problem_graph[n+1])
                    else:
                        current_boss = set()
                        break

                current_boss = list(current_boss)
                if len(current_boss) == 0:
                    return objectives, 'irr'# this is another level of difficulty I guess, irreducible
                elif len(current_boss)  == 1:
                    return objectives, 'red' # we had a total reduction here, just account for variability
        else:
            return ['ARCsolver can not generate an objective for this task yet.'], None

    # problem_graph --> assignments_leads, int_anchor_vals, asssignments_output
    def generate_set_assignemtns(self):
        if len(self.problem_graph) > 0 and type(self.problem_graph[0]) == list:
            results = [generate_set_assignemtns_per_graph(problem_graph, token_to_colors, traininput, trainoutput) for problem_graph, token_to_colors, traininput, trainoutput in zip(self.problem_graph, self.token_to_colors, self.traininputs, self.trainoutputs)]
            assignments_leads = []
            int_anchor_vals = []
            asssignments_output = []
            for n in range(len(results)):
                assignments_leads.append(tuple(results[n][0]))
                int_anchor_vals.append(tuple(results[n][1]))
                asssignments_output.append(results[n][2])
            return assignments_leads, int_anchor_vals, asssignments_output
        else:
            return ['ARCsolver doesn can not generate sets out of the problem graph yet.'], [], ['This requires a different mindset!']

    # Tokenize test according to tokenized_train
    def get_size_sorted_nonbg(self):
        assignments_leads_set = set()
        handles_set = set()
        modified_combs = set()
        if type(self.assignments_leads[0]) != str:
            # get handles_list
            for x in self.assignments_leads:
                for l in x:
                    if 'nonbg' in l and 'pr' not in l:
                        handles_set.add(l)
            if len(handles_set) > 0:
                handle_list = list(handles_set)
                # if you have a handle list, get a dict_of_lengths based on the training
                dict_of_lengths = {}
                for x in range(len(handle_list)):
                    this_target = handle_list[x]
                    for m in range(len(self.asssignments_output)):
                        for d in self.asssignments_output[m]:
                            for k, v in d.items():
                                if this_target == k[1][0]:
                                    if this_target not in  dict_of_lengths.keys():
                                        dict_of_lengths[this_target] = []
                                    dict_of_lengths[this_target] =  dict_of_lengths[this_target] + [len(l) for l in v]


                lists_of_lengths = list(dict_of_lengths.values())
                if not all([x == lists_of_lengths[0] for x in lists_of_lengths]) and \
                       all([len(x) == len(lists_of_lengths[0]) for x in lists_of_lengths]) and \
                       len(lists_of_lengths) > 1:
                    combs = list(combinations(handle_list, 2))

                    for n in range(len(combs)):
                        if all([x > y for x, y in zip(dict_of_lengths[combs[n][0]], dict_of_lengths[combs[n][1]])]):
                            modified_combs.add((combs[n][1], sum(dict_of_lengths[combs[n][1]])))
                            modified_combs.add((combs[n][0], sum(dict_of_lengths[combs[n][0]])))
                        elif all([x < y for x, y in zip(dict_of_lengths[combs[n][0]], dict_of_lengths[combs[n][1]])]):
                            modified_combs.add((combs[n][1], sum(dict_of_lengths[combs[n][1]])))
                            modified_combs.add((combs[n][0], sum(dict_of_lengths[combs[n][0]])))
                    modified_combs = sorted(list(modified_combs), key=lambda tup: tup[1])
                    modified_combs = [x[0] for x in modified_combs]
        return modified_combs

    def get_test_token_to_color(self):
        test_token_to_colors = []
        if len(self.token_to_colors) > 0:
            if all([x == self.token_to_colors[0] for x in self.token_to_colors]):
                for n in range(len(self.testinputs)):
                    test_token_to_colors.append(self.token_to_colors[0])

            else:
                pr_tokens = get_priority_token(self.token_to_colors)
                testinputs_vals = [np.unique(n).tolist() for n in self.testinputs]
                final_test_nonbg = sorted(list(get_common_nonbg(testinputs_vals)))
                for n in range(len(self.testinputs)):
                    this_test_color_to_token = get_col_to_token_test(final_test_nonbg, pr_tokens, self.bg, self.size_sorted_nonbg, self.int_anchor_vals, self.testinputs[n])
                    test_token_to_colors.append(get_token_to_color(this_test_color_to_token))

        return test_token_to_colors

    # The following are supposed to be generic, unfortunately  infer_on_mechanism is geared towards captured module
    def set_test_expectations(self):
        current_objective = deepcopy(self.objective)

        test_objectives = []

        if self.objective_status == 'obd':
            for n in range(len(self.testinputs)):
                test_objectives.append(current_objective[0])

        else:
            train_dicts = deepcopy(self.token_to_colors)
            test_dicts = deepcopy(self.test_token_to_colors)

            train_dicts_keys = [set(x.keys()) for x in train_dicts]
            test_dicts_keys = [set(x.keys()) for x in test_dicts]

            for test in test_dicts_keys:
                match_screen = [x == test for x in train_dicts_keys]
                chekpoint = None
                if any(match_screen):
                    which_ind = match_screen.index(True)
                    test_objectives.append(deepcopy(self.objective[which_ind]))
                    chekpoint = 0

                if chekpoint == None:
                    intersections = [len(x.intersection(test)) for x in train_dicts_keys]

                    most_intersection_index = intersections.index(max(intersections))
                    to_modify_objective = deepcopy(self.objective[most_intersection_index])
                    # we wither add or remove from the to_modify_objective:
                    dict_to_compare_with = set(self.token_to_colors[most_intersection_index].keys())
                    if len(test) > len(dict_to_compare_with):
                        diff = test - dict_to_compare_with
                        for diff_element in diff:
                            to_modify_objective = expand_obj(to_modify_objective, diff_element)
                    elif len(test) < len(dict_to_compare_with):
                        diff = dict_to_compare_with - test
                        for diff_element in diff:
                            to_modify_objective = contract_obj(to_modify_objective, diff_element)
                    test_objectives.append(to_modify_objective)

        return test_objectives
