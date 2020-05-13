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
        
        # background
        self.traininputs_bg = [get_background(n) for n in self.traininputs]
        traininputs_bg_set = set(self.traininputs_bg)
        if len(traininputs_bg_set) == 1:
            self.global_input_bg = traininputs_bg_set.pop()
        else:
            self.global_input_bg = None
        
        self.trainoutputs_bg = [get_background(n) for n in self.trainoutputs]
        self.couple_same_bg = [n == m for n, m in zip(self.traininputs_bg, self.trainoutputs_bg)]
        
        
        self.bg_global_set = set(self.traininputs_bg + self.trainoutputs_bg)
        if len(self.bg_global_set) == 1:
            self.global_bg = True
            self.bg = self.bg_global_set.pop()
        else:
            self.global_bg = False
            self.bg = list(self.bg_global_set)
           
        
        # Move into deduction saver 0
        if self.is_similar_dim and self.global_input_bg != None:
            situation  = [process_diff(n, m, self.global_input_bg) for n, m in zip(self.traininputs, self.trainoutputs)]
            situation_set  = set([str(tuple(n)) for n in situation])
            if len(situation_set) == 1:
                self.deductive_coder0 = situation[0]
            else:
                self.deductive_coder0 = None
        else:
            self.deductive_coder0 = None
        
        # print prelim info about task
        print('is_similar_dim: ', self.is_similar_dim,)
        print('input_dims info: ', self.input_dims)
        print('output_dims info: ', self.output_dims)
        print('traininputs_vals: ', self.traininputs_vals)
        print('trainoutputs_vals: ', self.trainoutputs_vals)
        print('global_bg: ', self.global_bg)
        print('bg: ', self.bg)
        print('deductive_coder0: ', self.deductive_coder0)
        print('global_value_map: ', self.global_value_map)
        
        
    
        
        
        
        
        