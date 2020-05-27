# should be able to get started with this very soon. may be or may be not.

from utils import *
from capturedV02 import *

# The following examine simple matrix rotations and mirroring
def get_diagonal_mirror(arr):
    arr = fix_dim(arr)
    return np.fliplr(np.transpose(arr[::-1]))

def get_offdiagonal_mirror(arr):
    arr = fix_dim(arr)
    return np.flipud(np.transpose(arr[::-1]))

# There may be some redundancie in the screen_flips_rotation function, no time to gather test cases
def screen_flips_rotation(in_, out_):
    in_ = fix_dim(in_)
    out_ = fix_dim(out_)

    # rotation is more precedent over flipud or fliplr
    if np.all(np.rot90(in_, 1, axes = (0, 1)) == out_):
        return np.rot90, 1
    elif np.all(np.rot90(in_, 2, axes = (0, 1)) == out_):
        return np.rot90, 2
    elif np.all(np.rot90(in_, 3, axes = (0, 1)) == out_):
        return np.rot90, 3
    elif np.all(get_diagonal_mirror(in_) == out_):
        return get_diagonal_mirror, None
    elif np.all(get_offdiagonal_mirror(in_) == out_):
        return get_offdiagonal_mirror, None
    elif np.all(np.flipud(in_) == out_):
        return np.flipud, None
    elif np.all(np.fliplr(in_) == np.array(out_)):
        return np.fliplr, None
    else:
        return False, None

def screen_flips(traininputs, trainoutputs, testinputs, testoutputs):
    are_flips = [screen_flips_rotation(in_, out_) for in_, out_ in zip(traininputs, trainoutputs)] # this took care of train validation
    if all([x[0] != False for x in are_flips]) and all([x[0] == are_flips[0][0] for x in are_flips]):
        if are_flips[0][1] != None:
            testpreds = [are_flips[0][0](x, are_flips[0][1]) for x in testinputs]
        else:
            testpreds = [are_flips[0][0](x) for x in testinputs]
        test_evaluated = all([np.array_equal(x, y) for x, y in zip(testpreds, testoutputs)])
        if test_evaluated:
            return 'solved', [(are_flips[0][0], are_flips[0][1])], testpreds
        elif not test_evaluated:
            return 'overfit but not solved', [(are_flips[0][0], are_flips[0][1])], testpreds
        else:
            return 'unsolved', [], []
    return 'unsolved', [], []


# assess_captured_holistic_testing(this_task_output, trainoutputs, testinputs, order_, testoutputs = None)
# we're here and we're trying to turn inductive into a class on its own and maintain a modified objevtive and
# a current prediction and solved/partially solved/unsolved status, work this out so that inductive
# can be further extended and elaborated.
def screen_captured(traininputs, trainoutputs, testinputs, objective_status, objective, asssignments_output, bg, token_to_colors, testoutputs = None):
    # Now, we need to establish the way of screening here
    # what
    if objective_status == None:
        a, b = assess_captured_holistic_training(traininputs, testinputs)
        # from a, b to x, y, z
        return 'unsolved', [], [] #x, y, z
    elif objective_status == 'obd' or  objective_status == 'red':
        a, b =  assess_captured_target_training(asssignments_output, objective, bg, traininputs, testinputs)
        # from a, b to x, y, z: # we need to create outputs out of the modified objective and the b (testinputs as sets)
        if len(b) > 0:
            objective_satisfiability = [len(x) == 3 or len(x) == 5 for x in b]
            if all(objective_satisfiability):
                # you must ouput a mechanism, a solution

                return 'solved', [], []
            elif any(objective_satisfiability):
                # you must output a modified objective and a partial solution

                return 'partially solved', b, []
            else:
                return 'unsolved', [], []
    elif objective_status == 'pred' or objective_status == 'irr':
        return 'unsolved', [], []
    else:
        return 'unsolved', [], []



# we don't pass self.testoutputs
# we recieve a, b: captured_situation_whole(traininput), captured_situation_whole(testinput)
#            a, b: modified_objective, captured_situation_whole(testinput)
# what do we want from inductive?
# if objst == None, return captured_situation_whole(traininput), captured_situation_whole(testinput)
    # here in taskmanager, first principles will have to be formulated drawing conclusions
    # of how to derive the output from the captured_situation_whole resuls:
    # first_principle_one = np.unique(captured_vqlues) == 1 and  np.unique(output) == 1 and np.unique(captured_vqlues) == np.unique(output): task 345 for example
    # the above is abstracted so as to solve other similar problems, make it convolved pr capable of convolving:
    # how to further convolve the above pricniple: add the option for uncaptured_vqlues, in essence, there is no reason why this concolution won't work?
    # SO, the job of inductive here is not really much about inference as much as it's about labelling points with values and coordinates as captured, form a line, with 6 neighbours, whatever?
# if objst == 'obd' or 'red': return modified_objective, captured_situation_whole(testinput):
    # taskmanager checks whether the modified_objective is satisfiable or solvable
    # if it is solvables, it applies it on the test input sets so as to arrive to the predicted outputs
    # Here is quite a different situation but similar in a sense as of maintaining the control over
    # inference
    # as a matter of reality, it doesn't matter. I'd rather delegate inference comletely to inductiveV02
    # inductive offers the flexibility of extention and must offer the logic of solving

# a, b must go throguh either a first principle screenings of values, numbers, dimensions, etc, return x, y, z
#        or tested for satisfiability, applied to testpinputs, evaluated and return x, y z

    # boolean == len(objective) == satisfied rather than visited
    # for n in objective:
    # if len(n) == 1: nonbg --> bg, fire on the current preds working progresses, mark satisfied
    # elif len(n) == 2: find which first pricniple can differentiate between the two sets
    # return 'can be solved', [], []

    # are_flips = [screen_flips_rotation(in_, out_) for in_, out_ in zip(traininputs, trainoutputs)]
    # if all([x[0] != False for x in are_flips]) and all([x[0] == are_flips[0][0] for x in are_flips]):
    #     if are_flips[0][1] != None:
    #         trainpreds = [are_flips[0][0](x, are_flips[0][1]) for x in traininputs]
    #         testpreds = [are_flips[0][0](x, are_flips[0][1]) for x in testinputs]
    #     else:
    #         trainpreds = [are_flips[0][0](x) for x in traininputs]
    #         testpreds = [are_flips[0][0](x) for x in testinputs]
    #
    #     train_validated = all([np.array_equal(x, y) for x, y in zip(trainpreds, trainoutputs)])
    #     test_evaluated = all([np.array_equal(x, y) for x, y in zip(testpreds, testoutputs)])
    #     if train_validated and test_evaluated:
    #         return 'solved', [(are_flips[0][0], are_flips[0][1])], testpreds
    #     elif train_validated and not test_evaluated:
    #         return 'overfit but not solved', [(are_flips[0][0], are_flips[0][1])], testpreds
    #     elif not train_validated:
    #         return 'unsolved', [], []
    # return 'unsolved', [], []
