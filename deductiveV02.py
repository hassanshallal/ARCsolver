import z3
from z3 import Int, Ints, Real, Reals, Bool, Bools, BitVec, BitVecVal, Array, BitVecSort
from z3 import IntVector, RealVector, BoolVector
from z3 import IntSort, ForAll, simplify, pp, set_option
from z3 import Sum, sat, unsat, print_matrix, describe_tactics, describe_probes
from z3 import And, Or, Xor, Not, Implies, Distinct,  FailIf, Then, Exists, With, If, Store
from z3 import Optimize, solve, Solver, Function, Goal, Tactic, solve_using, Probe, Z3Exception, is_pattern

# z3 or SAT derived knowledge is deductive
# starts with a theory, confirms a hypothesis, tend to do quantitative operations
# Deductive mindset: Theory-->prediction-->experiment
# from general principle to special case

# z3 or SAT derived knowledge is inductive because it has to start with the data
# SAT treatment screens simple basic logical expressions against binary representations
# of the problem and after applying this type of logical screening on all the data, it
# should move in the direction of inferring conclusions such as most of the problems
# ARC solver would face can be described as one of 11 found categories, which by the way beaten
# our attempts to categorize into 4 then 5 then 6 cases, of course, our position after this foundation
# is much different than our position before that.


# The following provides a logical resource such as saver0 which allows the machine
# to unsupervisedly classify the problems into 11 different types based on the
# binary of the input against its background and the binary of the input against the Output
# This seems sort of deductive but its interesting

# This function finds logical operattors that represent a pixel change in regards to background and input-to-output transformation
def logical_screen_0(in_, out_):
    x, y = Bools('x y')
    x = True if in_ else False
    y = True if out_ else False

    chosen = []
    for m in [And(x, y), Or(x, y), Xor(x, y), Implies(x, y), Not(x), Not(y)]:
        s = Solver()
        s.add(Not(m))
        if s.check() == z3.z3.unsat:
            chosen.append(str(m).split('(')[0])
    return tuple(chosen)

# Coding: non-changing bg: ['Implies', 'Not', 'Not']
#             changing bg: ['Or', 'Xor', 'Implies', 'Not']
#             changing nonbg: ['And', 'Or', 'Implies']
#             non changing nonbg: ['Or', 'Xor', 'Not']

saver0 = {}
saver0[(True, True)] = logical_screen_0(True, True)
saver0[(True, False)] = logical_screen_0(True, False)
saver0[(False, True)] = logical_screen_0(False, True)
saver0[(False, False)] = logical_screen_0(False, False)

coder0 = {}
coder0[saver0[(True, True)]] = 'nonbg_C'
coder0[saver0[(True, False)]] = 'nonbg_S'
coder0[saver0[(False, True)]] = 'bg_C'
coder0[saver0[(False, False)]] = 'bg_S'

# in_mask = np.where(in_ != in_bg, True, False)
# out_mask = np.where(in_ != out_, True, False)
# out_extend_mask = np.where(in_ != out_ and in_ != in_bg and out_ != in_bg, True, False)
def logical_screen_1(in_, out_, out_extend_):  # out_extend:
    x, y, z = Bools('x y z')
    x = True if in_ else False
    y = True if out_ else False
    z = True if out_extend_ else False

    chosen = []
    for m in [And(x, y), Or(x, y), Xor(x, y), Implies(x, y), Not(x), Not(y), And(And(x, y), z), And(And(x, y), Not(z))]:
        s = Solver()
        s.add(Not(m))
        if s.check() == z3.z3.unsat:
            l = str(m).split('(')
            if len(l) == 2:
                chosen.append(l[0])
            elif len(l) > 2:
                if 'Not' in l[2]:
                    chosen.append(l[0]+'Not')
                else:
                    chosen.append(l[0]+'And')
    return tuple(chosen)


saver1 = {}
saver1[(True, True, True)] = logical_screen_1(True, True, True)
saver1[(True, True, False)] = logical_screen_1(True, True, False)
saver1[(True, False, False)] = logical_screen_1(True, False, False)
saver1[(False, True, False)] = logical_screen_1(False, True, False)
saver1[(False, False, False)] = logical_screen_1(False, False, False)

coder1 = {}
coder1[saver1[(True, True, True)]] = 'nonbg_C_nonbg'
coder1[saver1[(True, True, False)]] = 'nonbg_C_bg'
coder1[saver1[(True, False, False)]] = 'nonbg_S'
coder1[saver1[(False, True, False)]] = 'bg_C'
coder1[saver1[(False, False, False)]] = 'bg_S'
