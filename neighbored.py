
from utils import *
from communicate import *
from cages import *


nbh_conditional = lambda arr, i, j, nb_shifts: {
    (i+ip, j+jp) : arr[i+ip, j+jp]
        for x in nb_shifts
            for ip, jp in x
                if (0 <= i < arr.shape[0]) and (0 <= j < arr.shape[1]) and (0 <= i+ip < arr.shape[0]) and (0 <= j+jp < arr.shape[1])
}

def get_neighbour_shifts(l):
    shifts = []
    for ip, jp in itertools.product([*l], repeat=2):
        shifts.append((ip, jp))
    shifts = [x for x in shifts if x != (0, 0)]
    return shifts

def get_nb_location_space():
    nb_shifts = get_neighbour_shifts([1, -1, 0])
    all_combs = set()
    all_combs.add(tuple(nb_shifts))
    for n in nb_shifts:
        all_combs.add((n, ))

    for n in range(2, 8):
        the_list = list(itertools.combinations(get_neighbour_shifts([1, -1, 0]), n))
        for x in the_list:
            all_combs.add(x)

    l1 = list(all_combs)
    l1 = [list(x) for x in l1]
    nb_space = sorted(l1, key = len)
    assert len(nb_space) == 255
    return nb_space

class Neighbors:
    def __init__(self, communication, running_objective, cur_test_preds):
        self.communication = communication

        # print(self.communication.dimension_status, self.communication.output_dim_preds)
        self.running_objective = running_objective
        self.cur_test_preds = cur_test_preds

        # control the space in regards of number, locarion, and type of neighbours
        self.nb_num_space = [x for x in range(9)]
        self.nb_location_space = get_nb_location_space()
        self.nb_types_space = []

    def assess_point_neighbors(self, arr, i, j,  nb_shifts):
        # nb_shifts is an element at an index in the nb_location_space nb_location_space[254] is the holistic 8 nb space

        this_point = nbh_conditional(arr, i, j, nb_shifts)
        if (i, j) in this_point.keys():
            del this_point[(i, j)]

        output = {}
        for k, v in this_point.items():
            if v not in output.keys():
                output[v] = []
            output[v].append(k)
        return output, {k: len(v) for k, v in output.items()}
