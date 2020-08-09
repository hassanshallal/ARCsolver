from utils import *

nbh = lambda arr, i, j: {
    (ip, jp) : arr[i+ip, j+jp]
        for ip, jp in product([1, -1, 0], repeat=2)
            if 0 <= i+ip < arr.shape[0] and 0 <= j+jp < arr.shape[1]
}

nbh_conditional = lambda arr, i, j: {
    (i+ip, j+jp) : arr[i+ip, j+jp]
            for ip, jp in product([1, -1, 0], repeat=2)
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
    all_combs.add(tuple(nb_shifts)) # added 8

    for n in nb_shifts:
        all_combs.add((n, )) # added ones

    for n in range(2, 8): # between 2 and 8
        the_list = list(itertools.combinations(get_neighbour_shifts([1, -1, 0]), n))
        for x in the_list:
            all_combs.add(x)

    l1 = list(all_combs)
    l1 = [[]] + l1
    for x in range(len(l1)):
        if len(l1[x]) > 0:
            l1[x] = sorted([list(y) for y in l1[x]])
    nb_space = sorted(l1, key = len)
    assert len(nb_space) == 256
    return nb_space

nb_space = get_nb_location_space()

def neighbor_situation(arr, i, j):
    # nb_shifts is an element at an index in the nb_location_space nb_location_space[255] is the holistic 8 nb space
    this_location_space = nbh(arr, i, j)
    if (0, 0) in this_location_space.keys():
        del this_location_space[(0, 0)]
    location_output = {}
    for k, v in this_location_space.items():
        if v not in location_output.keys():
            location_output[v] = []
        location_output[v].append(list(k))
    trimmed_location_output = {}
    for k, v in location_output.items():
        trimmed_location_output[k] = sorted(v)
    for k, v in trimmed_location_output.items():
        trimmed_location_output[k] = nb_space.index(v)

    this_point = nbh_conditional(arr, i, j)
    if (i, j) in this_point.keys():
        del this_point[(i, j)]
    output = {}
    for k, v in this_point.items():
        if v not in output.keys():
            output[v] = []
        output[v].append(list(k))
    trimmed_output = {k: len(v) for k, v in output.items()}
    return location_output, trimmed_location_output, output, trimmed_output, len(trimmed_output)

# trimmed_location_output is the positioning of different neighbors on a space map
# the 266 space map may be exteded for 0-9 options, so, we end up with 2660 space
# how can represent this for each array point
# We need a code mapping the

def neighbor_situation_whole(maze):
    results = {}
    dim_0 = maze.shape[0]
    dim_1 = maze.shape[1]

    space_of_neighbors = np.empty((dim_0, dim_1), dtype = np.int64)
    num_of_diff_neighbors = np.empty((dim_0, dim_1), dtype = np.int64)
    most_common_neighbor = np.empty((dim_0, dim_1), dtype = np.int64)
    least_common_neighbor = np.empty((dim_0, dim_1), dtype = np.int64)


    for n in range(0, dim_0):
        for m in range(0, dim_1):
            results[(n, m), maze[n][m]] = neighbor_situation(maze, n, m)
            #print(results[(n, m), maze[n][m]])
            num_of_diff_neighbors[n, m] = results[(n, m), maze[n][m]][4]

            target_vals = list(results[(n, m), maze[n][m]][1].values())
            target_multiplier = len(results[(n, m), maze[n][m]][1])
            space_of_neighbors[n, m] = sum([x*target_multiplier for x in target_vals])

            max_value = max(results[(n, m), maze[n][m]][3], key=results[(n, m), maze[n][m]][3].get)
            most_common_neighbor[n,m ] = results[(n, m), maze[n][m]][1][max_value]
            min_value = min(results[(n, m), maze[n][m]][3], key=results[(n, m), maze[n][m]][3].get)
            least_common_neighbor[n,m ] = results[(n, m), maze[n][m]][1][min_value]

    return num_of_diff_neighbors, space_of_neighbors, most_common_neighbor, least_common_neighbor
