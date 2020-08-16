from untokenized import *

def get_edge_access_space(l):
    all_combs = set()
    for n in l:
        all_combs.add((n, )) # singular edges

    for n in range(2, 4): # between 2 and 8
        the_list = list(itertools.combinations(l, n))
        for x in the_list:
            all_combs.add(x)

    all_combs.add(tuple(l))
    l1 = list(all_combs)
    l1 = [[]] + l1
    for x in range(len(l1)):
        l1[x] = list(l1[x])
        if len(l1[x]) > 0:
            l1[x] = sorted(l1[x])
    edge_space = sorted(l1, key = len)
    assert len(edge_space) == 16
    return edge_space

def get_individual_edges(maze):
        dim_0 = maze.shape[0]
        dim_1 = maze.shape[1]
        left_list = []
        right_list = []
        top_list = []
        bottom_list = []

        # add corners manually:
        left_list.append([maze[0][0], (0, 0)])
        top_list.append([maze[0][0], (0, 0)])

        left_list.append([maze[dim_0-1][0], (dim_0-1, 0)])
        bottom_list.append([maze[dim_0-1][0], (dim_0-1, 0)])


        right_list.append([maze[0][dim_1-1], (0, dim_1-1)])
        top_list.append([maze[0][dim_1-1], (0, dim_1-1)])

        right_list.append([maze[dim_0-1][dim_1-1], (dim_0-1, dim_1-1)])
        bottom_list.append([maze[dim_0-1][dim_1-1], (dim_0-1, dim_1-1)])


        for n in range(dim_0):
            for m in range(dim_1):
                if n == 0 and [maze[n][m], (n, m)] not in top_list:
                    top_list.append([maze[n][m], (n, m)])
                elif n == dim_0-1 and [maze[n][m], (n, m)] not in bottom_list:
                    bottom_list.append([maze[n][m], (n, m)])
                elif m == 0 and [maze[n][m], (n, m)] not in left_list:
                    left_list.append([maze[n][m], (n, m)])
                elif m == dim_1-1 and [maze[n][m], (n, m)] not in right_list:
                    right_list.append([maze[n][m], (n, m)])


        return [left_list, top_list, right_list, bottom_list]

def search(x, y, grid, bg):
    dim_0 = grid.shape[0]
    dim_1 = grid.shape[1]

    if grid[x][y] == 11:
        #print( 'found edge at %d,%d' % (x, y))
        return True
    elif grid[x][y] == -1:
        #print('visited at %d,%d' % (x, y))
        return False
    elif grid[x][y] != bg:
        #print('wall at %d,%d' % (x, y))
        return False

    #print( 'visiting %d,%d' % (x, y))
    # mark as visited
    grid[x][y] = -1

    # explore neighbors clockwise starting by the one on the right
    if ((x < dim_0-1 and search(x+1, y, grid, bg))
        or (y > 0 and search(x, y-1, grid, bg))
        or (x > 0 and search(x-1, y, grid, bg))
        or (y < dim_1-1 and search(x, y+1, grid, bg))):
        return True
    return False

edge_space = get_edge_access_space([0, 1, 2, 3])

def get_edge_situation(grid, bg):

    dim_0 = grid.shape[0]
    dim_1 = grid.shape[1]

    edge_situation = np.empty((dim_0, dim_1), dtype = int)
    target_points = get_individual_edges(grid)

    for n in range(dim_0):
        for m in range(dim_1):
            #print(n, m)

            edges_connected = set()
            for x in range(len(target_points)):
                current_side = deepcopy(target_points[x])
                for y in range(len(current_side)):
                    if current_side[y][1][0] == n and current_side[y][1][1] == m:
                        edges_connected.add(x)
                        break

                    else:
                        result = None
                        if current_side[y][0] == bg:
                            this_grid = deepcopy(grid)

                            this_grid[current_side[y][1][0], current_side[y][1][1]] = 11
                            this_grid[n, m] = bg

                            #print(this_grid)
                            result = search(n, m, this_grid, bg)

                            if result:
                                edges_connected.add(x)
                                break

            edges_connected = sorted(edges_connected)
            edge_situation[n, m] = edge_space.index(edges_connected)

    return edge_situation
