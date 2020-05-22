# should be able to get started with this very soon. may be or may be not.

from utils import *

# we will apply A-star in order to determine whether a cell is trapped with no access to edges or not
def captured_astar(grid, x, y, bg, all_):
    # print('investigatin: ', (x, y))
    if (x == 0 or y == 0 or x == grid.shape[0]-1 or y == grid.shape[1]-1) and grid[x][y] == bg:
        # print('found edge at %d,%d' % (x, y))
        return True
    elif grid[x][y] == -1:
        # print('visited at %d,%d' % (x, y))
        return False
    elif grid[x][y] != bg:
        # print('wall at %d,%d' % (x, y))
        return False


    # mark as visited
    grid[x][y] = -1
    # print(grid)
    # explore neighbors clockwise starting by the one on the right
    if all_ == 'ver_hor':
        if ((x < grid.shape[0]-1 and captured_astar(grid, x+1, y, bg, 'ver_hor'))
            or (y > 0 and captured_astar(grid, x, y-1, bg, 'ver_hor'))
            or (x > 0 and captured_astar(grid, x-1, y, bg, 'ver_hor'))
            or (y < grid.shape[1]-1 and captured_astar(grid, x, y+1, bg, 'ver_hor'))):
            return True
        return (False)

    elif all_ == 'diag':
        if ((x > 0 and y > 0 and captured_astar(grid, x-1, y-1, bg, 'diag'))
            or (x < grid.shape[0]-1 and y < grid.shape[1]-1 and captured_astar(grid, x+1, y+1, bg, 'diag'))
            or (x > 0 and y < grid.shape[1]-1 and captured_astar(grid, x-1, y+1, bg, 'diag'))
            or (x < grid.shape[0]-1 and y > 0 and captured_astar(grid, x+1, y-1, bg, 'diag'))):
            return True
        return (False)

    elif all_ == 'all':
        if ((x < grid.shape[0]-1 and captured_astar(grid, x+1, y, bg, 'all'))
            or (y > 0 and captured_astar(grid, x, y-1, bg, 'all'))
            or (x > 0 and captured_astar(grid, x-1, y, bg, 'all'))
            or (y < grid.shape[1]-1 and captured_astar(grid, x, y+1, bg, 'all'))

            or (x > 0 and y > 0 and captured_astar(grid, x-1, y-1, bg, 'all'))
            or (x < grid.shape[0]-1 and y < grid.shape[1]-1 and captured_astar(grid, x+1, y+1, bg, 'all'))
            or (x > 0 and y < grid.shape[1]-1 and captured_astar(grid, x-1, y+1, bg, 'all'))
            or (x < grid.shape[0]-1 and y > 0 and captured_astar(grid, x+1, y-1, bg, 'all'))):
            return True
        return (False)


# For reasons I don't have time to investigate, we need to wrap up the astar algorithm in order to get it to work
def get_captured(arr, n, m, bg, all_):
    # print('from get_captured n is: ', str(n), ' and m is: ', str(m))
    if captured_astar(arr, n, m, bg, all_):
        return False
    else:
        return True

def captured_situation(maze, test_points, all_):
    test_points = test_points.tolist()
    bg = get_background(maze)
    captured = {}
    captured[1] = []
    captured[0] = []
    for n in range(1, maze.shape[0]-1):
        for m in range(1, maze.shape[1]-1):
            arr = maze.copy()
            if [n, m] in test_points:
                if arr[n][m] != bg:
                    arr[n][m] = bg
                if get_captured(arr, n, m, bg, all_):
                    captured[1].append((n, m))
                else:
                    captured[0].append((n, m))
    if len(captured[1]) == len(test_points):
        return True
    else:
        return False

def captured_situation_whole(maze, all_):
    bg = get_background(maze)
    #print(bg)
    captured = {}
    captured[1] = []
    captured[0] = []
    for n in range(1, maze.shape[0]-1):
        for m in range(1, maze.shape[1]-1):
            # print('from captured situation n is: ', str(n), ' and m is: ', str(m))
            arr = maze.copy()
            if arr[n][m] != bg: # in order to operate also on non-bg cells
                arr[n][m] = bg
            if get_captured(arr, n, m, bg, all_):
                captured[1].append([maze[n][m], (n, m)])
            else:
                captured[0].append([maze[n][m], (n, m)])
    return captured
