#! /usr/env/bin python3

import os
import netpbmfile as npbm
import numpy as np
import matplotlib.pyplot as plt

class MazeEnv():
    def __init__(self, maze_size, outer_wall_gap, inner_wall_gap, wall_thickness):

        t = np.zeros((maze_size, maze_size), dtype = np.uint16)
        SIZE = maze_size

        partition = SIZE//6
        outer_partition = SIZE - (SIZE//6)
        inner_partition = SIZE - (SIZE//3)

        start_position = (500,800)
        goal_position = (500,500)


        # building border walls

        t[0:SIZE, 0:wall_thickness] = 1
        t[0:wall_thickness, 0:SIZE] = 1
        t[SIZE-wall_thickness:SIZE, 0:SIZE] = 1
        t[0:SIZE, SIZE-wall_thickness:SIZE] = 1

        # building outer walls
        t[partition:outer_partition, partition:partition+wall_thickness] = 1
        t[partition:partition+wall_thickness, partition:outer_partition] = 1
        t[partition:outer_partition, outer_partition-wall_thickness:outer_partition] = 1
        t[outer_partition-wall_thickness:outer_partition, partition:outer_partition] = 1

        # building inner walls
        t[partition*2:inner_partition, partition*2:partition*2+wall_thickness] = 1
        t[partition*2:partition*2+wall_thickness, partition*2:inner_partition] = 1
        t[partition*2:inner_partition, inner_partition-wall_thickness:inner_partition] = 1
        t[inner_partition-wall_thickness:inner_partition, partition*2:inner_partition] = 1

        # outer wall gaps

        t[SIZE//2-outer_wall_gap//2:SIZE//2+outer_wall_gap//2, partition:partition+wall_thickness] = 0
        t[SIZE//2-outer_wall_gap//2:SIZE//2+outer_wall_gap//2, outer_partition-wall_thickness:outer_partition] = 0

        # inner wall gaps

        t[partition*2:partition*2+wall_thickness, SIZE//2-inner_wall_gap:SIZE//2+inner_wall_gap] = 0
        t[inner_partition-wall_thickness:inner_partition, SIZE//2-inner_wall_gap:SIZE//2+inner_wall_gap] = 0

        self.t = t

    def throw_array(self):
        return self.t

        # Marking start and goal state

        # t[start_position] = 1
        # t[goal_position] = 1


        # plt.imshow(t)
        # plt.show()

def save_as_pbm(img, name, dest='./'):

    with open(os.path.join(dest,name), 'wb') as F:
        F.write("P4\n%i %i\n" % img.shape[::-1])
        numpy.packbits(img, axis=-1).tofile(F)


if __name__ == '__main__':
    env = MazeEnv(1000, 200, 100, 12)
    arr = env.throw_array()

    npbm.imwrite('original_maze.pbm', arr)





