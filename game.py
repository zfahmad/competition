import os
import numpy as np
import tensorflow as tf
import utils.generate_map as gm
import utils.calcs as calcs
import matplotlib.pyplot as plt


class Game:
    def __init__(self, dim_x=10, dim_y=10):
        self.gen_map = gm.generate_map(1, dim_x, dim_y)
        self.dim_x = dim_x
        self.dim_y = dim_y
        self.num_p1 = 3
        self.num_p2 = 1

    def display_map(self, p1_pos, p2_pos):
        plt.matshow(tf.squeeze(self.gen_map), cmap=plt.get_cmap('summer_r'))
        plt.colorbar()

        plt.scatter(p1_pos[:, 0], p1_pos[:, 1], s=45, color='turquoise', edgecolors='black')
        plt.scatter(p2_pos[:, 0], p2_pos[:, 1], s=45, color='red', edgecolors='black')

        plt.xticks(range(10), range(1, 11))
        plt.yticks(range(10), range(1, 11))
        plt.grid(True, which='minor')
        plt.show()

    def display_partition(self, p1_pos, p2_pos):
        partition = calcs.partitioning(self.dim_x, self.dim_y, np.vstack((p1_pos, p2_pos)), self.num_p1)
        print(partition)
        plt.matshow(partition, cmap=plt.get_cmap('YlGn'))
        plt.colorbar()
        plt.scatter(p1_pos[:, 0], p1_pos[:, 1], s=45, color='turquoise', edgecolors='black')
        plt.scatter(p2_pos[:, 0], p2_pos[:, 1], s=45, color='red', edgecolors='black')

        plt.xticks(range(10), range(1, 11))
        plt.yticks(range(10), range(1, 11))
        plt.grid(True, which='minor')
        plt.show()


if __name__ == "__main__":
    game = Game()
    p1_pos = np.random.randint(low=0, high=9, size=(3, 2))  # np.array([[3, 5], [5, 5]])
    p2_pos = np.random.randint(low=0, high=9, size=(1, 2))  # np.array([[5, 7]])
    game.display_map(p1_pos, p2_pos)
    game.display_partition(p1_pos, p2_pos)
