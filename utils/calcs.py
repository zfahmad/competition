import numpy as np
import tensorflow as tf
import utils.generate_map as gm
import tensorflow_probability as tfp


def manhattan_distance(pos_1, pos_2):
    distance = 0

    for x, y in zip(pos_1, pos_2):
        distance += abs(x - y)

    return distance


def partitioning(dim_x, dim_y, points, split):
    partition = np.zeros([dim_x, dim_y])

    for x in range(dim_x):

        for y in range(dim_y):

            min_dist = float('inf')
            min_ind = []

            for ind, point in enumerate(points):
                dist = manhattan_distance([x, y], point)

                if dist < min_dist:
                    min_dist = dist
                    min_ind = [ind]
                elif dist == min_dist:
                    min_ind.append(ind)
                for ind in min_ind:
                    if ind > (split - 1):
                        partition[y, x] += 1
                    else:
                        partition[y, x] += 0

                print(x, ", ", y, ": ", manhattan_distance([x, y], point), " ", partition[y, x], " ", min_ind)
                partition[y, x] /= len(min_ind)

    return partition


def sample_actions(batch_policies, batch_size, dim):
    dists = tfp.distributions.Multinomial(total_count=1, probs=batch_policies)
    selection = dists.sample(1)
    samples = tf.where(tf.reshape(selection, [batch_size, 3, dim, dim]))
    return tf.reshape(tf.slice(samples, [0, 2], [batch_size*3, 2]), [batch_size, 3, 2]), tf.squeeze(selection)


def calc_shares(pos, num_p1, num_p2, dim, batch_size):
    _x = tf.tile(tf.expand_dims(tf.constant(np.reshape(np.arange(dim), (dim, 1, 1)),
                                            dtype=tf.int64), 0), [batch_size, 1, 1, 1])
    _y = tf.tile(tf.expand_dims(tf.constant(np.reshape(np.arange(dim), (1, dim, 1)),
                                            dtype=tf.int64), 0), [batch_size, 1, 1, 1])

    x = tf.abs(tf.reshape(tf.slice(pos, [0, 0, 0], [batch_size, num_p1 + num_p2, 1]), (batch_size, 1, 1, -1)) - _x)

    y = tf.abs(tf.reshape(tf.slice(pos, [0, 0, 1], [batch_size, num_p1 + num_p2, 1]), (batch_size, 1, 1, -1)) - _y)

    distances = x + y
    min_distances = tf.reduce_min(distances, axis=3, keepdims=True)
    diff_distances = distances - min_distances
    shares = tf.cast(tf.equal(diff_distances, 0), dtype=tf.int32)
    shares = shares / tf.reduce_sum(shares, axis=3, keepdims=True)

    return tf.cast(shares, dtype=tf.float64)


def calc_rewards(utilities, shares):
    rewards = shares * tf.expand_dims(utilities, axis=3)
    return tf.reduce_sum(rewards, axis=[1, 2])


if __name__ == '__main__':

    utilities = gm.generate_map(2, 10, 10)

    p1_pos = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    p2_pos = tf.constant([[[9, 0], [1, 5]], [[8, 2], [7, 3]]])

    pos = tf.concat([p1_pos, p2_pos], axis=1)
    print(pos)
    # shares = calc_shares(pos, 2, 2, 10, 2)
    #
    # rewards = shares * tf.expand_dims(utilities, axis=3)
    # print(tf.reduce_sum(rewards, axis=[1, 2]))
