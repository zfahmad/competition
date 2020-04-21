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


def sample_opponent(quantals, batch_size, dim):
    dists = tfp.distributions.Multinomial(total_count=1, probs=quantals)
    selection = dists.sample(1)
    samples = tf.where(tf.reshape(selection, [batch_size, dim, dim])==1)
    return tf.reshape(tf.slice(samples, [0, 1], [batch_size, 2]), [batch_size, 2]), tf.squeeze(selection)


def sample_actions(batch_policies, batch_size, num_locs, arms, dim):
    dists = tfp.distributions.Multinomial(total_count=1, logits=batch_policies)
    selection = dists.sample(1)
    # print(samples.size, batch_size*num_locs*arms)
    # print(selection)
    samples = tf.where(tf.reshape(selection, [batch_size, num_locs, arms, dim, dim])==1)
    # print(samples, batch_size * num_locs * arms)
    return tf.reshape(tf.slice(samples, [0, 3], [batch_size*num_locs*arms, 2]),
                      [batch_size, arms, num_locs, 2]), tf.squeeze(selection)


def calc_shares(pos, num_p1, num_p2, dim, arms, batch_size):
    _x = tf.tile(tf.expand_dims(tf.constant(np.reshape(np.arange(dim), (dim, 1, 1, 1)),
                                            dtype=tf.int64), 0), [batch_size, 1, 1, 1, 1])
    _y = tf.tile(tf.expand_dims(tf.constant(np.reshape(np.arange(dim), (1, dim, 1, 1)),
                                            dtype=tf.int64), 0), [batch_size, 1, 1, 1, 1])

    x = tf.abs(tf.reshape(tf.slice(pos, [0, 0, 0, 0], [batch_size, arms, num_p1 + num_p2, 1]), (batch_size, 1, 1, arms, -1)) - _x)

    y = tf.abs(tf.reshape(tf.slice(pos, [0, 0, 0, 1], [batch_size, arms, num_p1 + num_p2, 1]), (batch_size, 1, 1, arms, -1)) - _y)

    distances = x + y
    min_distances = tf.reduce_min(distances, axis=4, keepdims=True)
    diff_distances = distances - min_distances
    shares = tf.cast(tf.equal(diff_distances, 0), dtype=tf.int32)
    shares = shares / tf.reduce_sum(shares, axis=4, keepdims=True)

    # return tf.cast(shares, dtype=tf.float64)
    return tf.slice(tf.cast(shares, dtype=tf.float64), [0, 0, 0, 0, 0], [batch_size, dim, dim, arms, num_p1])


def calc_rewards(utilities, shares, arms):
    utilities = tf.tile(tf.expand_dims(utilities, axis=3), [1, 1, 1, arms])
    utilities = tf.tile(tf.expand_dims(utilities, axis=4), [1, 1, 1, 1, 1])
    rewards = shares * utilities
    return tf.reduce_sum(rewards, axis=[1, 2])


if __name__ == '__main__':

    # utilities = gm.generate_map(2, 10, 10)
    policies = tf.constant([[[[0.1, 0.2, 0.1, 0.05, 0.2, 0.05, 0.1, 0.05, 0.15],
                              [0.1, 0.2, 0.1, 0.05, 0.2, 0.05, 0.1, 0.05, 0.15]],
                             [[0.1, 0.2, 0.1, 0.05, 0.2, 0.05, 0.1, 0.05, 0.15],
                              [0.1, 0.2, 0.1, 0.05, 0.2, 0.05, 0.1, 0.05, 0.15]]],
                            [[[0.1, 0.2, 0.1, 0.05, 0.2, 0.05, 0.1, 0.05, 0.15],
                              [0.1, 0.2, 0.1, 0.05, 0.2, 0.05, 0.1, 0.05, 0.15]],
                             [[0.1, 0.2, 0.1, 0.05, 0.2, 0.05, 0.1, 0.05, 0.15],
                              [0.1, 0.2, 0.1, 0.05, 0.2, 0.05, 0.1, 0.05, 0.15]]]], dtype=tf.float64)

    selection = tf.constant([[[[0, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, 0, 0]],
                              [[0, 0, 0, 0, 0, 0, 1, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0, 0]]],
                             [[[0, 1, 0, 0, 0, 0, 0, 0, 0],
                               [0, 0, 0, 0, 0, 0, 1, 0, 0]],
                              [[0, 0, 0, 0, 0, 0, 1, 0, 0],
                               [0, 1, 0, 0, 0, 0, 0, 0, 0]]]], dtype=tf.float64)

    utilities = tf.constant([[[2, 1, 3], [4, 5, 1], [2, 1, 4]], [[2, 1, 3], [4, 5, 1], [2, 1, 4]]], dtype=tf.double)
    print(utilities)
    # p1_pos = tf.constant([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
    # p2_pos = tf.constant([[[9, 0], [1, 5]], [[8, 2], [7, 3]]])

    p1_pos = tf.constant([[[[0, 1]], [[2, 0]]], [[[0, 1]], [[2, 0]]]], dtype=tf.int64)
    p2_pos = tf.constant([[[[2, 0]], [[0, 1]]], [[[2, 0]], [[0, 1]]]], dtype=tf.int64)

    # p1_pos = tf.constant([[[[0, 1]]], [[[0, 1]]]], dtype=tf.int64)
    # p2_pos = tf.constant([[[[2, 0]]], [[[2, 0]]]], dtype=tf.int64)

    pos = tf.concat([p1_pos, p2_pos], axis=2)
    print(pos)
    shares = calc_shares(pos, 2, 0, 3, 2, 2)
    print(shares)
    rewards = calc_rewards(utilities, tf.cast(shares, dtype=tf.double), 2)
    print(rewards)

    probs = tf.transpose(tf.cast(tf.reduce_sum(policies * selection, axis=3), dtype=tf.float64), [0, 2, 1])
    print(probs)

    expected_utilities = tf.reduce_sum(rewards * probs, axis=2)
    print(expected_utilities)
    #
    # rewards = shares * tf.expand_dims(utilities, axis=3)
    # print(tf.reduce_sum(rewards, axis=[1, 2]))
