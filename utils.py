import tensorflow as tf
import itertools
import tensorflow_probability as tfp
import numpy as np


ALPHA = tf.constant(3.0)
BETA = tf.constant(1.0)


def generate_map(batch_size, dim_x=10, dim_y=10):
    dist = tfp.distributions.InverseGamma(ALPHA, BETA)
    gmap = dist.sample([batch_size, dim_x, dim_y])
    z = tf.reduce_sum(gmap, axis=[1, 2], keepdims=True)
    normalized_gmap = gmap / z

    return tf.cast(normalized_gmap, dtype=tf.float64)

def sample_opponent(quantals, batch_size, dim):
    dists = tfp.distributions.Multinomial(total_count=1, probs=quantals)
    selection = dists.sample(1)
    samples = tf.where(tf.reshape(selection, [batch_size, dim, dim]))
    return tf.reshape(tf.slice(samples, [0, 1], [batch_size, 2]), [batch_size, 2]), tf.squeeze(selection)


def sample_actions(batch_policies, batch_size, num_locs, arms, dim):
    dists = tfp.distributions.Multinomial(total_count=1, probs=batch_policies)
    selection = tf.squeeze(dists.sample(1), axis=0)
    samples = tf.where(tf.reshape(selection, [batch_size, num_locs, arms, dim, dim]))
    return tf.reshape(tf.slice(samples, [0, 3], [batch_size*num_locs*arms, 2]),
                      [batch_size, num_locs, arms, 2]), selection


def calc_shares(pos, num_p1, num_p2, dim, arms, batch_size):
    _x = tf.tile(tf.expand_dims(tf.constant(np.reshape(np.arange(dim), (dim, 1, 1, 1)),
                                            dtype=tf.int64), 0), [batch_size, 1, 1, 1, 1])
    _y = tf.tile(tf.expand_dims(tf.constant(np.reshape(np.arange(dim), (1, dim, 1, 1)),
                                            dtype=tf.int64), 0), [batch_size, 1, 1, 1, 1])

    x = tf.abs(tf.reshape(tf.slice(pos, [0, 0, 0, 0], [batch_size, num_p1 + num_p2, arms, 1]),
                          (batch_size, 1, 1, -1, arms)) - _x)

    y = tf.abs(tf.reshape(tf.slice(pos, [0, 0, 0, 1], [batch_size, num_p1 + num_p2, arms, 1]),
                          (batch_size, 1, 1, -1, arms)) - _y)

    distances = x + y
    min_distances = tf.reduce_min(distances, axis=3, keepdims=True)
    diff_distances = distances - min_distances
    shares = tf.cast(tf.equal(diff_distances, 0), dtype=tf.int32)
    shares = shares / tf.reduce_sum(shares, axis=3, keepdims=True)

    return tf.slice(tf.cast(shares, dtype=tf.float64), [0, 0, 0, 0, 0], [batch_size, dim, dim, num_p1, arms])


def calc_rewards(utilities, shares, num_locs, arms):
    utilities = tf.tile(tf.expand_dims(utilities, axis=3), [1, 1, 1, num_locs])
    utilities = tf.tile(tf.expand_dims(utilities, axis=4), [1, 1, 1, 1, arms])
    rewards = shares * utilities
    return tf.reduce_sum(rewards, axis=[1, 2])


def sgpe(pop_map, dim, num_locs, op_num_locs):
    coordinates = itertools.product(range(dim), repeat=2*num_locs)

    pop_map = tf.expand_dims(pop_map, axis=0)

    op_locs = []
    utilities_copy = tf.reshape(pop_map, [1, -1])

    for j in range(op_num_locs):
        indmax = tf.argmax(utilities_copy, axis=1)
        opponent = tf.expand_dims(tf.stack([tf.math.floordiv(indmax, dim), tf.math.floormod(indmax, dim)], axis=1), axis=1)
        opponent = tf.tile(tf.expand_dims(opponent, axis=2), [1, 1, 1, 1])
        op_locs.append(opponent)
        utilities_copy = utilities_copy - (utilities_copy * tf.cast(utilities_copy == tf.reduce_max(utilities_copy, axis=1, keepdims=True), dtype=tf.float64))

    op_locs = tf.concat(op_locs, axis=1)

    max_return = 0

    for x_y in coordinates:
        # Get coordinates and opponent
        locs = [x_y[(i*2):(i*2)+2] for i in range(num_locs)]
        samples = tf.cast(tf.expand_dims(tf.expand_dims(tf.stack(locs, axis=0), axis=0), axis=2), tf.int64)
        pos = tf.concat([samples, op_locs], axis=1)

        # Calculate shares

        shares = calc_shares(pos, num_locs, op_num_locs, dim, 1, 1)
        rewards = tf.reduce_sum(calc_rewards(pop_map, shares, num_locs, 1)).numpy()

        if rewards > max_return:
            max_return = rewards
            best = samples

    return max_return, best.numpy(), op_locs.numpy()
