import tensorflow as tf


def marginal_utility(reward_vector):
    first_vals = tf.slice(reward_vector, [0, 0], [tf.shape(reward_vector)[0], 1])

    first_tiled = tf.tile(tf.expand_dims(first_vals, 1), multiples=[1, tf.shape(reward_vector)[1],
                                                                    tf.shape(reward_vector)[1]])
    first_tiled = tf.linalg.band_part(first_tiled, 0, -1)

    reward_tiled = tf.tile(tf.expand_dims(reward_vector, 1),
                           multiples=[1, tf.shape(reward_vector)[1], 1])

    future_points = tf.linalg.band_part(reward_tiled, 0, -1)
    ordered_points = reward_tiled - future_points + first_tiled

    max_points = tf.stop_gradient(tf.reduce_max(ordered_points, axis=2))

    marg_utility = reward_vector - max_points
    marg_utility = tf.clip_by_value(tf.slice(marg_utility, [0, 1], [tf.shape(reward_vector)[0],
                                                                    tf.shape(reward_vector)[1] - 1]),
                                    clip_value_min=0,
                                    clip_value_max=tf.reduce_max(marg_utility))
    marg_utility = tf.concat([first_vals, marg_utility], axis=1)

    return tf.cast(tf.reduce_mean(tf.reduce_sum(marg_utility, axis=1)), dtype=tf.float64)
