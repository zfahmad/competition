import tensorflow as tf
import numpy as np
import network as nn
import utils.generate_map as gm
import utils.calcs as clc
import loss



# def create_dataset(file_names_src, batch_size):
#     dataset_file_names = tf.io.match_filenames_once(file_names_src)
#     dataset = tf.data.TFRecordDataset(dataset_file_names).shuffle(batch_size*2)
#     dataset = dataset.map(logger.parser).batch(batch_size, drop_remainder=True).repeat(10)
#     return dataset


batch_size = 4
dim = 5
arms = 2
model = nn.ResidualNetwork(dim, arms=2)
# value_loss_function = tf.keras.losses.MSE
# policy_loss_function = tf.keras.losses.categorical_crossentropy
optimizer = tf.optimizers.Adam(1e-4)
model.build(input_shape=(batch_size, dim, dim, 1))
# lmbda = 0.0


# dataset = create_dataset('/Users/zaheen/Documents/games/game_*.tfrecord', batch_size)
# model_file = '/Users/zaheen/projects/chinese-checkers/python/models/mod_0'
# step = 0

# print_list = ['Step', 'Loss', 'V Loss', 'P Loss']
# print(36 * "=")
# print(' {:^7} {:^8} {:^8} {:^8}'.format(*print_list))
# print(36 * "=")

for step in range(1):
    with tf.GradientTape() as tape:
        utilities = gm.generate_map(batch_size, dim, dim)
        inputs = tf.expand_dims(utilities, axis=3)
        policies = model(inputs, True)
        samples, selection = clc.sample_actions(policies, batch_size, dim, arms)
        # print(samples)
        # print(selection)
        shares = clc.calc_shares(samples, 3, 0, dim, arms, batch_size)
        rewards = clc.calc_rewards(utilities, shares)
        probs = tf.transpose(tf.cast(tf.reduce_sum(policies * selection, axis=3), dtype=tf.float64), [0, 2, 1])

        expected_utilities = tf.reduce_sum(rewards * probs, axis=2)

        mu = loss.marginal_utility(expected_utilities)
        expected_mr = tf.reduce_mean(tf.reduce_sum(mu, axis=1))
        print(expected_mr)
        grads = tape.gradient(-expected_mr, model.trainable_variables)
        # optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # if not step % 500:
        #     print('Step: {} Loss: {}'.format(step, expected_mr))

#     with tf.GradientTape() as tape:
#         y, policy = model(inputs, True)
#
#         sample_dist = sample_dist / tf.reduce_sum(sample_dist, axis=[1], keepdims=True)
#         value_loss = tf.reduce_mean(value_loss_function(outcome, y))
#         # print(outcome)
#         # print(y)
#         policy_loss = tf.reduce_mean(policy_loss_function(sample_dist, policy))
#         loss = value_loss + policy_loss
#
#         grads = tape.gradient(loss, model.trainable_variables)
#         optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
#     if not step % 500:
#         print(" {:>7} {:>8.4f} {:>8.4f} {:>8.4f}".format(step, loss, value_loss, policy_loss))
#         # print(y)
#         model.save_weights(filepath=model_file + '.h5')
#
#     step += 1
#
# print(" {:>7} {:>8.4f} {:>8.4f} {:>8.4f}".format(step, loss, value_loss, policy_loss))
# model.save_weights(filepath=model_file + '.h5')
