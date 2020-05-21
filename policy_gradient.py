import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import matplotlib.pyplot as plt
import losses as lf
import networks
import utils


tf.executing_eagerly()

batch_size = 32
dim = 10
arms = 8
num_locs = 3
op_num_locs = 2
model = networks.SimplePolicyNetwork(dim, arms=arms, num_locs=num_locs)
quantal_temp = 1
optimizer = tf.optimizers.Adam(1e-5)
model.build(input_shape=(batch_size, dim, dim, 1))
iterations = 80000
interval = 5000

file_name = '10_10_pg'


regrets = []
regrets_std = []
losses = []

for step in range(iterations):
    with tf.GradientTape() as tape:
        utilities = utils.generate_map(batch_size, dim, dim)
        inputs = tf.expand_dims(utilities, axis=3)
        policies = model(inputs, True)
        policies = tf.tile(tf.expand_dims(policies, axis=2), [1, 1, arms, 1])
        samples, selection = utils.sample_actions(policies, batch_size, num_locs, arms, dim)

        op_locs = []
        utilities_copy = tf.reshape(utilities, [batch_size, -1])

        for j in range(op_num_locs):
            indmax = tf.argmax(utilities_copy, axis=1)
            opponent = tf.expand_dims(tf.stack([tf.math.floordiv(indmax, dim), tf.math.floormod(indmax, dim)], axis=1), axis=1)
            opponent = tf.tile(tf.expand_dims(opponent, axis=2), [1, 1, arms, 1])
            op_locs.append(opponent)
            utilities_copy = utilities_copy - (utilities_copy * tf.cast(utilities_copy == tf.reduce_max(utilities_copy, axis=1, keepdims=True), dtype=tf.float64))

        op_locs = tf.concat(op_locs, axis=1)

        pos = tf.concat([samples, op_locs], axis=1)
        shares = utils.calc_shares(pos, num_locs, op_num_locs, dim, arms, batch_size)
        rewards = utils.calc_rewards(utilities, shares, num_locs, arms)
        probs = tf.cast(tf.reduce_sum(policies * selection, axis=3), dtype=tf.float64)

        total_rewards = tf.reduce_sum(rewards, axis=1)
        action_probs = tf.math.log(tf.reduce_prod(probs, axis=1))

        expected_utilities = total_rewards * action_probs
        loss = lf.marginal_utility(expected_utilities)
        grads = tape.gradient(-loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        if not step % interval:
            validation_pop = utils.generate_map(batch_size, dim, dim)
            inputs = tf.expand_dims(validation_pop, axis=3)
            policies = model(inputs, False)
            policies = tf.tile(tf.expand_dims(policies, axis=2), [1, 1, arms, 1])
            samples, selection = utils.sample_actions(policies, batch_size, num_locs, arms, dim)
            op_locs = []
            utilities_copy = tf.reshape(validation_pop, [batch_size, -1])

            for j in range(op_num_locs):
                indmax = tf.argmax(utilities_copy, axis=1)
                opponent = tf.expand_dims(tf.stack([tf.math.floordiv(indmax, dim), tf.math.floormod(indmax, dim)], axis=1), axis=1)
                opponent = tf.tile(tf.expand_dims(opponent, axis=2), [1, 1, arms, 1])
                op_locs.append(opponent)
                utilities_copy = utilities_copy - (utilities_copy * tf.cast(utilities_copy == tf.reduce_max(utilities_copy, axis=1, keepdims=True), dtype=tf.float64))

            op_locs = tf.concat(op_locs, axis=1)
            pos = tf.concat([samples, op_locs], axis=1)
            shares = utils.calc_shares(pos, num_locs, op_num_locs, dim, arms, batch_size)
            rewards = utils.calc_rewards(utilities, shares, num_locs, arms)

            losses.append(loss.numpy())
            print('Step: {} Rew: {} EMU: {}'.format(step,
                                                    tf.reduce_mean(tf.reduce_max(tf.reduce_sum(rewards, axis=1), axis=1)),
                                                    loss))
            model.save_weights(file_name + '.h5')


plt.plot(range(0, iterations, interval), losses)
plt.savefig('mu_{}.png'.format(file_name))
