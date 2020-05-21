import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import networks
import utils
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.patheffects as path_effects


batch_size = 32
dim = 5
arms = 8
num_locs = 3
op_num_locs = 2
model = networks.SimpleNetwork(dim, arms=arms, num_locs=num_locs)
quantal_temp = 1
optimizer = tf.optimizers.Adam(1e-5)
model.build(input_shape=(batch_size, dim, dim, 1))
iterations = 20000
interval = 2000

file_name = '5-5_3-2'


test_population = utils.generate_map(batch_size, dim, dim)
inputs = tf.expand_dims(test_population, axis=3)
policies = model(inputs, False)
samples, selection = utils.sample_actions(policies, batch_size, num_locs, arms, dim)
samples = samples.numpy()

op_locs = []
utilities_copy = tf.reshape(test_population, [batch_size, -1])

for j in range(op_num_locs):
    indmax = tf.argmax(utilities_copy, axis=1)
    opponent = tf.expand_dims(tf.stack([tf.math.floordiv(indmax, dim), tf.math.floormod(indmax, dim)], axis=1), axis=1)
    opponent = tf.tile(tf.expand_dims(opponent, axis=2), [1, 1, arms, 1])
    op_locs.append(opponent)
    utilities_copy = utilities_copy - (utilities_copy * tf.cast(utilities_copy == tf.reduce_max(utilities_copy, axis=1, keepdims=True), dtype=tf.float64))

op_locs = tf.concat(op_locs, axis=1).numpy()

pos = tf.concat([samples, op_locs], axis=1)
shares = utils.calc_shares(pos, num_locs, op_num_locs, dim, arms, batch_size)
rewards = tf.reduce_sum(utils.calc_rewards(test_population, shares, num_locs, arms), axis=1)
partitions = tf.transpose(tf.reduce_sum(tf.ones([batch_size, dim, dim, num_locs, arms]) * tf.cast(shares, tf.float32),
                                        axis=3), [0, 3, 1, 2])

best = tf.argmax(rewards,axis=1)
marks = np.linspace(0, dim, dim+1) - 0.5

for batch in range(batch_size):
    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 10))
    im = ax1.imshow(test_population[batch], cmap=plt.get_cmap('YlGn'))
    ax1.set_xticks(marks)
    ax1.set_yticks(marks)
    ax1.set_xticklabels([])
    ax1.set_yticklabels([])
    divider = make_axes_locatable(ax1)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    # plt.colorbar(im, ax=ax1)
    for loc in range(num_locs):
        ax1.scatter(samples[batch][loc][best[batch]][1], samples[batch][loc][best[batch]][0], c='cyan',
                    edgecolor='black', s=200)
        txt = ax1.text(samples[batch][loc][best[batch]][1] + 0.3, samples[batch][loc][best[batch]][0] - 0.3, loc,
                 fontweight='bold', c='white')
        txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                       path_effects.Normal()])
    for loc in range(op_num_locs):
        ax1.scatter(op_locs[batch][loc][best[batch]][1], op_locs[batch][loc][best[batch]][0], c='magenta', edgecolor='black', s=100)
    ax1.grid()
    # print(partitions[batch][:][:][best[batch]])
    im = ax2.imshow(partitions[batch][:][:][best[batch]], cmap=plt.get_cmap('cool_r'), vmin=0)
    ax2.set_xticks(marks)
    ax2.set_yticks(marks)
    ax2.set_xticklabels([])
    ax2.set_yticklabels([])
    divider = make_axes_locatable(ax2)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    ax2.grid()

    max_return, br, opp = sgpe(test_population[batch], dim, num_locs, op_num_locs)
    im = ax3.imshow(test_population[batch], cmap=plt.get_cmap('YlGn'))
    ax3.set_xticks(marks)
    ax3.set_yticks(marks)
    ax3.set_xticklabels([])
    ax3.set_yticklabels([])
    divider = make_axes_locatable(ax3)
    cax = divider.append_axes("right", size="5%", pad=0.1)
    plt.colorbar(im, cax=cax)
    for loc in range(num_locs):
        ax3.scatter(br[0][loc][0][1], br[0][loc][0][0], c='cyan',
                    edgecolor='black', s=200)
        txt = ax3.text(br[0][loc][0][1] + 0.3, br[0][loc][0][0] - 0.3, loc,
                 fontweight='bold', c='white')
        txt.set_path_effects([path_effects.Stroke(linewidth=2, foreground='black'),
                       path_effects.Normal()])
    for loc in range(op_num_locs):
        ax3.scatter(opp[0][loc][0][1], opp[0][loc][0][0], c='magenta',
                    edgecolor='black', s=100)
    ax3.grid()

    plt.savefig('imgs_{}/test_{}.png'.format(file_name, batch), bbox_inches='tight')
    print("Candidates: ", rewards[batch].numpy(), "Selected: ", rewards[batch][best[batch]].numpy(), "BR: ", max_return)
