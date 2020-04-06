import tensorflow as tf
import tensorflow_probability as tfp
import matplotlib.pyplot as plt

tf.executing_eagerly()


ALPHA = tf.constant(2.0)
BETA = tf.constant(1.0)


def generate_map(batch_size, dim_x=10, dim_y=10):
    dist = tfp.distributions.InverseGamma(ALPHA, BETA)
    gmap = dist.sample([batch_size, dim_x, dim_y])

    return tf.cast(gmap, dtype=tf.float64)


if __name__ == '__main__':
    gmap = generate_map()
    plt.imshow(gmap, cmap=plt.get_cmap("OrRd"))
    print(gmap)
    plt.show()
