import tensorflow as tf


class ResidualBlock(tf.keras.Model):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.cnn_1 = tf.keras.layers.Conv2D(64, 1, 1,
                                            padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(l=0.00001))
        self.bn_1 = tf.keras.layers.BatchNormalization()

        self.cnn_2 = tf.keras.layers.Conv2D(64, 3, 1,
                                            padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(l=0.00001))
        self.bn_2 = tf.keras.layers.BatchNormalization()

        self.cnn_3 = tf.keras.layers.Conv2D(256, 1, 1,
                                            padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(l=0.00001))
        self.bn_3 = tf.keras.layers.BatchNormalization()


    def call(self, input_tensor, training=False):
        cnn_1 = self.cnn_1(input_tensor)
        bn_1 = self.bn_1(cnn_1, training=training)

        cnn_2 = self.cnn_2(bn_1)
        bn_2 = self.bn_2(cnn_2, training=training)

        cnn_3 = self.cnn_3(bn_2)
        bn_3 = self.bn_3(cnn_3, training=training)

        output_tensor = bn_3 + input_tensor

        return tf.nn.relu(output_tensor)


class ResidualNetwork(tf.keras.Model):
    def __init__(self, dim):
        super(ResidualNetwork, self).__init__()

        self.cnn_1 = tf.keras.layers.Conv2D(256, 3, 1,
                                       activation='relu',
                                       padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(l=0.00001))
        self.rb_1 = ResidualBlock()
        # self.rb_2 = ResidualBlock()
        # self.rb_3 = ResidualBlock()

        self.cnn_flat = tf.keras.layers.Flatten()

        self.pi_1 = tf.keras.layers.Dense(units=dim**2, activation='softmax', use_bias=False)
        self.pi_2 = tf.keras.layers.Dense(units=dim**2, activation='softmax', use_bias=False)
        self.pi_3 = tf.keras.layers.Dense(units=dim**2, activation='softmax', use_bias=False)

    def call(self, input_tensor, training=False):
        cnn_1 = self.cnn_1(input_tensor)
        rb_1 = self.rb_1(cnn_1, training)
        # rb_2 = self.rb_1(rb_1, training)
        # rb_3 = self.rb_1(rb_2, training)
        cnn_flat = self.cnn_flat(rb_1)

        pi_1 = self.pi_1(cnn_flat)
        pi_2 = self.pi_1(cnn_flat)
        pi_3 = self.pi_1(cnn_flat)

        return tf.stack([pi_1, pi_2, pi_3], axis=1)