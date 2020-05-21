import tensorflow as tf


class SimpleNetwork(tf.keras.Model):
    def __init__(self, dim, arms, num_locs):
        super(SimpleNetwork, self).__init__()
        self.num_arms = arms
        self.dim = dim
        self.num_locs = num_locs

        self.cnn_1 = tf.keras.layers.Conv2D(32, 3, 1,
                                            padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(l=0.00001))

        self.cnn_2 = tf.keras.layers.Conv2D(32, 3, 1,
                                            padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(l=0.00001))

        self.cnn_flat = tf.keras.layers.Flatten()

        self.output_policies = []
        for i in range(num_locs):
            self.output_policies.append(tf.keras.layers.Dense(units=arms*(dim**2), activation='softmax', use_bias=False))

    def call(self, input_tensor, training=False):
        cnn_1 = self.cnn_1(input_tensor)
        cnn_2 = self.cnn_2(cnn_1)
        cnn_flat = self.cnn_flat(cnn_2)

        policies = []
        for i in range(self.num_locs):
            policy = self.output_policies[i](cnn_flat)
            policies.append(tf.reshape(policy, [-1, self.num_arms, self.dim**2]))

        return tf.stack(policies, axis=1)


class ResidualBlock(tf.keras.Model):
    def __init__(self):
        super(ResidualBlock, self).__init__()

        self.cnn_1 = tf.keras.layers.Conv2D(32, 1, 1,
                                            padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(l=0.00001))
        self.bn_1 = tf.keras.layers.BatchNormalization()

        self.cnn_2 = tf.keras.layers.Conv2D(32, 3, 1,
                                            padding='same',
                                            kernel_regularizer=tf.keras.regularizers.l2(l=0.00001))
        self.bn_2 = tf.keras.layers.BatchNormalization()

        self.cnn_3 = tf.keras.layers.Conv2D(64, 1, 1,
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
    def __init__(self, dim, arms, num_locs):
        super(ResidualNetwork, self).__init__()
        self.num_arms = arms
        self.dim = dim
        self.num_locs = num_locs

        self.cnn_1 = tf.keras.layers.Conv2D(64, 3, 1,
                                       activation='relu',
                                       padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(l=0.00001))
        self.rb_1 = ResidualBlock()
        self.rb_2 = ResidualBlock()
        self.rb_3 = ResidualBlock()

        self.cnn_flat = tf.keras.layers.Flatten()

        self.output_policies = []
        for i in range(num_locs):
            self.output_policies.append(tf.keras.layers.Dense(units=arms*(dim**2), activation='softmax', use_bias=False))


    def call(self, input_tensor, training=False):
        cnn_1 = self.cnn_1(input_tensor)
        rb_1 = self.rb_1(cnn_1, training)
        # rb_2 = self.rb_2(rb_1, training)
        # rb_3 = self.rb_3(rb_2, training)
        cnn_flat = self.cnn_flat(rb_1)

        policies = []
        for i in range(self.num_locs):
            policy = self.output_policies[i](cnn_flat)
            policies.append(tf.reshape(policy, [-1, self.num_arms, self.dim**2]))

        return tf.stack(policies, axis=1)


class SimplePolicyNetwork(tf.keras.Model):
    def __init__(self, dim, arms, num_locs):
        super(SimplePolicyNetwork, self).__init__()
        self.num_arms = arms
        self.dim = dim
        self.num_locs = num_locs

        self.cnn_1 = tf.keras.layers.Conv2D(64, 3, 1,
                                       activation='relu',
                                       padding='same',
                                       kernel_regularizer=tf.keras.regularizers.l2(l=0.00001))

        self.rb_1 = ResidualBlock()
        self.rb_2 = ResidualBlock()
        self.rb_3 = ResidualBlock()

        self.cnn_flat = tf.keras.layers.Flatten()

        self.output_policies = []
        for i in range(num_locs):
            self.output_policies.append(tf.keras.layers.Dense(units=dim**2, activation='softmax', use_bias=False))


    def call(self, input_tensor, training=False):
        cnn_1 = self.cnn_1(input_tensor)
        rb_1 = self.rb_1(cnn_1, training)
        # rb_2 = self.rb_2(rb_1, training)
        # rb_3 = self.rb_3(rb_2, training)
        cnn_flat = self.cnn_flat(rb_1)

        policies = []
        for i in range(self.num_locs):
            policies.append(self.output_policies[i](cnn_flat))

        return tf.stack(policies, axis=1)
