# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import tensorflow as tf
from tfwrapper import layers
from tfwrapper import normalisation as tfnorm


def simplenet2D(x, nlabels, training, n0=32, norm=tfnorm.batch_norm, scope_reuse=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv2D(x, 'conv1_1', num_filters=n0, training=training, normalisation=norm, add_bias=False)

        pool1 = layers.maxpool2D(conv1_1)

        conv2_1 = layers.conv2D(pool1, 'conv2_1', num_filters=n0*2, training=training, normalisation=norm, add_bias=False)

        pool2 = layers.maxpool2D(conv2_1)

        conv3_1 = layers.conv2D(pool2, 'conv3_1', num_filters=n0*4, training=training, normalisation=norm, add_bias=False)
        conv3_2 = layers.conv2D(conv3_1, 'conv3_2', num_filters=n0*4, training=training, normalisation=norm, add_bias=False)

        pool3 = layers.maxpool2D(conv3_2)

        conv4_1 = layers.conv2D(pool3, 'conv4_1', num_filters=n0*8, training=training, normalisation=norm, add_bias=False)
        conv4_2 = layers.conv2D(conv4_1, 'conv4_2', num_filters=n0*8, training=training, normalisation=norm, add_bias=False)

        pool4 = layers.maxpool2D(conv4_2)

        conv5_1 = layers.conv2D(pool4, 'conv5_1', num_filters=n0*16, training=training, normalisation=norm, add_bias=False)
        conv5_2 = layers.conv2D(conv5_1, 'conv5_2', num_filters=n0*16, training=training, normalisation=norm, add_bias=False)

        convD_1 = layers.conv2D(conv5_2, 'convD_1', num_filters=n0*16, training=training, normalisation=norm, add_bias=False)

        dense1 = layers.dense_layer(convD_1, 'dense1', hidden_units=n0*16, training=training, normalisation=norm, add_bias=False)

        logits = layers.dense_layer(dense1, 'dense2', hidden_units=nlabels, training=training, activation=tf.identity, normalisation=norm, add_bias=False)


    return logits



def vgg_base(x, nlabels, training, n0=64, conv_per_level=(2,2,2,2), norm=tfnorm.batch_norm, scope_reuse=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        net = x

        for ii, num_conv in enumerate(conv_per_level):
            for jj in range(num_conv):
                num_filt = n0*(2**ii) if ii <= 3 else  n0*(2**3)  # stop increasing filter number after 3rd level
                net = layers.conv2D(net, 'conv_%d_%d' % (ii,jj), num_filters=num_filt, training=training, normalisation=norm, add_bias=False)
            net = layers.maxpool2D(net)

        net = layers.dense_layer(net, 'dense1', hidden_units=n0*64, training=training, normalisation=norm, add_bias=False)
        net = layers.dense_layer(net, 'dense2', hidden_units=n0*64, training=training, normalisation=norm, add_bias=False)
        output = layers.dense_layer(net, 'dense3', hidden_units=nlabels, training=training, activation=tf.identity,  normalisation=norm, add_bias=False)

    return output

def vgg16(x, nlabels, training, n0=64, norm=tfnorm.batch_norm, scope_reuse=False):
    return vgg_base(x, nlabels, training, n0=n0, conv_per_level=(2, 2, 3, 3, 3), norm=norm, scope_reuse=scope_reuse)


def vgg19(x, nlabels, training, n0=64, norm=tfnorm.batch_norm, scope_reuse=False):
    return vgg_base(x, nlabels, training, n0=n0, conv_per_level=(2, 2, 4, 4, 4), norm=norm, scope_reuse=scope_reuse)


def CAM_net(x, nlabels, training, n0=32, norm=tfnorm.batch_norm, scope_reuse=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv2D(x, 'conv1_1', num_filters=n0, training=training, normalisation=norm, add_bias=False)

        pool1 = layers.maxpool2D(conv1_1)

        conv2_1 = layers.conv2D(pool1, 'conv2_1', num_filters=n0*2, training=training, normalisation=norm, add_bias=False)

        pool2 = layers.maxpool2D(conv2_1)

        conv3_1 = layers.conv2D(pool2, 'conv3_1', num_filters=n0*4, training=training, normalisation=norm, add_bias=False)
        conv3_2 = layers.conv2D(conv3_1, 'conv3_2', num_filters=n0*4, training=training, normalisation=norm, add_bias=False)

        conv4_1 = layers.conv2D(conv3_2, 'conv4_1', num_filters=n0*8, training=training, normalisation=norm, add_bias=False)
        conv4_2 = layers.conv2D(conv4_1, 'conv4_2', num_filters=n0*8, training=training, normalisation=norm, add_bias=False)

        conv5_1 = layers.conv2D(conv4_2, 'conv5_1', num_filters=n0*16, training=training, normalisation=norm, add_bias=False)
        conv5_2 = layers.conv2D(conv5_1, 'conv5_2', num_filters=n0*16, training=training, normalisation=norm, add_bias=False)

        convD_1 = layers.conv2D(conv5_2, 'feature_maps', num_filters=n0*16, training=training, normalisation=norm, add_bias=False)

        fm_averages = layers.global_averagepool2D(convD_1, name='fm_averages')

        logits = layers.dense_layer(fm_averages, 'weight_layer', hidden_units=nlabels, activation=tf.identity, norm=tf.identity, add_bias=False)

    return logits