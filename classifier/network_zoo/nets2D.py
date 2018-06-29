# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import tensorflow as tf
from tfwrapper import layers
from tfwrapper import normalisation as tfnorm

def resnet_base(x, nlabels, training, n0=64, units_per_level=(2, 2, 2, 2), resunit=layers.residual_unit2D, norm=tfnorm.batch_norm, scope_reuse=False, downsample_first=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        add_bias = False if norm == tfnorm.batch_norm else True

        if downsample_first:
            net = layers.conv2D(x, name='input_layer', training=training, kernel_size=(7,7), strides=(2,2), num_filters=n0, normalisation=norm, add_bias=add_bias)
        else:
            net = layers.conv2D(x, name='input_layer', training=training, kernel_size=(7,7), strides=(1,1), num_filters=n0, normalisation=norm, add_bias=add_bias)

        add_bias = False if norm == tfnorm.batch_norm else True

        for ii, num_conv in enumerate(units_per_level):
            for jj in range(num_conv):
                num_filt = n0*(2**ii) if ii <= 3 else  n0*(2**3)  # stop increasing filter number after 3rd level
                down_sample = True if jj == 0 else False
                net = resunit(net, 'resunit_%d_%d' % (ii,jj), num_filters=num_filt, training=training, normalisation=norm, add_bias=add_bias, down_sample=down_sample)
            net = layers.maxpool2D(net)

        net = layers.global_averagepool2D(net)

        output = layers.dense_layer(net, 'dense3', hidden_units=nlabels, training=training, activation=tf.identity, normalisation=norm, add_bias=add_bias)

    return output

def resnet34(x, nlabels, training, n0=64, resunit=layers.residual_unit2D, norm=tfnorm.batch_norm, scope_reuse=False, downsample_first=False):

    return resnet_base(x=x,
                       nlabels=nlabels,
                       training=training,
                       n0=n0,
                       units_per_level=(3, 4, 6, 3),
                       resunit=resunit,
                       norm=norm,
                       scope_reuse=scope_reuse,
                       downsample_first=downsample_first)


def identity_resnet34(x, nlabels, training, n0=64, norm=tfnorm.batch_norm, scope_reuse=False, downsample_first=False):

    return resnet_base(x=x,
                       nlabels=nlabels,
                       training=training,
                       n0=n0,
                       units_per_level=(3, 4, 6, 3),
                       resunit=layers.identity_residual_unit2D,
                       norm=norm,
                       scope_reuse=scope_reuse,
                       downsample_first=downsample_first)


def vgg_base(x, nlabels, training, n0=64, conv_per_level=(2,2,2,2), norm=tfnorm.batch_norm, scope_reuse=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        add_bias = False if norm == tfnorm.batch_norm else True
        net = x

        for ii, num_conv in enumerate(conv_per_level):
            for jj in range(num_conv):
                num_filt = n0*(2**ii) if ii <= 3 else  n0*(2**3)  # stop increasing filter number after 3rd level
                net = layers.conv2D(net, 'conv_%d_%d' % (ii,jj), num_filters=num_filt, training=training, normalisation=norm, add_bias=add_bias)
            net = layers.maxpool2D(net)

        net = layers.dense_layer(net, 'dense1', hidden_units=n0*64, training=training, normalisation=norm, add_bias=add_bias)
        net = layers.dense_layer(net, 'dense2', hidden_units=n0*64, training=training, normalisation=norm, add_bias=add_bias)
        output = layers.dense_layer(net, 'dense3', hidden_units=nlabels, training=training, activation=tf.identity,  normalisation=norm, add_bias=add_bias)

    return output

def vgg16(x, nlabels, training, n0=64, norm=tfnorm.batch_norm, scope_reuse=False):
    return vgg_base(x, nlabels, training, n0=n0, conv_per_level=(2, 2, 3, 3, 3), norm=norm, scope_reuse=scope_reuse)


def vgg19(x, nlabels, training, n0=64, norm=tfnorm.batch_norm, scope_reuse=False):
    return vgg_base(x, nlabels, training, n0=n0, conv_per_level=(2, 2, 4, 4, 4), norm=norm, scope_reuse=scope_reuse)


def simplenet2D(x, nlabels, training, n0=32, norm=tfnorm.batch_norm, scope_reuse=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        add_bias = False if norm == tfnorm.batch_norm else True

        conv1_1 = layers.conv2D(x, 'conv1_1', num_filters=n0, training=training, normalisation=norm, add_bias=add_bias)

        pool1 = layers.maxpool2D(conv1_1)

        conv2_1 = layers.conv2D(pool1, 'conv2_1', num_filters=n0*2, training=training, normalisation=norm, add_bias=add_bias)

        pool2 = layers.maxpool2D(conv2_1)

        conv3_1 = layers.conv2D(pool2, 'conv3_1', num_filters=n0*4, training=training, normalisation=norm, add_bias=add_bias)
        conv3_2 = layers.conv2D(conv3_1, 'conv3_2', num_filters=n0*4, training=training, normalisation=norm, add_bias=add_bias)

        pool3 = layers.maxpool2D(conv3_2)

        conv4_1 = layers.conv2D(pool3, 'conv4_1', num_filters=n0*8, training=training, normalisation=norm, add_bias=add_bias)
        conv4_2 = layers.conv2D(conv4_1, 'conv4_2', num_filters=n0*8, training=training, normalisation=norm, add_bias=add_bias)

        pool4 = layers.maxpool2D(conv4_2)

        conv5_1 = layers.conv2D(pool4, 'conv5_1', num_filters=n0*16, training=training, normalisation=norm, add_bias=add_bias)
        conv5_2 = layers.conv2D(conv5_1, 'conv5_2', num_filters=n0*16, training=training, normalisation=norm, add_bias=add_bias)

        convD_1 = layers.conv2D(conv5_2, 'convD_1', num_filters=n0*16, training=training, normalisation=norm, add_bias=add_bias)

        dense1 = layers.dense_layer(convD_1, 'dense1', hidden_units=n0*16, training=training, normalisation=norm, add_bias=add_bias)

        logits = layers.dense_layer(dense1, 'dense2', hidden_units=nlabels, training=training, activation=tf.identity, normalisation=norm, add_bias=add_bias)


    return logits

def CAM_net(x, nlabels, training, n0=32, norm=tfnorm.batch_norm, scope_reuse=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        add_bias = False if norm == tfnorm.batch_norm else True

        conv1_1 = layers.conv2D(x, 'conv1_1', num_filters=n0, training=training, normalisation=norm, add_bias=add_bias)

        pool1 = layers.maxpool2D(conv1_1)

        conv2_1 = layers.conv2D(pool1, 'conv2_1', num_filters=n0*2, training=training, normalisation=norm, add_bias=add_bias)

        pool2 = layers.maxpool2D(conv2_1)

        conv3_1 = layers.conv2D(pool2, 'conv3_1', num_filters=n0*4, training=training, normalisation=norm, add_bias=add_bias)
        conv3_2 = layers.conv2D(conv3_1, 'conv3_2', num_filters=n0*4, training=training, normalisation=norm, add_bias=add_bias)

        conv4_1 = layers.conv2D(conv3_2, 'conv4_1', num_filters=n0*8, training=training, normalisation=norm, add_bias=add_bias)
        conv4_2 = layers.conv2D(conv4_1, 'conv4_2', num_filters=n0*8, training=training, normalisation=norm, add_bias=add_bias)

        conv5_1 = layers.conv2D(conv4_2, 'conv5_1', num_filters=n0*16, training=training, normalisation=norm, add_bias=add_bias)
        conv5_2 = layers.conv2D(conv5_1, 'conv5_2', num_filters=n0*16, training=training, normalisation=norm, add_bias=add_bias)

        convD_1 = layers.conv2D(conv5_2, 'feature_maps', num_filters=n0*16, training=training, normalisation=norm, add_bias=add_bias)

        fm_averages = layers.global_averagepool2D(convD_1, name='fm_averages')

        logits = layers.dense_layer(fm_averages, 'weight_layer', hidden_units=nlabels, activation=tf.identity, normalisation=tfnorm.identity, add_bias=False)  # no bias for last layer

    return logits