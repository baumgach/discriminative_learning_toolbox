# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)


import tensorflow as tf
from tfwrapper import layers
from tfwrapper import normalisation as tfnorm


def FCN_32_bn(x, nlabels, training, n0=32, norm=tfnorm.batch_norm, scope_reuse=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        add_bias = False if norm == tfnorm.batch_norm else True

        conv1_1 = layers.conv3D(x, 'conv1_1', num_filters=n0, training=training, normalisation=norm, add_bias=add_bias)

        pool1 = layers.maxpool3D(conv1_1)

        conv2_1 = layers.conv3D(pool1, 'conv2_1', num_filters=n0*2, training=training, normalisation=norm, add_bias=add_bias)

        pool2 = layers.maxpool3D(conv2_1)

        conv3_1 = layers.conv3D(pool2, 'conv3_1', num_filters=n0*4, training=training, normalisation=norm, add_bias=add_bias)
        conv3_2 = layers.conv3D(conv3_1, 'conv3_2', num_filters=n0*4, training=training, normalisation=norm, add_bias=add_bias)

        pool3 = layers.maxpool3D(conv3_2)

        conv4_1 = layers.conv3D(pool3, 'conv4_1', num_filters=n0*8, training=training, normalisation=norm, add_bias=add_bias)
        conv4_2 = layers.conv3D(conv4_1, 'conv4_2', num_filters=n0*8, training=training, normalisation=norm, add_bias=add_bias)

        pool4 = layers.maxpool3D(conv4_2)

        conv5_1 = layers.conv3D(pool4, 'conv5_1', num_filters=n0*8, training=training, normalisation=norm, add_bias=add_bias)
        conv5_2 = layers.conv3D(conv5_1, 'conv5_2', num_filters=n0*8, training=training, normalisation=norm, add_bias=add_bias)

        convD_1 = layers.conv3D(conv5_2, 'convD_1', num_filters=n0*8, training=training, normalisation=norm, add_bias=add_bias)
        convD_2 = layers.conv3D(convD_1,
                                'convD_2',
                                num_filters=nlabels,
                                training=training,
                                kernel_size=(1,1,1),
                                activation=tf.identity,
                                normalisation=tfnorm.batch_norm,
                                add_bias=False)

        diag_logits = layers.global_averagepool3D(convD_2, name='diagnosis_avg')

    return diag_logits



def allconv_bn(x, nlabels, training, n0=32, norm=tfnorm.batch_norm, scope_reuse=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        add_bias = False if norm==tfnorm.batch_norm else True

        conv1_1 = layers.conv3D(x, 'conv1_1', num_filters=n0, training=training, strides=(2, 2, 2), normalisation=norm, add_bias=add_bias)

        conv2_1 = layers.conv3D(conv1_1, 'conv2_1', num_filters=n0*2, training=training, strides=(2,2,2), normalisation=norm, add_bias=add_bias)

        conv3_1 = layers.conv3D(conv2_1, 'conv3_1', num_filters=n0*4, training=training, normalisation=norm, add_bias=add_bias)
        conv3_2 = layers.conv3D(conv3_1, 'conv3_2', num_filters=n0*4, training=training, strides=(2,2,2), normalisation=norm, add_bias=add_bias)

        conv4_1 = layers.conv3D(conv3_2, 'conv4_1', num_filters=n0*4, training=training, normalisation=norm, add_bias=add_bias)
        conv4_2 = layers.conv3D(conv4_1, 'conv4_2', num_filters=n0*4, training=training, strides=(2,2,2))

        conv5_1 = layers.conv3D(conv4_2, 'conv5_1', num_filters=n0*4, training=training, normalisation=norm, add_bias=add_bias)
        conv5_2 = layers.conv3D(conv5_1, 'conv5_2', num_filters=n0*4, training=training, normalisation=norm, add_bias=add_bias)

        convD_1 = layers.conv3D(conv5_2, 'convD_1', num_filters=n0*4, training=training, normalisation=norm, add_bias=add_bias)
        convD_2 = layers.conv3D(convD_1,
                                'convD_2',
                                num_filters=nlabels,
                                training=training,
                                kernel_size=(1,1,1),
                                activation=tf.identity,
                                normalisation=norm,
                                add_bias=add_bias)

        diag_logits = layers.global_averagepool3D(convD_2, name='diagnosis_avg')

    return diag_logits


def C3D_32_bn(x, nlabels, training, n0=32, norm=tfnorm.batch_norm, scope_reuse=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        add_bias = False if norm == tfnorm.batch_norm else True

        conv1_1 = layers.conv3D(x, 'conv1_1', num_filters=n0, training=training, normalisation=norm, add_bias=add_bias)

        pool1 = layers.maxpool3D(conv1_1)

        conv2_1 = layers.conv3D(pool1, 'conv2_1', num_filters=n0*2, training=training, normalisation=norm, add_bias=add_bias)

        pool2 = layers.maxpool3D(conv2_1)

        conv3_1 = layers.conv3D(pool2, 'conv3_1', num_filters=n0*4, training=training, normalisation=norm, add_bias=add_bias)
        conv3_2 = layers.conv3D(conv3_1, 'conv3_2', num_filters=n0*4, training=training, normalisation=norm, add_bias=add_bias)

        pool3 = layers.maxpool3D(conv3_2)

        conv4_1 = layers.conv3D(pool3, 'conv4_1', num_filters=n0*8, training=training, normalisation=norm, add_bias=add_bias)
        conv4_2 = layers.conv3D(conv4_1, 'conv4_2', num_filters=n0*8, training=training, normalisation=norm, add_bias=add_bias)

        pool4 = layers.maxpool3D(conv4_2)

        dense1 = layers.dense_layer(pool4, 'dense1', hidden_units=n0*16, training=training, normalisation=norm, add_bias=add_bias)
        diag_logits = layers.dense_layer(dense1, 'diag_logits', hidden_units=nlabels, activation=tf.identity, training=training, normalisation=norm, add_bias=add_bias)

    return diag_logits
