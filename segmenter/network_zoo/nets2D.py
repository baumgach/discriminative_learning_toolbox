# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import tensorflow as tf
from tfwrapper import layers


def unet2D_modified_bn(images, training, nlabels, n0=64, scope_reuse=False):

    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv2D_layer_bn(images, 'conv1_1', num_filters=n0, training=training)
        conv1_2 = layers.conv2D_layer_bn(conv1_1, 'conv1_2', num_filters=n0, training=training)

        pool1 = layers.maxpool2D_layer(conv1_2)

        conv2_1 = layers.conv2D_layer_bn(pool1, 'conv2_1', num_filters=n0*2, training=training)
        conv2_2 = layers.conv2D_layer_bn(conv2_1, 'conv2_2', num_filters=n0*2, training=training)

        pool2 = layers.maxpool2D_layer(conv2_2)

        conv3_1 = layers.conv2D_layer_bn(pool2, 'conv3_1', num_filters=n0*4, training=training)
        conv3_2 = layers.conv2D_layer_bn(conv3_1, 'conv3_2', num_filters=n0*4, training=training)

        pool3 = layers.maxpool2D_layer(conv3_2)

        conv4_1 = layers.conv2D_layer_bn(pool3, 'conv4_1', num_filters=n0*8, training=training)
        conv4_2 = layers.conv2D_layer_bn(conv4_1, 'conv4_2', num_filters=n0*8, training=training)

        pool4 = layers.maxpool2D_layer(conv4_2)

        conv5_1 = layers.conv2D_layer_bn(pool4, 'conv5_1', num_filters=n0*16, training=training)
        conv5_2 = layers.conv2D_layer_bn(conv5_1, 'conv5_2', num_filters=n0*16, training=training)

        upconv4 = layers.deconv2D_layer_bn(conv5_2, name='upconv4', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat4 = layers.crop_and_concat_layer([upconv4, conv4_2], axis=3)

        conv6_1 = layers.conv2D_layer_bn(concat4, 'conv6_1', num_filters=n0*8, training=training)
        conv6_2 = layers.conv2D_layer_bn(conv6_1, 'conv6_2', num_filters=n0*8, training=training)

        upconv3 = layers.deconv2D_layer_bn(conv6_2, name='upconv3', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)

        concat3 = layers.crop_and_concat_layer([upconv3, conv3_2], axis=3)

        conv7_1 = layers.conv2D_layer_bn(concat3, 'conv7_1', num_filters=n0*4, training=training)
        conv7_2 = layers.conv2D_layer_bn(conv7_1, 'conv7_2', num_filters=n0*4, training=training)

        upconv2 = layers.deconv2D_layer_bn(conv7_2, name='upconv2', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat2 = layers.crop_and_concat_layer([upconv2, conv2_2], axis=3)

        conv8_1 = layers.conv2D_layer_bn(concat2, 'conv8_1', num_filters=n0*2, training=training)
        conv8_2 = layers.conv2D_layer_bn(conv8_1, 'conv8_2', num_filters=n0*2, training=training)

        upconv1 = layers.deconv2D_layer_bn(conv8_2, name='upconv1', kernel_size=(4, 4), strides=(2, 2), num_filters=nlabels, training=training)
        concat1 = layers.crop_and_concat_layer([upconv1, conv1_2], axis=3)

        conv9_1 = layers.conv2D_layer_bn(concat1, 'conv9_1', num_filters=n0, training=training)
        conv9_2 = layers.conv2D_layer_bn(conv9_1, 'conv9_2', num_filters=n0, training=training)

        pred = layers.conv2D_layer_bn(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training)

        return pred
