# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)


import tensorflow as tf
from tfwrapper import layers
from tfwrapper import normalisation

def unet3D_modified_bn(images, training, nlabels, n0=32, scope_reuse=False):


    with tf.variable_scope('classifier') as scope:

        if scope_reuse:
            scope.reuse_variables()


        conv1_1 = layers.conv3D_layer_bn(images, 'conv1_1', num_filters=n0, training=training)
        conv1_2 = layers.conv3D_layer_bn(conv1_1, 'conv1_2', num_filters=n0*2, training=training)

        pool1 = layers.maxpool3D_layer(conv1_2)

        conv2_1 = layers.conv3D_layer_bn(pool1, 'conv2_1', num_filters=n0*2, training=training)
        conv2_2 = layers.conv3D_layer_bn(conv2_1, 'conv2_2', num_filters=n0*4, training=training)

        pool2 = layers.maxpool3D_layer(conv2_2)

        conv3_1 = layers.conv3D_layer_bn(pool2, 'conv3_1', num_filters=n0*4, training=training)
        conv3_2 = layers.conv3D_layer_bn(conv3_1, 'conv3_2', num_filters=n0*8, training=training)

        pool3 = layers.maxpool3D_layer(conv3_2)

        conv4_1 = layers.conv3D_layer_bn(pool3, 'conv4_1', num_filters=n0*8, training=training)
        conv4_2 = layers.conv3D_layer_bn(conv4_1, 'conv4_2', num_filters=n0*16, training=training)

        upconv3 = layers.deconv3D_layer_bn(conv4_2, name='upconv3', num_filters=n0*16, training=training)
        concat3 = layers.crop_and_concat([upconv3, conv3_2], axis=4)

        conv5_1 = layers.conv3D_layer_bn(concat3, 'conv5_1', num_filters=n0*8, training=training)
        conv5_2 = layers.conv3D_layer_bn(conv5_1, 'conv5_2', num_filters=n0*8, training=training)

        upconv2 = layers.deconv3D_layer_bn(conv5_2, name='upconv2', num_filters=n0*8, training=training)
        concat2 = layers.crop_and_concat([upconv2, conv2_2], axis=4)

        conv6_1 = layers.conv3D_layer_bn(concat2, 'conv6_1', num_filters=n0*4, training=training)
        conv6_2 = layers.conv3D_layer_bn(conv6_1, 'conv6_2', num_filters=n0*4, training=training)

        upconv1 = layers.deconv3D_layer_bn(conv6_2, name='upconv1', num_filters=n0*4, training=training)
        concat1 = layers.crop_and_concat([upconv1, conv1_2], axis=4)

        conv8_1 = layers.conv3D_layer_bn(concat1, 'conv8_1', num_filters=n0*2, training=training)
        conv8_2 = layers.conv3D_layer_bn(conv8_1, 'conv8_2', num_filters=n0*2, training=training)

        pred = layers.conv3D_layer_bn(conv8_2, 'pred', num_filters=nlabels, kernel_size=(1, 1, 1),
                                      activation=tf.identity, training=training)

        return pred


def unet3D_modified(x, training, nlabels, n0=32, resolution_levels=3, norm=normalisation.batch_norm, scope_reuse=False, return_net=False):

    with tf.variable_scope('segmenter') as scope:

        if scope_reuse:
            scope.reuse_variables()

        enc = []

        with tf.variable_scope('encoder'):

            for ii in range(resolution_levels):

                enc.append([])

                # In first layer set input to x rather than max pooling
                if ii == 0:
                    enc[ii].append(x)
                else:
                    enc[ii].append(layers.maxpool3D(enc[ii-1][-1]))

                enc[ii].append(layers.conv3D(enc[ii][-1], 'conv_%d_1' % ii, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=False))
                enc[ii].append(layers.conv3D(enc[ii][-1], 'conv_%d_2' % ii, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=False))

        dec = []

        with tf.variable_scope('decoder'):

            for jj in range(resolution_levels-1):

                ii = resolution_levels - jj - 1  # used to index the encoder again

                dec.append([])

                if jj == 0:
                    next_inp = enc[ii][-1]
                else:
                    next_inp = dec[jj-1][-1]

                dec[jj].append(layers.transposed_conv3D(next_inp, name='upconv_%d' % jj, num_filters=nlabels, training=training, normalisation=norm, add_bias=False))

                # skip connection
                dec[jj].append(layers.crop_and_concat([dec[jj][-1], enc[ii-1][-1]], axis=3))

                dec[jj].append(layers.conv3D(dec[jj][-1], 'conv_%d_1' % jj, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=False))
                dec[jj].append(layers.conv3D(dec[jj][-1], 'conv_%d_2' % jj, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=False))

        output = layers.conv2D(dec[-1][-1], 'prediction', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training, normalisation=norm, add_bias=False)

        dec[-1].append(output)

        if return_net:
            net = enc + dec
            return output, net

        return output