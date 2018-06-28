# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import tensorflow as tf
from tfwrapper import layers
from tfwrapper import normalisation


def unet2D_modified_bn(images, training, nlabels, n0=64, scope_reuse=False):

    with tf.variable_scope('segmenter') as scope:

        if scope_reuse:
            scope.reuse_variables()

        conv1_1 = layers.conv2D(images, 'conv1_1', num_filters=n0, training=training, normalisation=normalisation.batch_norm, add_bias=False)
        conv1_2 = layers.conv2D(conv1_1, 'conv1_2', num_filters=n0, training=training, normalisation=normalisation.batch_norm, add_bias=False)

        pool1 = layers.maxpool2D(conv1_2)

        conv2_1 = layers.conv2D(pool1, 'conv2_1', num_filters=n0*2, training=training, normalisation=normalisation.batch_norm, add_bias=False)
        conv2_2 = layers.conv2D(conv2_1, 'conv2_2', num_filters=n0*2, training=training, normalisation=normalisation.batch_norm, add_bias=False)

        pool2 = layers.maxpool2D(conv2_2)

        conv3_1 = layers.conv2D(pool2, 'conv3_1', num_filters=n0*4, training=training, normalisation=normalisation.batch_norm, add_bias=False)
        conv3_2 = layers.conv2D(conv3_1, 'conv3_2', num_filters=n0*4, training=training, normalisation=normalisation.batch_norm, add_bias=False)

        pool3 = layers.maxpool2D(conv3_2)

        conv4_1 = layers.conv2D(pool3, 'conv4_1', num_filters=n0*8, training=training, normalisation=normalisation.batch_norm, add_bias=False)
        conv4_2 = layers.conv2D(conv4_1, 'conv4_2', num_filters=n0*8, training=training, normalisation=normalisation.batch_norm, add_bias=False)

        pool4 = layers.maxpool2D(conv4_2)

        conv5_1 = layers.conv2D(pool4, 'conv5_1', num_filters=n0*16, training=training, normalisation=normalisation.batch_norm, add_bias=False)
        conv5_2 = layers.conv2D(conv5_1, 'conv5_2', num_filters=n0*16, training=training, normalisation=normalisation.batch_norm, add_bias=False)

        upconv4 = layers.transposed_conv2D(conv5_2, name='upconv4', num_filters=nlabels, training=training, normalisation=normalisation.batch_norm, add_bias=False)
        concat4 = layers.crop_and_concat([upconv4, conv4_2], axis=3)

        conv6_1 = layers.conv2D(concat4, 'conv6_1', num_filters=n0*8, training=training, normalisation=normalisation.batch_norm, add_bias=False)
        conv6_2 = layers.conv2D(conv6_1, 'conv6_2', num_filters=n0*8, training=training, normalisation=normalisation.batch_norm, add_bias=False)

        upconv3 = layers.transposed_conv2D(conv6_2, name='upconv3', num_filters=nlabels, training=training, normalisation=normalisation.batch_norm, add_bias=False)

        concat3 = layers.crop_and_concat([upconv3, conv3_2], axis=3)

        conv7_1 = layers.conv2D(concat3, 'conv7_1', num_filters=n0*4, training=training, normalisation=normalisation.batch_norm, add_bias=False)
        conv7_2 = layers.conv2D(conv7_1, 'conv7_2', num_filters=n0*4, training=training, normalisation=normalisation.batch_norm, add_bias=False)

        upconv2 = layers.transposed_conv2D(conv7_2, name='upconv2', num_filters=nlabels, training=training, normalisation=normalisation.batch_norm, add_bias=False)
        concat2 = layers.crop_and_concat([upconv2, conv2_2], axis=3)

        conv8_1 = layers.conv2D(concat2, 'conv8_1', num_filters=n0*2, training=training, normalisation=normalisation.batch_norm, add_bias=False)
        conv8_2 = layers.conv2D(conv8_1, 'conv8_2', num_filters=n0*2, training=training, normalisation=normalisation.batch_norm, add_bias=False)

        upconv1 = layers.transposed_conv2D(conv8_2, name='upconv1', num_filters=nlabels, training=training, normalisation=normalisation.batch_norm, add_bias=False)
        concat1 = layers.crop_and_concat([upconv1, conv1_2], axis=3)

        conv9_1 = layers.conv2D(concat1, 'conv9_1', num_filters=n0, training=training, normalisation=normalisation.batch_norm, add_bias=False)
        conv9_2 = layers.conv2D(conv9_1, 'conv9_2', num_filters=n0, training=training, normalisation=normalisation.batch_norm, add_bias=False)

        pred = layers.conv2D(conv9_2, 'pred', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training, normalisation=normalisation.batch_norm, add_bias=False)

        return pred


def unet2D_modified(x, training, nlabels, n0=64, resolution_levels=4, norm=normalisation.batch_norm, scope_reuse=False, return_net=False):

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
                    enc[ii].append(layers.maxpool2D(enc[ii-1][-1]))

                enc[ii].append(layers.conv2D(enc[ii][-1], 'conv_%d_1' % ii, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=False))
                enc[ii].append(layers.conv2D(enc[ii][-1], 'conv_%d_2' % ii, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=False))

        dec = []

        with tf.variable_scope('decoder'):

            for jj in range(resolution_levels-1):

                ii = resolution_levels - jj - 1  # used to index the encoder again

                dec.append([])

                if jj == 0:
                    next_inp = enc[ii][-1]
                else:
                    next_inp = dec[jj-1][-1]

                dec[jj].append(layers.transposed_conv2D(next_inp, name='upconv_%d' % jj, num_filters=nlabels, training=training, normalisation=norm, add_bias=False))

                # skip connection
                dec[jj].append(layers.crop_and_concat([dec[jj][-1], enc[ii-1][-1]], axis=3))

                dec[jj].append(layers.conv2D(dec[jj][-1], 'conv_%d_1' % jj, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=False))
                dec[jj].append(layers.conv2D(dec[jj][-1], 'conv_%d_2' % jj, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=False))

        output = layers.conv2D(dec[-1][-1], 'prediction', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training, normalisation=norm, add_bias=False)

        dec[-1].append(output)

        if return_net:
            net = enc + dec
            return output, net

        return output


def residual_unet2D(x, training, nlabels, n0=64, resolution_levels=4, norm=normalisation.batch_norm, scope_reuse=False, return_net=False):

    with tf.variable_scope('segmenter') as scope:

        if scope_reuse:
            scope.reuse_variables()

        enc = []

        input = layers.conv2D(x, 'input_layer', num_filters=n0, training=training, normalisation=norm, add_bias=False)

        with tf.variable_scope('encoder'):

            for ii in range(resolution_levels):

                enc.append([])

                # In first layer set input to x rather than max pooling
                if ii == 0:
                    enc[ii].append(input)
                else:
                    enc[ii].append(layers.maxpool2D(enc[ii-1][-1]))

                enc[ii].append(layers.residual_unit2D(enc[ii][-1], 'conv_%d_1' % ii, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=False))
                enc[ii].append(layers.residual_unit2D(enc[ii][-1], 'conv_%d_2' % ii, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=False))

        dec = []

        with tf.variable_scope('decoder'):

            for jj in range(resolution_levels-1):

                ii = resolution_levels - jj - 1  # used to index the encoder again

                dec.append([])

                if jj == 0:
                    next_inp = enc[ii][-1]
                else:
                    next_inp = dec[jj-1][-1]

                dec[jj].append(layers.transposed_conv2D(next_inp, name='upconv_%d' % jj, num_filters=nlabels, training=training, normalisation=norm, add_bias=False))

                # skip connection
                dec[jj].append(layers.crop_and_concat([dec[jj][-1], enc[ii-1][-1]], axis=3))

                dec[jj].append(layers.residual_unit2D(dec[jj][-1], 'conv_%d_1' % jj, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=False))
                dec[jj].append(layers.residual_unit2D(dec[jj][-1], 'conv_%d_2' % jj, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=False))

        output = layers.conv2D(dec[-1][-1], 'prediction', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training, normalisation=norm, add_bias=False)

        dec[-1].append(output)

        if return_net:
            net = enc + dec
            return output, net

        return output



def unet2D_modified_crfrnn(x, training, nlabels, n0=64, resolution_levels=4, norm=normalisation.batch_norm, scope_reuse=False, return_net=False):

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
                    enc[ii].append(layers.maxpool2D(enc[ii-1][-1]))

                enc[ii].append(layers.conv2D(enc[ii][-1], 'conv_%d_1' % ii, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=False))
                enc[ii].append(layers.conv2D(enc[ii][-1], 'conv_%d_2' % ii, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=False))

        dec = []

        with tf.variable_scope('decoder'):

            for jj in range(resolution_levels-1):

                ii = resolution_levels - jj - 1  # used to index the encoder again

                dec.append([])

                if jj == 0:
                    next_inp = enc[ii][-1]
                else:
                    next_inp = dec[jj-1][-1]

                dec[jj].append(layers.transposed_conv2D(next_inp, name='upconv_%d' % jj, num_filters=nlabels, training=training, normalisation=norm, add_bias=False))

                # skip connection
                dec[jj].append(layers.crop_and_concat([dec[jj][-1], enc[ii-1][-1]], axis=3))

                dec[jj].append(layers.conv2D(dec[jj][-1], 'conv_%d_1' % jj, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=False))
                dec[jj].append(layers.conv2D(dec[jj][-1], 'conv_%d_2' % jj, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=False))

        crfinput = layers.conv2D(dec[-1][-1], 'prediction', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training, normalisation=norm, add_bias=False)
        dec[-1].append(crfinput)

        output = layers.crfrnn(crfinput,
                               x,
                               num_classes=nlabels,
                               theta_alpha=160.,
                               theta_beta=3,
                               theta_gamma=10.,
                               num_iterations=5)

        if return_net:
            net = enc + dec
            return output, net

        return output