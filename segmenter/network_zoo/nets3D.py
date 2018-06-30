# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)


import tensorflow as tf
from tfwrapper import layers
from tfwrapper import normalisation as tfnorm


def unet3D(x, training, nlabels, n0=32, resolution_levels=3, norm=tfnorm.batch_norm, scope_reuse=False, return_net=False):

    with tf.variable_scope('segmenter') as scope:

        if scope_reuse:
            scope.reuse_variables()

        add_bias = False if norm == tfnorm.batch_norm else True

        enc = []

        with tf.variable_scope('encoder'):

            for ii in range(resolution_levels):

                enc.append([])

                # In first layer set input to x rather than max pooling
                if ii == 0:
                    enc[ii].append(x)
                else:
                    enc[ii].append(layers.maxpool3D(enc[ii-1][-1]))

                enc[ii].append(layers.conv3D(enc[ii][-1], 'conv_%d_1' % ii, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=add_bias))
                enc[ii].append(layers.conv3D(enc[ii][-1], 'conv_%d_2' % ii, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=add_bias))

        dec = []

        with tf.variable_scope('decoder'):

            for jj in range(resolution_levels-1):

                ii = resolution_levels - jj - 1  # used to index the encoder again

                dec.append([])

                if jj == 0:
                    next_inp = enc[ii][-1]
                else:
                    next_inp = dec[jj-1][-1]

                dec[jj].append(layers.transposed_conv3D(next_inp, name='upconv_%d' % jj, num_filters=nlabels, training=training, normalisation=norm, add_bias=add_bias))

                # skip connection
                dec[jj].append(layers.crop_and_concat([dec[jj][-1], enc[ii-1][-1]], axis=3))

                dec[jj].append(layers.conv3D(dec[jj][-1], 'conv_%d_1' % jj, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=add_bias))
                dec[jj].append(layers.conv3D(dec[jj][-1], 'conv_%d_2' % jj, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=add_bias))

        output = layers.conv2D(dec[-1][-1], 'prediction', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training, normalisation=norm, add_bias=add_bias)

        dec[-1].append(output)

        if return_net:
            net = enc + dec
            return output, net

        return output