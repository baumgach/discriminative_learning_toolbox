# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import tensorflow as tf
from tfwrapper import layers
from tfwrapper import normalisation as tfnorm

def unet2D(x,
           training,
           nlabels,
           n0=64,
           resolution_levels=5,
           norm=tfnorm.batch_norm,
           conv_unit=layers.conv2D,
           deconv_unit=layers.transposed_conv2D,
           simplified_dec=False,
           scope_reuse=False,
           return_net=False):

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
                    enc[ii].append(layers.maxpool2D(enc[ii-1][-1]))

                enc[ii].append(conv_unit(enc[ii][-1], 'conv_%d_1' % ii, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=add_bias))
                enc[ii].append(conv_unit(enc[ii][-1], 'conv_%d_2' % ii, num_filters=n0*(2**ii), training=training, normalisation=norm, add_bias=add_bias))

        dec = []

        with tf.variable_scope('decoder'):

            for jj in range(resolution_levels-1):

                ii = resolution_levels - jj - 1  # used to index the encoder again

                dec.append([])

                if jj == 0:
                    next_inp = enc[ii][-1]
                else:
                    next_inp = dec[jj-1][-1]

                if simplified_dec:
                    num_transconv_filters = nlabels
                else:
                    num_transconv_filters = n0*(2**(ii-1))

                dec[jj].append(deconv_unit(next_inp, name='upconv_%d' % jj, num_filters=num_transconv_filters, training=training, normalisation=norm, add_bias=add_bias))

                # skip connection
                dec[jj].append(layers.crop_and_concat([dec[jj][-1], enc[ii-1][-1]], axis=3))

                dec[jj].append(conv_unit(dec[jj][-1], 'conv_%d_1' % jj, num_filters=n0*(2**(ii-1)), training=training, normalisation=norm, add_bias=add_bias, projection=True))  # projection True to make it work with res units.
                dec[jj].append(conv_unit(dec[jj][-1], 'conv_%d_2' % jj, num_filters=n0*(2**(ii-1)), training=training, normalisation=norm, add_bias=add_bias))

        output = layers.conv2D(dec[-1][-1], 'prediction', num_filters=nlabels, kernel_size=(1,1), activation=tf.identity, training=training, normalisation=norm, add_bias=add_bias)

        dec[-1].append(output)

        if return_net:
            net = enc + dec
            return output, net

        return output


def res_unet2D(x, training, nlabels, n0=64, resolution_levels=5, norm=tfnorm.batch_norm, scope_reuse=False, return_net=False):

    add_bias = False if norm == tfnorm.batch_norm else True
    input_layer = layers.conv2D(x, training=training, num_filters=n0, name='input_layer', normalisation=norm, add_bias=add_bias)

    return unet2D(input_layer,
                  training,
                  nlabels,
                  n0=n0,
                  resolution_levels=resolution_levels,
                  norm=norm,
                  conv_unit=layers.residual_unit2D,
                  scope_reuse=scope_reuse,
                  return_net=return_net)


def id_res_unet2D(x, training, nlabels, n0=64, resolution_levels=5, norm=tfnorm.batch_norm, scope_reuse=False, return_net=False):

    add_bias = False if norm == tfnorm.batch_norm else True
    input_layer = layers.conv2D(x, training=training, num_filters=n0, name='input_layer', normalisation=norm, add_bias=add_bias)

    return unet2D(input_layer,
                  training,
                  nlabels,
                  n0=n0,
                  resolution_levels=resolution_levels,
                  norm=norm,
                  conv_unit=layers.identity_residual_unit2D,
                  scope_reuse=scope_reuse,
                  return_net=return_net)


def unet2D_crfrnn(x, training, nlabels, n0=64, resolution_levels=5, norm=tfnorm.batch_norm, scope_reuse=False):

        crfinput = unet2D(x,
                          training,
                          nlabels,
                          n0=n0,
                          resolution_levels=resolution_levels,
                          norm=norm,
                          conv_unit=layers.conv2D,
                          scope_reuse=scope_reuse,
                          return_net=False)

        with tf.variable_scope('segmenter'):

            output = layers.crfrnn(crfinput,
                                   x,
                                   num_classes=nlabels,
                                   theta_alpha=160.,
                                   theta_beta=3,
                                   theta_gamma=10.,
                                   num_iterations=5)

        return output
