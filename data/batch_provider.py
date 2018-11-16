
import numpy as np
import utils

class BatchProvider():
    """
    This is a helper class to conveniently access mini batches of training, testing and validation data
    """

    def __init__(self, X, y, indices, normalise_images=False, map_to_unity_range=False, add_dummy_dimension=True, **kwargs):
        # indices don't always cover all of X and Y (e.g. in the case of val set)

        self.X = X
        self.y = y
        self.indices = indices
        self.unused_indices = indices.copy()
        self.normalise_images = normalise_images
        self.add_dummy_dimension = add_dummy_dimension
        self.map_to_unity_range = map_to_unity_range
        self.unity_percentile = kwargs.get('unity_percentile', 0.95)

        self.do_augmentations = kwargs.get('do_augmentations', False)
        self.augmentation_options = kwargs.get('augmentation_options', None)

        self.convert_to_gray = kwargs.get('convert_to_gray', False)



    def next_batch(self, batch_size):
        """
        Get a single random batch. This implements sampling without replacement (not just on a batch level), this means
        all the data gets sampled eventually.
        """

        if len(self.unused_indices) < batch_size:
            self.unused_indices = self.indices

        batch_indices = np.random.choice(self.unused_indices, batch_size, replace=False)
        self.unused_indices = np.setdiff1d(self.unused_indices, batch_indices)

        # HDF5 requires indices to be in increasing order
        batch_indices = np.sort(batch_indices)

        X_batch = np.float32(self.X[batch_indices, ...])
        y_batch = self.y[batch_indices, ...]

        X_batch, y_batch = self._post_process_batch(X_batch, y_batch)

        return X_batch, y_batch


    def iterate_batches(self, batch_size):
        """
        Get a range of batches. Use as argument of a for loop like you would normally use
        the range() function.
        """

        np.random.shuffle(self.indices)
        N = self.indices.shape[0]

        for b_i in range(0, N, batch_size):

            # if b_i + batch_size > N:
            #     continue

            # HDF5 requires indices to be in increasing order
            batch_indices = np.sort(self.indices[b_i:b_i + batch_size])

            X_batch = np.float32(self.X[batch_indices, ...])
            y_batch = self.y[batch_indices, ...]

            X_batch, y_batch = self._post_process_batch(X_batch, y_batch)

            yield X_batch, y_batch


    def _post_process_batch(self, X_batch, y_batch):

        if self.convert_to_gray:
            X_batch = np.mean(X_batch, axis=-1, keepdims=True)

        if self.do_augmentations:
            X_batch, y_batch = self._augmentation_function(X_batch, y_batch)

        if self.normalise_images:
            utils.normalise_images(X_batch)

        if self.map_to_unity_range:
            X_batch = utils.map_images_to_intensity_range(X_batch, -1, 1, percentiles=0.95)

        if self.add_dummy_dimension:
            X_batch = np.expand_dims(X_batch, axis=-1)

        return X_batch, y_batch


    def _augmentation_function(self, images, labels):
        '''
        Function for augmentation of minibatches. It will transform a set of images and corresponding labels
        by a number of optional transformations. Each image/mask pair in the minibatch will be seperately transformed
        with random parameters.
        :param images: A numpy array of shape [minibatch, X, Y, (Z), nchannels]
        :param labels: A numpy array containing a corresponding label mask
        :param do_rotations: Rotate the input images by a random angle between -15 and 15 degrees.
        :param do_scaleaug: Do scale augmentation by sampling one length of a square, then cropping and upsampling the image
                            back to the original size.
        :param do_fliplr: Perform random flips with a 50% chance in the left right direction.
        :return: A mini batch of the same size but with transformed images and masks.
        '''

        def get_option(name, default):
            return self.augmentation_options[name] if name in self.augmentation_options else default

        try:
            import cv2
        except:
            return False
        else:

            if images.ndim > 4:
                raise AssertionError('Augmentation will only work with 2D images')

            # If segmentation labels also augment them, otherwise don't
            augment_labels = True if labels.ndim > 1 else False

            do_rotations = get_option('do_rotations', False)
            do_scaleaug = get_option('do_scaleaug', False)
            do_fliplr = get_option('do_fliplr', False)
            do_flipud = get_option('do_flipud', False)
            augment_every_nth = get_option('augment_every_nth', 2)  # 2 means augment half of the images
                                                                    # 1 means augment every image

            new_images = []
            new_labels = []
            num_images = images.shape[0]

            for ii in range(num_images):

                img = np.squeeze(images[ii, ...])
                lbl = np.squeeze(labels[ii, ...])

                coin_flip = np.random.randint(augment_every_nth)
                if coin_flip == 0:

                    # ROTATE
                    if do_rotations:

                        angles = get_option('rot_degrees', 10.0)
                        random_angle = np.random.uniform(-angles, angles)
                        img = utils.rotate_image(img, random_angle)
                        if augment_labels:
                            lbl = utils.rotate_image(lbl, random_angle, interp=cv2.INTER_NEAREST)

                    # RANDOM CROP SCALE
                    if do_scaleaug:

                        offset = get_option('offset', 30)
                        n_x, n_y = img.shape
                        r_y = np.random.random_integers(n_y - offset, n_y)
                        p_x = np.random.random_integers(0, n_x - r_y)
                        p_y = np.random.random_integers(0, n_y - r_y)

                        img = utils.resize_image(img[p_y:(p_y + r_y), p_x:(p_x + r_y)], (n_x, n_y))
                        if augment_labels:
                            lbl = utils.resize_image(lbl[p_y:(p_y + r_y), p_x:(p_x + r_y)], (n_x, n_y), interp=cv2.INTER_NEAREST)

                # RANDOM FLIP
                if do_fliplr:
                    coin_flip = np.random.randint(max(2, augment_every_nth))  # Flipping wouldn't make sense if you do it always
                    if coin_flip == 0:
                        img = np.fliplr(img)
                        if augment_labels:
                            lbl = np.fliplr(lbl)

                if do_flipud:
                    coin_flip = np.random.randint(max(2, augment_every_nth))
                    if coin_flip == 0:
                        img = np.flipud(img)
                        if augment_labels:
                            lbl = np.flipud(lbl)

                new_images.append(img[...])
                new_labels.append(lbl[...])

            sampled_image_batch = np.asarray(new_images)
            sampled_label_batch = np.asarray(new_labels)

            return sampled_image_batch, sampled_label_batch


#