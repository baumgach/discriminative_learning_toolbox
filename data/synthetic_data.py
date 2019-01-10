# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import numpy as np
from sklearn.model_selection import train_test_split
from data import synthetic_data_loader
from data.batch_provider import BatchProvider

class synthetic_data():

    def __init__(self, exp_config):

        data = synthetic_data_loader.load_and_maybe_generate_data(output_folder=exp_config.preproc_folder,
                                                                  image_size=exp_config.image_size,
                                                                  effect_size=exp_config.effect_size,
                                                                  num_samples=exp_config.num_samples,
                                                                  moving_effect=exp_config.moving_effect,
                                                                  scale_to_one=exp_config.rescale_to_one,
                                                                  force_overwrite=False)

        self.data = data

        lhr_size = data['features'].shape[0]
        imsize = int(np.sqrt(lhr_size))

        images = np.reshape(data['features'][:], [imsize, imsize, -1])
        images = np.transpose(images, [2, 0, 1])

        masks = np.reshape(data['gt'][:], [imsize, imsize, -1])
        masks = np.transpose(masks, [2, 0, 1])

        labels = data['labels'][:]

        images_train_and_val, images_test, \
        labels_train_and_val, labels_test, \
        masks_train_and_val, masks_test = train_test_split(images,
                                                           labels,
                                                           masks,
                                                           test_size=0.2,
                                                           stratify=labels,
                                                           random_state=42)

        images_train, images_val, \
        labels_train, labels_val, \
        masks_train, masks_val = train_test_split(images_train_and_val,
                                                  labels_train_and_val,
                                                  masks_train_and_val,
                                                  test_size=0.2,
                                                  stratify=labels_train_and_val,
                                                  random_state=42)



        if exp_config.label_type == 'image_level':
            labels_use_train = labels_train
            labels_use_val = labels_val
            labels_use_test = labels_test
        elif exp_config.label_type == 'mask':
            labels_use_train = masks_train
            labels_use_val = masks_val
            labels_use_test = masks_test
        else:
            raise ValueError("Unknown label_type '%s' in exp_config" % exp_config.label_type)

        self.images_test = images_test
        self.labels_test = labels_use_test

        N_train = images_train.shape[0]
        N_test = images_test.shape[0]
        N_val = images_val.shape[0]

        train_indices = np.arange(N_train)
        train_c1_indices = train_indices[np.where(labels_train == 1)]
        train_c0_indices = train_indices[np.where(labels_train == 0)]

        test_indices = np.arange(N_test)
        test_c1_indices = test_indices[np.where(labels_test == 1)]
        test_c0_indices = test_indices[np.where(labels_test == 0)]

        val_indices = np.arange(N_val)
        val_c1_indices = val_indices[np.where(labels_val == 1)]
        val_c0_indices = val_indices[np.where(labels_val == 0)]

        # Create the batch providers

        augmentation_options = exp_config.augmentation_options
        augmentation_options['nlabels'] = exp_config.nlabels

        self.train_c1 = BatchProvider(images_train, labels_use_train, train_c1_indices,
                                      do_augmentations=exp_config.do_augmentations,
                                      augmentation_options=augmentation_options)
        self.train_c0 = BatchProvider(images_train, labels_use_train, train_c0_indices,
                                      do_augmentations=exp_config.do_augmentations,
                                      augmentation_options=augmentation_options)

        self.validation_c1 = BatchProvider(images_val, labels_use_val, val_c1_indices)
        self.validation_c0 = BatchProvider(images_val, labels_use_val, val_c0_indices)

        self.test_c1 = BatchProvider(images_test, labels_use_test, test_c1_indices)
        self.test_c0 = BatchProvider(images_test, labels_use_test, test_c0_indices)


        self.train = BatchProvider(images_train, labels_train, train_indices,
                                   do_augmentations=exp_config.do_augmentations,
                                   augmentation_options=augmentation_options)
        self.validation = BatchProvider(images_val, labels_use_val, val_indices)
        self.test = BatchProvider(images_test, labels_use_test, test_indices)



