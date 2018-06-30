# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import numpy as np
from data import acdc_data_loader
from data.batch_provider import BatchProvider

class acdc_data():

    def __init__(self, exp_config):

        data = acdc_data_loader.load_and_maybe_process_data(
            input_folder=exp_config.data_root,
            preprocessing_folder=exp_config.preproc_folder,
            size=exp_config.image_size,
            target_resolution=exp_config.target_resolution,
            force_overwrite=False,
        )

        self.data = data

        label_name = 'masks'

        # the following are HDF5 datasets, not numpy arrays
        images_train = data['images_train']
        labels_train = data['%s_train' % label_name]

        images_test = data['images_test']
        labels_test = data['%s_test' % label_name]

        images_val = data['images_val']
        labels_val = data['%s_validation' % label_name]

        # Extract the number of training and testing points
        N_train = images_train.shape[0]
        N_test = images_test.shape[0]
        N_val = images_val.shape[0]

        # Create a shuffled range of indices for both training and testing data
        train_indices = np.arange(N_train)
        test_indices = np.arange(N_test)
        val_indices = np.arange(N_val)

        # Create the batch providers
        self.train = BatchProvider(images_train, labels_train, train_indices)
        self.validation = BatchProvider(images_val, labels_val, val_indices)
        self.test = BatchProvider(images_test, labels_test, test_indices)
