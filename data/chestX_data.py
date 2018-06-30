import numpy as np
from data import chestX_data_loader
from data.batch_provider import BatchProvider
# from IPython.core.debugger import Pdb as pdb
# import pdb

np.random.seed(42)

class chestX_data():

    def __init__(self, exp_config):

        data = chestX_data_loader.load_and_maybe_generate_data(
            input_folder=exp_config.data_root,
            preprocessing_folder=exp_config.preproc_folder,
            image_size=exp_config.image_size,
            force_overwrite=False,
        )

        self.data = data

        label_name = 'findings'

        # the following are HDF5 datasets, not numpy arrays
        images_train = data['images_train']
        images_test = data['images_test']
        images_val = data['images_val']

        train_labels = data['%s_train' % label_name]
        test_labels = data['%s_test' % label_name]
        val_labels = data['%s_val' % label_name]

        # Extract the number of training and testing points
        N_train = images_train.shape[0]
        N_test = images_test.shape[0]
        N_val = images_val.shape[0]

        # Create a shuffled range of indices for both training and testing data
        train_indices = np.arange(N_train)
        test_indices = np.arange(N_test)
        val_indices = np.arange(N_val)

        # def _filter_by_labels(labels, indices, background_labels, disease_labels):
        #
        #     N_pts = labels.shape[0]
        #
        #     indicators_0 = labels[:,background_labels].sum(axis=1)>0
        #     indicators_1 = labels[:, disease_labels].sum(axis=1) > 0
        #
        #     indicators_0[indicators_1] = 0  # this is to make sure nothing can be disease and background at the same time
        #
        #     indices_0 = indices[indicators_0]
        #     indices_1 = indices[indicators_1]
        #
        #     indices = np.concatenate([indices_0, indices_1])
        #
        #     labels = 255*np.ones(N_pts, dtype=np.uint8)
        #     labels[indicators_0] = 0
        #     labels[indicators_1] = 1
        #     # Note that 255 should never be retrieved, it acts as a flag for debugging. If 255 pops up in a batch
        #     # then something's wrong.
        #
        #     indices = np.sort(indices)
        #
        #     return indices, labels



        # train_indices_select, train_labels = _filter_by_labels(train_labels, train_indices, exp_config.background_labels, exp_config.disease_labels)
        # val_indices_select, val_labels = _filter_by_labels(val_labels, val_indices, exp_config.background_labels, exp_config.disease_labels)
        # test_indices_select, test_labels = _filter_by_labels(test_labels, test_indices, exp_config.background_labels, exp_config.disease_labels)
        #
        # print('executed this..')
        #
        # train_c1_indices = train_indices[np.where(train_labels[:] == 1)]
        # train_c0_indices = train_indices[np.where(train_labels[:] == 0)]
        #
        # test_c1_indices = test_indices[np.where(test_labels[:] == 1)]
        # test_c0_indices = test_indices[np.where(test_labels[:] == 0)]
        #
        # val_c1_indices = val_indices[np.where(val_labels[:] == 1)]
        # val_c0_indices = val_indices[np.where(val_labels[:] == 0)]

        # pdb.set_trace()

        # assert len(np.unique(val_indices_select)) == len(val_indices_select), 'Debugging flag violated: indices not unique'
        #
        #
        # # Create the batch providers
        # self.train_c1 = BatchProvider(images_train, train_labels, train_c1_indices,
        #                               add_dummy_dimension=True,
        #                               normalise_images=True,
        #                               map_to_unity_range=exp_config.rescale_to_one)
        # self.train_c0 = BatchProvider(images_train, train_labels, train_c0_indices,
        #                               add_dummy_dimension=True,
        #                               normalise_images=True,
        #                               map_to_unity_range=exp_config.rescale_to_one)
        #
        # self.validation_c1 = BatchProvider(images_val, val_labels, val_c1_indices,
        #                                    add_dummy_dimension=True,
        #                                    normalise_images=True,
        #                                    map_to_unity_range=exp_config.rescale_to_one)
        # self.validation_c0 = BatchProvider(images_val, val_labels, val_c0_indices,
        #                                    add_dummy_dimension=True,
        #                                    normalise_images=True,
        #                                    map_to_unity_range=exp_config.rescale_to_one)
        #
        # self.test_c1 = BatchProvider(images_test, test_labels, test_c1_indices,
        #                              add_dummy_dimension=True,
        #                              normalise_images=True,
        #                              map_to_unity_range=exp_config.rescale_to_one)
        # self.test_c0 = BatchProvider(images_test, test_labels, test_c0_indices,
        #                              add_dummy_dimension=True,
        #                              normalise_images=True,
        #                              map_to_unity_range=exp_config.rescale_to_one)

        self.train = BatchProvider(images_train, train_labels, train_indices,
                                   add_dummy_dimension=True,
                                   normalise_images=True,
                                   map_to_unity_range=exp_config.rescale_to_one)
        self.validation = BatchProvider(images_val, val_labels, val_indices,
                                        add_dummy_dimension=True,
                                        normalise_images=True,
                                        map_to_unity_range=exp_config.rescale_to_one)
        self.test = BatchProvider(images_test, test_labels, test_indices,
                                  add_dummy_dimension=True,
                                  normalise_images=True,
                                  map_to_unity_range=exp_config.rescale_to_one)

