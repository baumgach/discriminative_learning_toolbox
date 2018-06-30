# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)

import os
import numpy as np
import logging
import gc
import h5py
from skimage import transform
from scipy import misc
import math
# import matplotlib.pyplot as plt

import utils

import pandas as pd
from sklearn.model_selection import train_test_split



logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

diagnosis_dict = {'No Finding': 0, 'Atelectasis': 1, 'Cardiomegaly': 2, 'Effusion': 3, 'Infiltration': 4,
                  'Infiltrate': 4, 'Mass': 5, 'Nodule': 6, 'Pneumonia': 7, 'Pneumothorax': 8, 'Consolidation': 9,
                  'Edema': 10, 'Emphysema': 11,'Fibrosis': 12, 'Pleural_Thickening': 13, 'Hernia': 14}
gender_dict = {'M': 0, 'F': 1}
view_dict = {'AP': 0, 'PA': 1}


# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 1000

def fix_nan_and_unknown(input, target_data_format=lambda x: x, nan_val=-1, unknown_val=-2):
    if math.isnan(float(input)):
        input = nan_val
    elif input == 'unknown':
        input = unknown_val

    return target_data_format(input)

def age_str_to_float(age_str):
    if age_str.endswith('Y'):
        return float(age_str.rstrip('Y'))
    elif age_str.endswith('M'):
        return float(age_str.rstrip('M')) / 12
    elif age_str.endswith('D'):
        return float(age_str.rstrip('D')) / 365
    else:
        raise ValueError('Undocumented age format: %s' % age_str)

def prepare_data(input_folder, output_file, image_size):

    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    csv_summary_file = os.path.join(input_folder, 'Data_Entry_2017.csv')
    bbox_summary_file = os.path.join(input_folder, 'BBox_List_2017.csv')

    summary = pd.read_csv(csv_summary_file)
    bboxes = pd.read_csv(bbox_summary_file)

    # Get list of unique rids
    pids = summary['Patient ID'].unique()

    bbox_filenames = bboxes['Image Index'].values
    bbox_pids = np.unique([int(tt.split('_')[0]) for tt in bboxes['Image Index']])
    # bbox_viscode = [tt.split('_')[1].rstrip('.png') for tt in bboxes['Image Index']]
    # bbox_findings = [diagnosis_dict[tt] for tt in bboxes['Finding Labels']]

    # Get initial diagnosis for rough stratification
    init_findings = []
    for pid in pids:
        init_finding = summary.loc[summary['Patient ID'] == pid]['Finding Labels'].values[0]
        init_finding = diagnosis_dict[init_finding.split('|')[0]]
        init_findings.append(init_finding)


    train_and_val_pids, test_pids, train_and_val_diagnoses, test_labels = train_test_split(pids, init_findings, test_size=0.2, stratify=init_findings)
    train_pids, val_pids = train_test_split(train_and_val_pids, test_size=0.2, stratify=train_and_val_diagnoses)

    print(len(train_pids), len(test_pids), len(val_pids))

    # Add the PIDS with bounding boxes to test set and remove from training and validation set
    test_pids = np.unique(np.concatenate([test_pids, bbox_pids]))
    train_pids = np.setdiff1d(train_pids, bbox_pids)
    val_pids = np.setdiff1d(val_pids, bbox_pids)

    print(len(train_pids), len(test_pids), len(val_pids))

    # n_images_train = len(summary.loc[summary['rid'].isin(train_rids)])
    # n_images_test = len(summary.loc[summary['rid'].isin(test_rids)])
    # n_images_val = len(summary.loc[summary['rid'].isin(val_rids)])

    hdf5_file = h5py.File(output_file, "w")

    findings_list = {'test': [], 'train': [], 'val': []}
    bbox_findings_list = {'test': [], 'train': [], 'val': []}
    bbox_list = {'test': [], 'train': [], 'val': []}
    bbox_presence_list = {'test': [], 'train': [], 'val': []}
    age_list = {'test': [], 'train': [], 'val': []}
    gender_list  = {'test': [], 'train': [], 'val': []}
    pid_list = {'test': [], 'train': [], 'val': []}
    viscode_list = {'test': [], 'train': [], 'val': []}
    view_list = {'test': [], 'train': [], 'val': []}

    file_list = {'test': [], 'train': [], 'val': []}

    logging.info('Counting files and parsing meta data...')

    for train_test, set_pids in zip(['train', 'test', 'val'], [train_pids, test_pids, val_pids]):

        for ii, row in summary.iterrows():

            file_name = row['Image Index']

            pid = row['Patient ID']
            if pid not in set_pids:
                continue  # pid is not in the train/test/val set

            # Finding
            finding = np.zeros(len(diagnosis_dict))
            fining_str = row['Finding Labels']
            finding_split = fining_str.split('|')
            finding_lbls = [diagnosis_dict[diag] for diag in finding_split]
            for lbl in finding_lbls:
                finding[lbl] = 1
            findings_list[train_test].append(finding)


            pid_list[train_test].append(int(pid))
            viscode_list[train_test].append(int(row['Follow-up #']))
            age_list[train_test].append(age_str_to_float(row['Patient Age']))
            gender_list[train_test].append(gender_dict[row['Patient Gender']])
            view_list[train_test].append(view_dict[row['View Position']])

            file_list[train_test].append(os.path.join(input_folder, 'images', file_name))

            if file_name in bbox_filenames:
                # logging.info('File %s has a bounding box!' % (file_name))

                scale_factor_x = float(image_size[0]) / 1024
                scale_factor_y = float(image_size[1]) / 1024

                bbox_row = bboxes.loc[bboxes['Image Index'] == file_name]
                bbox_findings_list[train_test].append(diagnosis_dict[bbox_row['Finding Label'].values[0]])
                bbox_x = bbox_row['Bbox x'].values[0] * scale_factor_x
                bbox_y = bbox_row['Bbox y'].values[0] * scale_factor_y
                bbox_w = bbox_row['Bbox w'].values[0] * scale_factor_x
                bbox_h = bbox_row['Bbox h'].values[0] * scale_factor_y
                bbox_list[train_test].append([bbox_x, bbox_y, bbox_w, bbox_h])
                bbox_presence_list[train_test].append(1)
            else:
                bbox_findings_list[train_test].append(0)
                bbox_list[train_test].append([0, 0, 0, 0])
                bbox_presence_list[train_test].append(0)


    # Write the small datasets
    for tt in ['test', 'train', 'val']:

        hdf5_file.create_dataset('pid_%s' % tt, data=np.asarray(pid_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('viscode_%s' % tt, data=np.asarray(viscode_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('findings_%s' % tt, data=np.asarray(findings_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('age_%s' % tt, data=np.asarray(age_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('gender_%s' % tt, data=np.asarray(gender_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('view_%s' % tt, data=np.asarray(view_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('bbox_presence_%s' % tt, data=np.asarray(bbox_presence_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('bbox_findings_%s' % tt, data=np.asarray(bbox_findings_list[tt], dtype=np.uint8))
        hdf5_file.create_dataset('bbox_coords_%s' % tt, data=np.asarray(bbox_list[tt], dtype=np.float32))


    n_train = len(file_list['train'])
    n_test = len(file_list['test'])
    n_val = len(file_list['val'])

    print(n_train, n_test, n_val)

    # assert n_train == n_images_train, 'Mismatch in data sizes, %d not == %d' % (n_train, n_images_train)
    # assert n_test == n_images_test, 'Mismatch in data sizes, %d not == %d' % (n_test, n_images_test)
    # assert n_val == n_images_val, 'Mismatch in data sizes, %d not == %d' % (n_val, n_images_val)

    # Create datasets for images and masks
    data = {}
    for tt, num_points in zip(['test', 'train', 'val'], [n_test, n_train, n_val]):
        data['images_%s' % tt] = hdf5_file.create_dataset("images_%s" % tt, [num_points] + list(image_size), dtype=np.uint8)

    img_list = {'test': [], 'train': [] , 'val': []}

    logging.info('Parsing image files')

    for train_test in ['test', 'train', 'val']:

        logging.info('******* Doing %s **********' % train_test)

        write_buffer = 0
        counter_from = 0

        for ii, file in enumerate(file_list[train_test]):

            if ii % 100 == 0:
                logging.info('Completed %d in this train/test/val category' % ii)

            image = misc.imread(file)

            image = transform.resize(image,
                                     image_size,
                                     order=1,
                                     preserve_range=True,
                                     mode='reflect')

            image = image.astype(np.uint8)

            if len(image.shape) > 2:

                image = np.squeeze(image[:,:,0])

                # fig = plt.figure()
                # fig.add_subplot(141)
                # plt.imshow(image[:,:,0])
                # fig.add_subplot(142)
                # plt.imshow(image[:,:,1])
                # fig.add_subplot(143)
                # plt.imshow(image[:,:,2])
                # fig.add_subplot(144)
                # plt.imshow(image[:,:,3])
                # plt.show()

            img_list[train_test].append(image)

            write_buffer += 1

            if write_buffer >= MAX_WRITE_BUFFER:

                counter_to = counter_from + write_buffer
                _write_range_to_hdf5(data, train_test, img_list, counter_from, counter_to)
                _release_tmp_memory(img_list, train_test)

                # reset stuff for next iteration
                counter_from = counter_to
                write_buffer = 0


        # after file loop: Write the remaining data
        logging.info('Writing remaining data')
        counter_to = counter_from + write_buffer

        _write_range_to_hdf5(data, train_test, img_list, counter_from, counter_to)
        _release_tmp_memory(img_list, train_test)


    # After test train loop:
    hdf5_file.close()


def _write_range_to_hdf5(hdf5_data, train_test, img_list, counter_from, counter_to, dtype=np.uint8):
    '''
    Helper function to write a range of data to the hdf5 datasets
    '''

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))
    img_arr = np.asarray(img_list[train_test], dtype=dtype)
    hdf5_data['images_%s' % train_test][counter_from:counter_to, ...] = img_arr



def _release_tmp_memory(img_list, train_test):
    '''
    Helper function to reset the tmp lists and free the memory
    '''

    img_list[train_test].clear()
    gc.collect()


def load_and_maybe_generate_data(input_folder,
                                 preprocessing_folder,
                                 image_size,
                                 force_overwrite=False):

    '''
    This function is used to load and if necessary preprocesses the ACDC challenge data
    
    :param input_folder: Folder where the raw ACDC challenge data is located 
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :param image_size: Size of the output slices/volumes in pixels/voxels
    :param target_resolution: Resolution to which the data should resampled. Should have same shape as size
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]
     
    :return: Returns an h5py.File handle to the dataset
    '''

    size_str = '_'.join([str(i) for i in image_size])


    data_file_name = 'chestx_data_size_%s.hdf5' % (size_str)
    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder, data_file_path, image_size)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


if __name__ == '__main__':


    print('nop nop nop...')