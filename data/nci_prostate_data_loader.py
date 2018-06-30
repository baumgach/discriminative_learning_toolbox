# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import os
import glob
import numpy as np
import logging
import nibabel as nib
import gc
import h5py
from skimage import transform
import pydicom as dicom
import nrrd

import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Dictionary to translate a diagnosis into a number
# NOR  - Normal
# MINF - Previous myiocardial infarction (EF < 40%)
# DCM  - Dialated Cardiomypopathy
# HCM  - Hypertrophic cardiomyopathy
# RV   - Abnormal right ventricle (high volume or low EF)
diagnosis_dict = {'NOR': 0, 'MINF': 1, 'DCM': 2, 'HCM': 3, 'RV': 4}

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5


def crop_or_pad_slice_to_size(slice, nx, ny):
    x, y = slice.shape

    x_s = (x - nx) // 2
    y_s = (y - ny) // 2
    x_c = (nx - x) // 2
    y_c = (ny - y) // 2

    if x > nx and y > ny:
        slice_cropped = slice[x_s:x_s + nx, y_s:y_s + ny]
    else:
        slice_cropped = np.zeros((nx, ny))
        if x <= nx and y > ny:
            slice_cropped[x_c:x_c + x, :] = slice[:, y_s:y_s + ny]
        elif x > nx and y <= ny:
            slice_cropped[:, y_c:y_c + y] = slice[x_s:x_s + nx, :]
        else:
            slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice[:, :]

    return slice_cropped


def test_train_val_split(patient_id):

    if patient_id % 5 == 0:
        return 'test'
    elif patient_id % 4 == 0:
        return 'validation'
    else:
        return 'train'

def count_slices(input_folder, folder_base):

    num_slices = {'train': 0, 'test': 0, 'validation': 0}

    for folder in os.listdir(input_folder):
        if not folder.startswith(folder_base):
            continue

        patient_id = int(folder.split('-')[-1])
        if patient_id > 30:
            continue

        # print(folder)
        # print(patient_id)
        # print(os.path.join(input_folder, folder))
        # print('--')

        path = os.path.join(input_folder, folder)
        for dirName, subdirList, fileList in os.walk(path):
            for filename in fileList:
                if filename.lower().endswith('.dcm'):  # check whether the file's DICOM

                    train_test = test_train_val_split(patient_id)

                    num_slices[train_test] += 1

    return num_slices


def get_patient_folders(input_folder, folder_base):

    folder_list = {'train': [], 'test': [], 'validation': []}

    for folder in os.listdir(input_folder):
        if folder.startswith('Prostate3T-01'):
            patient_id = int(folder.split('-')[-1])
            if patient_id > 30:
                continue
            train_test = test_train_val_split(patient_id)
            folder_list[train_test].append(os.path.join(input_folder, folder))

    return folder_list


def prepare_data(input_folder, output_file, mode, size, target_resolution):
    '''
    Main function that prepares a dataset from the raw challenge data to an hdf5 dataset
    '''

    assert (mode in ['2D', '3D']), 'Unknown mode: %s' % mode
    if mode == '2D' and not len(size) == 2:
        raise AssertionError('Inadequate number of size parameters')
    if mode == '3D' and not len(size) == 3:
        raise AssertionError('Inadequate number of size parameters')
    if mode == '2D' and not len(target_resolution) == 2:
        raise AssertionError('Inadequate number of target resolution parameters')
    if mode == '3D' and not len(target_resolution) == 3:
        raise AssertionError('Inadequate number of target resolution parameters')

    image_folder = os.path.join(input_folder, 'Prostate-3T')
    mask_folder = os.path.join(input_folder, 'NCI_ISBI_Challenge-Prostate3T_Training_Segmentations')

    hdf5_file = h5py.File(output_file, "w")

    logging.info('Counting files and parsing meta data...')
    folder_list = get_patient_folders(image_folder, folder_base='Prostate3T-01')

    if mode == '3D':
        nx, ny, nz_max = size
        n_train = len(folder_list['train'])
        n_test = len(folder_list['test'])
        n_val = len(folder_list['validation'])
    elif mode == '2D':
        num_slices = count_slices(image_folder, folder_base='Prostate3T-01')
        nx, ny = size
        n_test = num_slices['test']
        n_train = num_slices['train']
        n_val = num_slices['validation']
    else:
        raise AssertionError('Wrong mode setting. This should never happen.')

    print('Debug: Check if sets add up to correct value:')
    print(n_train, n_val, n_test, n_train + n_val + n_test)

    # Create datasets for images and masks
    data = {}
    for tt, num_points in zip(['test', 'train', 'validation'], [n_test, n_train, n_val]):

        if num_points > 0:
            data['images_%s' % tt] = hdf5_file.create_dataset("images_%s" % tt, [num_points] + list(size), dtype=np.float32)
            data['masks_%s' % tt] = hdf5_file.create_dataset("masks_%s" % tt, [num_points] + list(size), dtype=np.uint8)


    mask_list = {'test': [], 'train': [], 'validation': []}
    img_list = {'test': [], 'train': [], 'validation': []}

    logging.info('Parsing image files')

    for train_test in ['test', 'train', 'validation']:

        write_buffer = 0
        counter_from = 0

        patient_counter = 0

        for folder in folder_list[train_test]:

            patient_counter += 1

            logging.info('-----------------------------------------------------------')
            logging.info('Doing: %s' % folder)

            lstFilesDCM = []  # create an empty list
            for dirName, subdirList, fileList in os.walk(folder):
                # fileList.sort()
                for filename in fileList:
                    if ".dcm" in filename.lower():  # check whether the file's DICOM
                        lstFilesDCM.append(os.path.join(dirName, filename))

            # Get ref file
            RefDs = dicom.read_file(lstFilesDCM[0])

            # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
            ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

            # Load spacing values (in mm)
            pixel_size = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
            # print("pixel spacing 0,1; slice thickness ",ConstPixelSpacing)

            print('PixelDims')
            print(ConstPixelDims)
            print('PixelSpacing')
            print(pixel_size)

            # The array is sized based on 'ConstPixelDims'
            img = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

            # loop through all the DICOM files
            for filenameDCM in lstFilesDCM:
                # read the file
                ds = dicom.read_file(filenameDCM)
                # store the raw image data
                # img[:, :, lstFilesDCM.index(filenameDCM)] = ds.pixel_array # index number field is not set correctly ! instead instance no is the slice no !
                img[:, :, ds.InstanceNumber - 1] = ds.pixel_array

            mask_path = os.path.join(mask_folder, folder.split('/')[-1] + '.nrrd')
            mask, options = nrrd.read(mask_path)

            # fix swap axis
            mask = np.swapaxes(mask, 0, 1)

            print('mask.shape')
            print(mask.shape)
            print('img.shape')
            print(img.shape)

            ### PROCESSING LOOP FOR SLICE-BY-SLICE 3D DATA ###################
            if mode == '3D':

                scale_vector = [pixel_size[0] / target_resolution[0],
                                pixel_size[1] / target_resolution[1],
                                pixel_size[2] / target_resolution[2]]

                img_scaled = transform.rescale(img,
                                               scale_vector,
                                               order=1,
                                               preserve_range=True,
                                               multichannel=False,
                                               mode='constant')
                mask_scaled = transform.rescale(mask,
                                                scale_vector,
                                                order=0,
                                                preserve_range=True,
                                                multichannel=False,
                                                mode='constant')

                slice_vol = np.zeros((nx, ny, nz_max), dtype=np.float32)
                mask_vol = np.zeros((nx, ny, nz_max), dtype=np.uint8)

                nz_curr = img_scaled.shape[2]
                stack_from = (nz_max - nz_curr) // 2

                if stack_from < 0:
                    raise AssertionError('nz_max is too small for the chosen through plane resolution. Consider changing'
                                         'the size or the target resolution in the through-plane.')

                for zz in range(nz_curr):

                    slice_rescaled = img_scaled[:,:,zz]
                    mask_rescaled = mask_scaled[:,:,zz]

                    slice_cropped = crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
                    mask_cropped = crop_or_pad_slice_to_size(mask_rescaled, nx, ny)

                    slice_vol[:,:,stack_from] = slice_cropped
                    mask_vol[:,:,stack_from] = mask_cropped

                    stack_from += 1

                img_list[train_test].append(slice_vol)
                mask_list[train_test].append(mask_vol)

                write_buffer += 1

                if write_buffer >= MAX_WRITE_BUFFER:

                    counter_to = counter_from + write_buffer
                    _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to)
                    _release_tmp_memory(img_list, mask_list, train_test)

                    # reset stuff for next iteration
                    counter_from = counter_to
                    write_buffer = 0

            ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################
            elif mode == '2D':

                scale_vector = [pixel_size[0] / target_resolution[0], pixel_size[1] / target_resolution[1]]

                for zz in range(img.shape[2]):

                    slice_img = np.squeeze(img[:, :, zz])
                    slice_rescaled = transform.rescale(slice_img,
                                                       scale_vector,
                                                       order=1,
                                                       preserve_range=True,
                                                       multichannel=False,
                                                       mode = 'constant')

                    slice_mask = np.squeeze(mask[:, :, zz])
                    mask_rescaled = transform.rescale(slice_mask,
                                                      scale_vector,
                                                      order=0,
                                                      preserve_range=True,
                                                      multichannel=False,
                                                      mode='constant')

                    slice_cropped = crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
                    mask_cropped = crop_or_pad_slice_to_size(mask_rescaled, nx, ny)

                    img_list[train_test].append(slice_cropped)
                    mask_list[train_test].append(mask_cropped)

                    write_buffer += 1

                    # Writing needs to happen inside the loop over the slices
                    if write_buffer >= MAX_WRITE_BUFFER:

                        counter_to = counter_from + write_buffer
                        _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to)
                        _release_tmp_memory(img_list, mask_list, train_test)

                        # reset stuff for next iteration
                        counter_from = counter_to
                        write_buffer = 0


        logging.info('Writing remaining data')
        counter_to = counter_from + write_buffer

        _write_range_to_hdf5(data, train_test, img_list, mask_list, counter_from, counter_to)
        _release_tmp_memory(img_list, mask_list, train_test)

    # After test train loop:
    hdf5_file.close()


def _write_range_to_hdf5(hdf5_data, train_test, img_list, mask_list, counter_from, counter_to):
    '''
    Helper function to write a range of data to the hdf5 datasets
    '''

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    img_arr = np.asarray(img_list[train_test], dtype=np.float32)
    mask_arr = np.asarray(mask_list[train_test], dtype=np.uint8)

    hdf5_data['images_%s' % train_test][counter_from:counter_to, ...] = img_arr
    hdf5_data['masks_%s' % train_test][counter_from:counter_to, ...] = mask_arr


def _release_tmp_memory(img_list, mask_list, train_test):
    '''
    Helper function to reset the tmp lists and free the memory
    '''

    img_list[train_test].clear()
    mask_list[train_test].clear()
    gc.collect()


def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                mode,
                                size,
                                target_resolution,
                                force_overwrite=False):
    '''
    This function is used to load and if necessary preprocesses the ACDC challenge data

    :param input_folder: Folder where the raw ACDC challenge data is located
    :param preprocessing_folder: Folder where the proprocessed data should be written to
    :param mode: Can either be '2D' or '3D'. 2D saves the data slice-by-slice, 3D saves entire volumes
    :param size: Size of the output slices/volumes in pixels/voxels
    :param target_resolution: Resolution to which the data should resampled. Should have same shape as size
    :param force_overwrite: Set this to True if you want to overwrite already preprocessed data [default: False]

    :return: Returns an h5py.File handle to the dataset
    '''

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'data_%s_size_%s_res_%s.hdf5' % (mode, size_str, res_str)

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder, data_file_path, mode, size, target_resolution)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')


if __name__ == '__main__':
    input_folder = '/scratch_net/bmicdl03/data/ACDC_challenge_20170617'
    preprocessing_folder = 'preproc_data'

    d = load_and_maybe_process_data(input_folder, preprocessing_folder, (256, 256), (1.36719, 1.36719),
                                    force_overwrite=True)

