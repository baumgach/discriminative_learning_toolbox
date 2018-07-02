# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)
# Lisa M. Koch (lisa.margret.koch@gmail.com)

import numpy as np
import time
from skimage import transform
import os
import glob
from importlib.machinery import SourceFileLoader
import argparse
from medpy.metric import dc, assd, hd

import config.system as sys_config
from segmenter.model_segmenter import segmenter as segmenter

import logging
from data.data_switch import data_switch
import utils

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Set SGE_GPU environment variable if we are not on the local host
sys_config.setup_GPU_environment()


def main(input_folder, output_folder, model_path, exp_config, do_postprocessing=False, gt_exists=True):

    # Get Data
    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config)

    # Make and restore vagan model
    segmenter_model = segmenter(exp_config=exp_config, data=data, fixed_batch_size=1)  # CRF model requires fixed batch size
    segmenter_model.load_weights(model_path, type='best_dice')

    total_time = 0
    total_volumes = 0

    dice_list = []
    assd_list = []
    hd_list = []

    for folder in os.listdir(input_folder):

        folder_path = os.path.join(input_folder, folder)

        if os.path.isdir(folder_path):

            infos = {}
            for line in open(os.path.join(folder_path, 'Info.cfg')):
                label, value = line.split(':')
                infos[label] = value.rstrip('\n').lstrip(' ')

            patient_id = folder.lstrip('patient')

            if not int(patient_id) % 5 == 0:
                continue

            ED_frame = int(infos['ED'])
            ES_frame = int(infos['ES'])

            for file in glob.glob(os.path.join(folder_path, 'patient???_frame??.nii.gz')):

                logging.info(' ----- Doing image: -------------------------')
                logging.info('Doing: %s' % file)
                logging.info(' --------------------------------------------')

                file_base = file.split('.nii.gz')[0]

                frame = int(file_base.split('frame')[-1])
                img, img_affine, img_header = utils.load_nii(file)
                img = utils.normalise_image(img)
                zooms = img_header.get_zooms()

                if gt_exists:
                    file_mask = file_base + '_gt.nii.gz'
                    mask, mask_affine, mask_header = utils.load_nii(file_mask)

                start_time = time.time()

                if exp_config.dimensionality_mode == '2D':

                    pixel_size = (img_header.structarr['pixdim'][1], img_header.structarr['pixdim'][2])
                    scale_vector = (pixel_size[0] / exp_config.target_resolution[0],
                                    pixel_size[1] / exp_config.target_resolution[1])

                    predictions = []

                    nx, ny = exp_config.image_size

                    for zz in range(img.shape[2]):

                        slice_img = np.squeeze(img[:,:,zz])
                        slice_rescaled = transform.rescale(slice_img,
                                                           scale_vector,
                                                           order=1,
                                                           preserve_range=True,
                                                           multichannel=False,
                                                           mode='constant')

                        x, y = slice_rescaled.shape

                        x_s = (x - nx) // 2
                        y_s = (y - ny) // 2
                        x_c = (nx - x) // 2
                        y_c = (ny - y) // 2

                        # Crop section of image for prediction
                        if x > nx and y > ny:
                            slice_cropped = slice_rescaled[x_s:x_s+nx, y_s:y_s+ny]
                        else:
                            slice_cropped = np.zeros((nx,ny))
                            if x <= nx and y > ny:
                                slice_cropped[x_c:x_c+ x, :] = slice_rescaled[:,y_s:y_s + ny]
                            elif x > nx and y <= ny:
                                slice_cropped[:, y_c:y_c + y] = slice_rescaled[x_s:x_s + nx, :]
                            else:
                                slice_cropped[x_c:x_c+x, y_c:y_c + y] = slice_rescaled[:, :]


                        # GET PREDICTION
                        network_input = np.float32(np.tile(np.reshape(slice_cropped, (nx, ny, 1)), (1, 1, 1, 1)))
                        mask_out, softmax = segmenter_model.predict(network_input)

                        prediction_cropped = np.squeeze(softmax[0,...])

                        # ASSEMBLE BACK THE SLICES
                        slice_predictions = np.zeros((x,y,exp_config.nlabels))
                        # insert cropped region into original image again
                        if x > nx and y > ny:
                            slice_predictions[x_s:x_s+nx, y_s:y_s+ny,:] = prediction_cropped
                        else:
                            if x <= nx and y > ny:
                                slice_predictions[:, y_s:y_s+ny,:] = prediction_cropped[x_c:x_c+ x, :,:]
                            elif x > nx and y <= ny:
                                slice_predictions[x_s:x_s + nx, :,:] = prediction_cropped[:, y_c:y_c + y,:]
                            else:
                                slice_predictions[:, :,:] = prediction_cropped[x_c:x_c+ x, y_c:y_c + y,:]


                        # RESCALING ON THE LOGITS
                        if gt_exists:
                            prediction = transform.resize(slice_predictions,
                                                          (mask.shape[0], mask.shape[1], exp_config.nlabels),
                                                          order=1,
                                                          preserve_range=True,
                                                          mode='constant')
                        else:  # This can occasionally lead to wrong volume size, therefore if gt_exists
                               # we use the gt mask size for resizing.
                            prediction = transform.rescale(slice_predictions,
                                                           (1.0/scale_vector[0], 1.0/scale_vector[1], 1),
                                                           order=1,
                                                           preserve_range=True,
                                                           multichannel=False,
                                                           mode='constant')

                        prediction = np.uint8(np.argmax(prediction, axis=-1))
                        # import matplotlib.pyplot as plt
                        # fig = plt.Figure()
                        # for ii in range(3):
                        #     plt.subplot(1, 3, ii + 1)
                        #     plt.imshow(np.squeeze(prediction))
                        # plt.show()

                        predictions.append(prediction)

                    prediction_arr = np.transpose(np.asarray(predictions, dtype=np.uint8), (1,2,0))

                elif exp_config.dimensionality_mode == '3D':

                    nx, ny, nz = exp_config.image_size

                    pixel_size = (img_header.structarr['pixdim'][1], img_header.structarr['pixdim'][2],
                                  img_header.structarr['pixdim'][3])

                    scale_vector = (pixel_size[0] / exp_config.target_resolution[0],
                                    pixel_size[1] / exp_config.target_resolution[1],
                                    pixel_size[2] / exp_config.target_resolution[2])

                    vol_scaled = transform.rescale(img,
                                                   scale_vector,
                                                   order=1,
                                                   preserve_range=True,
                                                   multichannel=False,
                                                   mode='constant')

                    nz_max = exp_config.image_size[2]
                    slice_vol = np.zeros((nx, ny, nz_max), dtype=np.float32)

                    nz_curr = vol_scaled.shape[2]
                    stack_from = (nz_max - nz_curr) // 2
                    stack_counter = stack_from

                    x, y, z = vol_scaled.shape

                    x_s = (x - nx) // 2
                    y_s = (y - ny) // 2
                    x_c = (nx - x) // 2
                    y_c = (ny - y) // 2

                    for zz in range(nz_curr):

                        slice_rescaled = vol_scaled[:, :, zz]

                        if x > nx and y > ny:
                            slice_cropped = slice_rescaled[x_s:x_s + nx, y_s:y_s + ny]
                        else:
                            slice_cropped = np.zeros((nx, ny))
                            if x <= nx and y > ny:
                                slice_cropped[x_c:x_c + x, :] = slice_rescaled[:, y_s:y_s + ny]
                            elif x > nx and y <= ny:
                                slice_cropped[:, y_c:y_c + y] = slice_rescaled[x_s:x_s + nx, :]

                            else:
                                slice_cropped[x_c:x_c + x, y_c:y_c + y] = slice_rescaled[:, :]

                        slice_vol[:, :, stack_counter] = slice_cropped
                        stack_counter += 1

                    stack_to = stack_counter

                    network_input = np.float32(np.reshape(slice_vol, (1, nx, ny, nz_max, 1)))
                    start_time = time.time()
                    mask_out, softmax = segmenter_model.predict(network_input)
                    logging.info('Classified 3D: %f secs' % (time.time() - start_time))

                    prediction_nzs = mask_out[0, :, :, stack_from:stack_to]  # non-zero-slices

                    if not prediction_nzs.shape[2] == nz_curr:
                        raise ValueError('sizes mismatch')

                    # ASSEMBLE BACK THE SLICES
                    prediction_scaled = np.zeros(vol_scaled.shape)  # last dim is for logits classes

                    # insert cropped region into original image again
                    if x > nx and y > ny:
                        prediction_scaled[x_s:x_s + nx, y_s:y_s + ny, :] = prediction_nzs
                    else:
                        if x <= nx and y > ny:
                            prediction_scaled[:, y_s:y_s + ny, :] = prediction_nzs[x_c:x_c + x, :, :]
                        elif x > nx and y <= ny:
                            prediction_scaled[x_s:x_s + nx, :, :] = prediction_nzs[:, y_c:y_c + y, :]
                        else:
                            prediction_scaled[:, :, :] = prediction_nzs[x_c:x_c + x, y_c:y_c + y, :]

                    logging.info('Prediction_scaled mean %f' % (np.mean(prediction_scaled)))

                    prediction = transform.resize(prediction_scaled,
                                                  (mask.shape[0], mask.shape[1], mask.shape[2], 1),
                                                  order=1,
                                                  preserve_range=True,
                                                  mode='constant')
                    prediction = np.argmax(prediction, axis=-1)
                    prediction_arr = np.asarray(prediction, dtype=np.uint8)


                # This is the same for 2D and 3D again
                if do_postprocessing:
                    prediction_arr = utils.keep_largest_connected_components(prediction_arr)

                elapsed_time = time.time() - start_time
                total_time += elapsed_time
                total_volumes += 1

                logging.info('Evaluation of volume took %f secs.' % elapsed_time)

                if frame == ED_frame:
                    frame_suffix = '_ED'
                elif frame == ES_frame:
                    frame_suffix = '_ES'
                else:
                    raise ValueError('Frame doesnt correspond to ED or ES. frame = %d, ED = %d, ES = %d' %
                                     (frame, ED_frame, ES_frame))

                # Save prediced mask
                out_file_name = os.path.join(output_folder, 'prediction',
                                             'patient' + patient_id + frame_suffix + '.nii.gz')
                if gt_exists:
                    out_affine = mask_affine
                    out_header = mask_header
                else:
                    out_affine = img_affine
                    out_header = img_header

                logging.info('saving to: %s' % out_file_name)
                utils.save_nii(out_file_name, prediction_arr, out_affine, out_header)

                # Save image data to the same folder for convenience
                image_file_name = os.path.join(output_folder, 'image',
                                        'patient' + patient_id + frame_suffix + '.nii.gz')
                logging.info('saving to: %s' % image_file_name)
                utils.save_nii(image_file_name, img, out_affine, out_header)

                if gt_exists:

                    # Save GT image
                    gt_file_name = os.path.join(output_folder, 'ground_truth', 'patient' + patient_id + frame_suffix + '.nii.gz')
                    logging.info('saving to: %s' % gt_file_name)
                    utils.save_nii(gt_file_name, mask, out_affine, out_header)

                    # Save difference mask between predictions and ground truth
                    difference_mask = np.where(np.abs(prediction_arr-mask) > 0, [1], [0])
                    difference_mask = np.asarray(difference_mask, dtype=np.uint8)
                    diff_file_name = os.path.join(output_folder,
                                                  'difference',
                                                  'patient' + patient_id + frame_suffix + '.nii.gz')
                    logging.info('saving to: %s' % diff_file_name)
                    utils.save_nii(diff_file_name, difference_mask, out_affine, out_header)

                # calculate metrics
                y_ = prediction_arr
                y = mask

                per_lbl_dice = []
                per_lbl_assd = []
                per_lbl_hd = []

                for lbl in range(exp_config.nlabels):

                    binary_pred = (y_ == lbl) * 1
                    binary_gt = (y == lbl) * 1

                    if np.sum(binary_gt) == 0 and np.sum(binary_pred) == 0:
                        per_lbl_dice.append(1)
                        per_lbl_assd.append(0)
                        per_lbl_hd.append(0)
                    elif np.sum(binary_pred) > 0 and np.sum(binary_gt) == 0 or np.sum(binary_pred) == 0 and np.sum(binary_gt) > 0:
                        logging.warning(
                            'Structure missing in either GT (x)or prediction. ASSD and HD will not be accurate.')
                        per_lbl_dice.append(0)
                        per_lbl_assd.append(1)
                        per_lbl_hd.append(1)
                    else:
                        per_lbl_dice.append(dc(binary_pred, binary_gt))
                        per_lbl_assd.append(assd(binary_pred, binary_gt, voxelspacing=zooms))
                        per_lbl_hd.append(hd(binary_pred, binary_gt, voxelspacing=zooms))

                dice_list.append(per_lbl_dice)
                assd_list.append(per_lbl_assd)
                hd_list.append(per_lbl_hd)


    logging.info('Average time per volume: %f' % (total_time/total_volumes))

    dice_arr = np.asarray(dice_list)
    assd_arr = np.asarray(assd_list)
    hd_arr = np.asarray(hd_list)

    mean_per_lbl_dice = dice_arr.mean(axis=0)
    mean_per_lbl_assd = assd_arr.mean(axis=0)
    mean_per_lbl_hd = hd_arr.mean(axis=0)

    logging.info('Dice')
    logging.info(mean_per_lbl_dice)
    logging.info(np.mean(mean_per_lbl_dice))
    logging.info('ASSD')
    logging.info(mean_per_lbl_assd)
    logging.info(np.mean(mean_per_lbl_assd))
    logging.info('HD')
    logging.info(mean_per_lbl_hd)
    logging.info(np.mean(mean_per_lbl_hd))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script for a simple test loop evaluating a network on the test dataset")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    args = parser.parse_args()

    gt_exists = True  # Change this to false if you are trying to evaluate the test volumes on the ACDC server

    base_path = sys_config.project_root

    model_path = os.path.join(base_path, args.EXP_PATH)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    input_path = exp_config.data_root
    output_path = os.path.join(model_path, 'predictions')
    path_pred = os.path.join(output_path, 'prediction')
    path_image = os.path.join(output_path, 'image')
    utils.makefolder(path_pred)
    utils.makefolder(path_image)

    if gt_exists:
        path_gt = os.path.join(output_path, 'ground_truth')
        path_diff = os.path.join(output_path, 'difference')
        path_eval = os.path.join(output_path, 'eval')

        utils.makefolder(path_diff)
        utils.makefolder(path_gt)

    main(input_path,
         output_path,
         model_path,
         exp_config=exp_config,
         do_postprocessing=True,
         gt_exists=gt_exists)






