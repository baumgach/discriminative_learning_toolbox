# Get classification metrics for a trained classifier model
# Authors:
# Christian F. Baumgartner (c.f.baumgartner@gmail.com)


import numpy as np
import os
import glob
from importlib.machinery import SourceFileLoader
import argparse
from sklearn.metrics import f1_score, classification_report, confusion_matrix

import config.system as sys_config
from classifier.model_classifier import classifier

import logging
from data.data_switch import data_switch
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


def main(model_path, exp_config):

    # Get Data
    data_loader = data_switch(exp_config.data_identifier)
    data = data_loader(exp_config)

    # Make and restore vagan model
    classifier_model = classifier(exp_config=exp_config, data=data)
    classifier_model.load_weights(model_path, type='latest')

    # Run predictions in an endless loop
    pred_list = []
    gt_list = []

    for batch in data.test.iterate_batches(32):

        x, y = batch

        y_ = classifier_model.predict(x)[0]

        pred_list += list(y_)
        gt_list += list(y)


    # print(pred_list)
    # print(gt_list)

    logging.info(classification_report(np.asarray(gt_list), np.asarray(pred_list)))
    logging.info(confusion_matrix(np.asarray(gt_list), np.asarray(pred_list)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(
        description="Script for a simple test loop evaluating a network on the test dataset")
    parser.add_argument("EXP_PATH", type=str, help="Path to experiment folder (assuming you are in the working directory)")
    args = parser.parse_args()

    base_path = sys_config.project_root

    model_path = os.path.join(base_path, args.EXP_PATH)
    config_file = glob.glob(model_path + '/*py')[0]
    config_module = config_file.split('/')[-1].rstrip('.py')

    exp_config = SourceFileLoader(config_module, os.path.join(config_file)).load_module()

    main(model_path, exp_config=exp_config)
