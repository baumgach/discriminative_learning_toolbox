from classifier.network_zoo import nets2D
import tensorflow as tf
import os
import config.system as sys_config

experiment_name = 'chestx_vgg16'

# Model settings
classifier_net = nets2D.vgg16

# Data settings
# Label dict:
# 1: Atelectasis, 2: Cardiomegaly, 3: Effusion, 4: Infiltration, 5: Mass, 6: Nodule, 7: Pneumonia,
# 8: Pneumothorax, 9: Consolidation, 10: Edema, 11: Emphysema, 12: Fibrosis,
# 13: Pleural_Thickening, 14: Hernia
data_identifier = 'chestX'
data_root = '/scratch-second/data/ChestX-ray8'
preproc_folder = os.path.join(sys_config.project_root, 'data/preproc_data/chestxdata')
image_size = [224, 224]
n_input_channels = 1
rescale_to_one = True
crop_black=True
nlabels = 15

# Cost function
weight_decay = 0.0

# Training settings
batch_size = 32  # if list use specific class ratio
n_accum_grads = 1
learning_rate = 1e-4
optimizer_handle = tf.train.AdamOptimizer
beta1=0.9
beta2=0.999
schedule_lr = False
divide_lr_frequency = None
warmup_training = False
momentum = None

# Rarely changed settings
use_data_fraction = False  # Should normally be False
max_iterations = 1000000
train_eval_frequency = 100
val_eval_frequency = 100
