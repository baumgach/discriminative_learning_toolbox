from segmenter.network_zoo import nets3D
import tensorflow as tf
import os
import config.system as sys_config

experiment_name = 'acdc_unet3D_depth4_n0_32'

# Model settings
network = nets3D.unet3D

# Data settings
data_identifier = 'acdc'
preproc_folder = os.path.join(sys_config.project_root, 'data/preproc_data/acdc')
data_root = '/scratch_net/bmicdl03/data/ACDC_challenge_20170617'
dimensionality_mode = '3D'
image_size = (112, 112, 24)
target_resolution = (2.5, 2.5, 5.0)
nlabels = 4
tensorboard_slice = 10

# Network settings
n0 = 32

# Cost function
weight_decay = 0.0
loss_type = 'crossentropy'  # 'dice_micro'/'dice_macro'/'dice_macro_robust'/'crossentropy'

# Training settings
batch_size = 2
n_accum_grads = 4
learning_rate = 1e-2
optimizer_handle = tf.train.AdamOptimizer
beta1=0.9
beta2=0.999
schedule_lr = False
divide_lr_frequency = None
warmup_training = False
momentum = None

# Augmentation
do_augmentations = False
augmentation_options = { }

# Rarely changed settings
use_data_fraction = False  # Should normally be False
max_iterations = 1000000
train_eval_frequency = 50
val_eval_frequency = 50
