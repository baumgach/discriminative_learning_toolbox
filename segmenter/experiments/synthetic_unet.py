from segmenter.network_zoo import nets2D
import tensorflow as tf
import os
import config.system as sys_config

experiment_name = 'synth_unet_xent_rererun'

# Model settings
network = nets2D.unet2D

# Data settings
data_identifier = 'synthetic'
preproc_folder = os.path.join(sys_config.project_root, 'data/preproc_data/synthetic')
image_size = [112, 112]
effect_size = 100
num_samples = 10000
moving_effect = True
rescale_to_one = True
nlabels = 2
label_type = 'mask'

# Network settings
n0 = 64

# Cost function
weight_decay = 0.0
loss_type = 'crossentropy'  # 'dice_micro'/'dice_macro'/'dice_macro_robust'/'crossentropy'

# Training settings
batch_size = 30
n_accum_grads = 1
learning_rate = 1e-3
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
train_eval_frequency = 50
val_eval_frequency = 50
