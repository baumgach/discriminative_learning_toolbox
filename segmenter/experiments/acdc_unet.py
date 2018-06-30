from segmenter.network_zoo import nets2D
import tensorflow as tf
import os
import config.system as sys_config

experiment_name = 'acdc_unet_crfrnn_xent_lr0.001_bs10'

# Model settings
network = nets2D.unet2D_crfrnn

# Data settings
data_identifier = 'acdc'
preproc_folder = os.path.join(sys_config.project_root, 'data/preproc_data/acdc')
data_root = '/scratch_net/bmicdl03/data/ACDC_challenge_20170617'
image_size = (256, 256)
target_resolution = (1.36719, 1.36719)
nlabels = 4

# Cost function
weight_decay = 0.0
loss_type = 'crossentropy'  # 'dice_micro'/'dice_macro'/'dice_macro_robust'/'crossentropy'

# Training settings
batch_size = 10
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
