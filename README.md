# Discriminative Learning Toolbox

A collection of tensorflow code for training segmentation and classification networks. 



This repository contains a collection of code to train and evaluate state-of-the-art 
segmentation and classification networks. 

Specifically, it contains code for:

- ADNI Alzheimer's classification
- ACDC cardiac challenge segmentation
- NCI ISBI prostate challenge segmentation
- Segmentation and classification of synthetic toy data

It includes wrappers and implementations for many state-of-the-art deep learning techniques
such as:
 - U-Net (2D and 3D)
 - Resnet34
 - Identity Resnet34
 - Aggregating gradient updates over multiple batches
 - CRF-RNN
 - Instance normalisation
 - Layer normalisation
 - Group normalisation
 - Saliency maps (guided backprop, class activation mappings (CAM) and more)
 - etc. 

Author:
 - Christian F. Baumgartner ([email](mailto:baumgartner@vision.ee.ethz.ch))

Contributions by:
 - Ender Konukoglu (code for synthetic data generation)
 - Krishna Chaitanya (code for reading NCI prostate data)
 - Yigit Baran Can (code for CRF-RNN layer)
 - Lisa Koch (code for ACDC evaluation)
 - Firat Ozdemir (code for summing over batches in Dice calculation)

## Requirements 

- Python 3.4 (only tested with 3.4.3)
- Tensorflow (tested with 1.2.0 and 1.8.0)
- The remainder of the requirements are given in `requirements.txt`

## Getting the code

Clone the repository by typing

``` git clone https://github.com/baumgach/discriminative_learning_toolkit.git ```


## Installing required Python packages

Create an environment with Python 3.4. If you use virutalenv it 
might be necessary to first upgrade pip (``` pip install --upgrade pip ```).

Next, install the required packages listed in the `requirements.txt` file:

``` pip install -r requirements.txt ```

Then, install tensorflow:

``` pip install tensorflow==1.8 ```
or
``` pip install tensorflow-gpu==1.8 ```


## Running the code locally

Open the `config/system.py` and edit all the paths there to match your system.

Next, open `classifier_train.py` or `segmenter_train.py` and, at the top of the file, select the experiment you want to run (or simply use the default).

Make sure the data path in your chosen experiment config file points to a location accesible
from your workstation. 

To train a classification model run:

``` python classifier_train.py ```

or to train a segmentation model run:

``` python segmenter_train.py ```

WARNING: When you run the code on CPU, you need around 12 GB of RAM. Make sure your system is up to the task. If not you can try reducing the batch size, or simplifying the network. 

In `system.py`, a log directory was defined. By default it is called `logdir`. You can start a tensorboard
session in order to monitor the training of the network(s) by typing the following in a shell with your virtualenv
activated

``` tensorboard --logdir=logdir --port 8008 ```

Then, navigate in your web browser to [localhost:8008](localhost:8008) in your browser to open tensorboard.
You can also set the logdir to `logdir/segmenter` if you only want to show the 
segmentation logs.

At any point during the training, or after, you can evaluate your model by typing the following:

``` python segmenter_test_predictions.py logdir/segmenter/acdc_unet_xent ```

where you have to adapt the line to match your experiment. Note that, the path must be given relative to your
working directory. Giving the full path will not work.


## Running the code on the ETH CVL (Biwi) GPU infrastructure:

Instructions for setting everything up to run this code on the Biwi GPU 
infrastructure can be found [here](https://git.ee.ethz.ch/baumgach/biwi_tensorflow_setup_instructions).

Don't forget to change the `at_biwi` option in `config/system.py`! 

## Known issues

 - None yet