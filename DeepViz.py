#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Tran Minh Quan, quantm@unist.ac.kr
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os, sys, argparse, glob, cv2, six



# Misc. libraries
from six.moves import map, zip, range
from natsort import natsorted 

# Array and image processing toolboxes
import numpy as np 
import skimage
import skimage.io
import skimage.color
import skimage.transform

import tensorflow as tf
###################################################################################################
from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.utils import get_rng
from tensorpack.tfutils import optimizer, gradproc
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils import logger

###################################################################################################
class Model(ModelDesc):
	def _get_inputs(self):
		pass

	def _build_graph(self, inputs):
		sImg2d # sImg the projection 2D, reshape from 
		VGG_MEAN = np.array([123.68, 116.779, 103.939])  # RGB
		VGG_MEAN_TENSOR = tf.constant(VGG_MEAN, dtype=tf.float32)

		with tf.variable_scope('VGG19'):

			sImg2d = sImg2d - VGG_MEAN_TENSOR

			# VGG 19
			with varreplace.freeze_variables():
				with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu):
					conv1_1 = Conv2D('conv1_1', sImgs, 64)
					conv1_2 = Conv2D('conv1_2', conv1_1, 64)
					pool1 = MaxPooling('pool1', conv1_2, 2)  # 64
					conv2_1 = Conv2D('conv2_1', pool1, 128)
					conv2_2 = Conv2D('conv2_2', conv2_1, 128)
					pool2 = MaxPooling('pool2', conv2_2, 2)  # 32
					conv3_1 = Conv2D('conv3_1', pool2, 256)
					conv3_2 = Conv2D('conv3_2', conv3_1, 256)
					conv3_3 = Conv2D('conv3_3', conv3_2, 256)
					conv3_4 = Conv2D('conv3_4', conv3_3, 256)
					pool3 = MaxPooling('pool3', conv3_4, 2)  # 16
					conv4_1 = Conv2D('conv4_1', pool3, 512)
					conv4_2 = Conv2D('conv4_2', conv4_1, 512)
					conv4_3 = Conv2D('conv4_3', conv4_2, 512)
					conv4_4 = Conv2D('conv4_4', conv4_3, 512)
					pool4 = MaxPooling('pool4', conv4_4, 2)  # 8
					conv5_1 = Conv2D('conv5_1', pool4, 512)
					conv5_2 = Conv2D('conv5_2', conv5_1, 512)
					conv5_3 = Conv2D('conv5_3', conv5_2, 512)
					conv5_4 = Conv2D('conv5_4', conv5_3, 512)
					pool5 = MaxPooling('pool5', conv5_4, 2)  # 4
					feats = pool5 # Take the feature map
		with tf.variable_scope('Decoder'):
			# with varreplace.freeze_variables():
				with argscope([Conv2D, Deconv2D], kernel_shape=3, nl=tf.nn.relu):
					
					conv5_4 = Conv2D('conv5_4', feats,   512)
					conv5_3 = Conv2D('conv5_3', conv5_2, 512)
					conv5_2 = Conv2D('conv5_2', conv5_1, 512)
					conv5_1 = Conv2D('conv5_1', pool4, 512)
					pool4 = Deconv2D('pool4', conv4_4, 2)  # 8
					conv4_4 = Conv2D('conv4_4', conv4_3, 512)
					conv4_3 = Conv2D('conv4_3', conv4_2, 512)
					conv4_2 = Conv2D('conv4_2', conv4_1, 512)
					conv4_1 = Conv2D('conv4_1', pool3, 512)
					pool3 = Deconv2D('pool3', conv3_4, 2)  # 16
					conv3_4 = Conv2D('conv3_4', conv3_3, 256)
					conv3_3 = Conv2D('conv3_3', conv3_2, 256)
					conv3_2 = Conv2D('conv3_2', conv3_1, 256)
					conv3_1 = Conv2D('conv3_1', pool2, 256)
					pool2 = Deconv2D('pool2', conv2_2, 2)  # 32
					conv2_2 = Conv2D('conv2_2', conv2_1, 128)
					conv2_1 = Conv2D('conv2_1', pool1, 128)
					pool1 = Deconv2D('pool1', conv1_2, 2)  # 64
					conv1_2 = Conv2D('conv1_2', conv1_1, 64)
					conv1_1 = Conv2D('conv1_1', conv1_2, 64)

					dImg2d =  Conv2D('conv0_1', conv1_1, 3) # destination

		# Build loss in here
	def _get_optimizer(self):
		lr  = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
		opt = tf.train.AdamOptimizer(lr)
		return opt

###################################################################################################
def render(model_path, volume_path, style_path):
	pass

###################################################################################################
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', 		help='comma separated list of GPU(s) to use.')
	parser.add_argument('--load', 		help='load model')
	parser.add_argument('--vgg19', 		help='load model', default='data/vgg19.npz')
	parser.add_argument('--image_path', help='path to the image.', default='data/train/image/') 
	parser.add_argument('--style_path', help='path to the style.', default='data/train/style/') 
	parser.add_argument('--render', 	action='store_true')
	
	args = parser.parse_args()

	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	if args.render:
		render(args.load, args.volume, args.style)
	else:
		# Set the logger directory
		logger.auto_set_dir()

		nr_tower = max(get_nr_gpu(), 1)
		train_ds = QueueInput(get_data(args.data))
		model = Model()

		if args.load:
            session_init = SaverRestore(args.load)
        else: # For training from scratch, read the vgg model from args.vgg19
            assert os.path.isfile(args.vgg19)
            param_dict = dict(np.load(args.vgg19))
            param_dict = {'VGG19/' + name: value for name, value in six.iteritems(param_dict)}
            session_init = DictRestore(param_dict)

		
		# Set up configuration
		config = TrainConfig(
			model           =   model, 
			dataflow        =   train_ds,
			callbacks       =   [
				PeriodicTrigger(ModelSaver(), every_k_epochs=50),
				# PeriodicTrigger(VisualizeRunner(valid_ds), every_k_epochs=5),
				ScheduledHyperParamSetter('learning_rate', [(0, 2e-4), (100, 1e-4), (200, 1e-5), (300, 1e-6)], interp='linear')
				#ScheduledHyperParamSetter('learning_rate', [(30, 6e-6), (45, 1e-6), (60, 8e-7)]),
            	#HumanHyperParamSetter('learning_rate'),
				],
			max_epoch       =   500, 
			session_init    =   session_init,
			nr_tower        =   max(get_nr_gpu(), 1)
			)
	
		# Train the model
		SyncMultiGPUTrainer(config).train()
