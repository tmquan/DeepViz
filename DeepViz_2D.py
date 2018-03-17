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

skimage.io.use_plugin('matplotlib')
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

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
np.warnings.filterwarnings('ignore')

DIMX  = 256
DIMY  = 256
SIZE  = 256 # For resize

EPOCH_SIZE = 2000
BATCH_SIZE = 20
###################################################################################################
class Model(ModelDesc):
	def _get_inputs(self):
		return [
			InputDesc(tf.float32, (BATCH_SIZE, DIMY, DIMX, 3), 'rgbImg'),
			]

	def _build_graph(self, inputs):
		# sImg2d # sImg the projection 2D, reshape from 
		VGG_MEAN = np.array([123.68, 116.779, 103.939])  # RGB
		VGG_MEAN_TENSOR = tf.constant(VGG_MEAN, dtype=tf.float32)

		rgbImg = inputs # Split the input

		rgbImg = tf.reshape(rgbImg, [BATCH_SIZE, DIMY, DIMX, 3])
		sImg2d = rgbImg - VGG_MEAN_TENSOR
		

		with tf.variable_scope('VGG19'):
			# VGG 19
			with varreplace.freeze_variables():
				with argscope(Conv2D, kernel_shape=3, nl=tf.nn.relu):
					conv1_1 = Conv2D('conv1_1', sImg2d,  64)
					conv1_2 = Conv2D('conv1_2', conv1_1, 64)
					pool1 = MaxPooling('pool1', conv1_2, 2)  # 64
					conv2_1 = Conv2D('conv2_1', pool1,   128)
					conv2_2 = Conv2D('conv2_2', conv2_1, 128)
					pool2 = MaxPooling('pool2', conv2_2, 2)  # 32
					conv3_1 = Conv2D('conv3_1', pool2,   256)
					conv3_2 = Conv2D('conv3_2', conv3_1, 256)
					conv3_3 = Conv2D('conv3_3', conv3_2, 256)
					conv3_4 = Conv2D('conv3_4', conv3_3, 256)
					pool3 = MaxPooling('pool3', conv3_4, 2)  # 16
					conv4_1 = Conv2D('conv4_1', pool3,   512)
					conv4_2 = Conv2D('conv4_2', conv4_1, 512)
					conv4_3 = Conv2D('conv4_3', conv4_2, 512)
					conv4_4 = Conv2D('conv4_4', conv4_3, 512)
					pool4 = MaxPooling('pool4', conv4_4, 2)  # 8
					conv5_1 = Conv2D('conv5_1', pool4,   512)
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
					conv5_1 = Conv2D('conv5_1', pool4,   512)
					pool4 = Deconv2D('pool4', conv4_4,   2)  # 8
					conv4_4 = Conv2D('conv4_4', conv4_3, 512)
					conv4_3 = Conv2D('conv4_3', conv4_2, 512)
					conv4_2 = Conv2D('conv4_2', conv4_1, 512)
					conv4_1 = Conv2D('conv4_1', pool3,   512)
					pool3 = Deconv2D('pool3', conv3_4,   2)  # 16
					conv3_4 = Conv2D('conv3_4', conv3_3, 256)
					conv3_3 = Conv2D('conv3_3', conv3_2, 256)
					conv3_2 = Conv2D('conv3_2', conv3_1, 256)
					conv3_1 = Conv2D('conv3_1', pool2,   256)
					pool2 = Deconv2D('pool2', conv2_2,   2)  # 32
					conv2_2 = Conv2D('conv2_2', conv2_1, 128)
					conv2_1 = Conv2D('conv2_1', pool1,   28)
					pool1 = Deconv2D('pool1', conv1_2,   2)  # 64
					conv1_2 = Conv2D('conv1_2', conv1_1, 64)
					conv1_1 = Conv2D('conv1_1', conv1_2, 64)

					dImg2d =  Conv2D('conv0_1', conv1_1, 3) # destination

		# Reconstruct img
		recImg = dImg2d + VGG_MEAN_TENSOR

		# Build loss in here
		losses = []
		with tf.name_scope('loss_abs'):
			abs_img2d = tf.reduce_mean(tf.abs(sImg2d - dImg2d), name='abs_img2d')
			losses.append(abs_img2d)
			add_moving_summary(abs_img2d)	

		# Aggregate the loss
		self.cost = tf.reduce_sum(losses, name='self.cost')
		add_moving_summary(self.cost)	

		# Visualization
		viz = tf.concat([rgbImg, recImg], axis=2)
		viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
		tf.summary.image('colorized', viz, max_outputs=50)

	def _get_optimizer(self):
		lr  = tf.get_variable('learning_rate', initializer=1e-4, trainable=False)
		opt = tf.train.AdamOptimizer(lr)
		return opt
####################################################################################################
class ImageDataFlow(RNGDataFlow):
	def __init__(self, image_path, style_path, size, dtype='float32', isTrain=False, isValid=False):
		self.dtype      	= dtype
		self.image_path   	= image_path
		self.style_path   	= style_path
		self._size      	= size
		self.isTrain    	= isTrain
		self.isValid    	= isValid

	def size(self):
		return self._size

	def reset_state(self):
		self.rng = get_rng(self)

	def get_data(self, shuffle=True):
		#
		# Read and store into pairs of images and labels
		#
		images = glob.glob(self.image_path + '/*.*')
		styles = glob.glob(self.style_path + '/*.*')

		if self._size==None:
			self._size = len(images)

		from natsort import natsorted
		images = natsorted(images)
		styles = natsorted(styles)

		# print(images)
		# print(styles)

		#
		# Pick the image over size 
		#
		for k in range(self._size):
			#
			# Pick randomly a tuple of training instance
			#
			rand_image = np.random.randint(0, len(images))
			rand_style = np.random.randint(0, len(styles))

			if self.isTrain:
				# image = skimage.io.imread(images[rand_image])		
				style = cv2.imread(styles[rand_style], cv2.IMREAD_COLOR)
				
				style = self.central_crop(style)
				style = self.resize_image(style, size=512)
				style = self.random_crop (style, size=256)
				# print(style.shape)
				# logger.auto_set_dir()
				# logger.info('Style image {} with shape {}'.format(styles[rand_style], style.shape))

			else:
				pass

			yield [
				   style.astype(np.float32), 
				   ]

	def central_crop(self, image):
		shape = image.shape
		dimy, dimx = shape[0], shape[1]
		shorter = min(dimy, dimx)
		y_pad, x_pad = (dimy - shorter) // 2, (dimx - shorter) // 2
		image = image[y_pad:y_pad+shorter, 
					  x_pad:x_pad+shorter]
		return image

	def resize_image(self, image, size=SIZE): # Scale to 512
		shape = image.shape
		dimy, dimx = shape[0], shape[1]
		from scipy.misc import imresize
		if dimy > dimx:
			image = imresize(image, (dimy*size//dimx, size), interp='bilinear')
		else:
			image = imresize(image, (size, dimx*size//dimy), interp='bilinear')
		return image

	def random_crop(image, crop_size):
		shape = image.shape
		dimy, dimx = shape[0], shape[1]
		assert crop_size<dimx and crop_size<dimy
		randy = self.rng.randint(0, dimy-DIMY+1)
		randx = self.rng.randint(0, dimx-DIMX+1)
		image = image[randy:randy+DIMY,randx:randx+DIMX,...]
		return image
	def random_flip(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			np.random.seed(seed)
		random_flip = np.random.randint(1,5)
		if random_flip==1:
			flipped = image[...,::1,::-1,:]
			image = flipped
		elif random_flip==2:
			flipped = image[...,::-1,::1,:]
			image = flipped
		elif random_flip==3:
			flipped = image[...,::-1,::-1,:]
			image = flipped
		elif random_flip==4:
			flipped = image
			image = flipped
		return image

	
				
	
####################################################################################################
def get_data(image_path, style_path, size=EPOCH_SIZE):
	ds_train = ImageDataFlow(image_path=image_path,
							 style_path=style_path, 
							 size=size, 
							 isTrain=True
							 )

	ds_valid = ImageDataFlow(image_path=image_path,
							 style_path=style_path, 
							 size=size, 
							 isValid=True
							 )

	ds_train.reset_state()
	ds_valid.reset_state() 

	ds_train = BatchData(ds_train, BATCH_SIZE)
	ds_valid = BatchData(ds_valid, BATCH_SIZE)

	ds_train = PrefetchDataZMQ(ds_train, 2)
	return ds_train, ds_valid

###################################################################################################
def render(model_path, volume_path, style_path):
	pass

###################################################################################################
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', 		help='comma separated list of GPU(s) to use.')
	parser.add_argument('--load', 		help='load model')
	parser.add_argument('--vgg19', 		help='load model', 		   default='data/model/vgg19.npz')
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

		ds_train, ds_valid = get_data(args.image_path, args.style_path)

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
			dataflow        =   ds_train,
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
