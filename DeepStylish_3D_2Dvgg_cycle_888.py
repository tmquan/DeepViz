#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Tran Minh Quan, quantm@unist.ac.kr
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os, sys, argparse, glob, cv2, six, h5py, time, shutil



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

import tflearn
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.conv import conv_3d, conv_3d_transpose, max_pool_3d 
from tflearn.layers.core import dropout
from tflearn.layers.merge_ops import merge
from tflearn.activations import linear, sigmoid, tanh, elu 
from tensorflow.python.framework import ops


# Tensorflow 
import tensorflow as tf
from tensorflow import layers
# from tensorflow.contrib.keras.python.keras import backend as K
# from tensorflow.keras import backend as K
from tensorflow.python.keras import backend as K

###################################################################################################
from tensorpack import *
from tensorpack.dataflow import dataset
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.utils import get_rng
from tensorpack.tfutils import optimizer, gradproc
from tensorpack.tfutils.symbolic_functions import flatten, batch_flatten
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary, add_tensor_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils import logger

from tensorpack import *
from tensorpack.utils.viz import *
from tensorpack.utils.gpu import get_nr_gpu
from tensorpack.utils.utils import get_rng
from tensorpack.utils.argtools import memoized
from tensorpack import (TowerTrainer,
						ModelDescBase, DataFlow, StagingInput)
from tensorpack.tfutils.tower import TowerContext, TowerFuncWrapper
from tensorpack.graph_builder import DataParallelBuilder, LeastLoadedDeviceSetter

###################################################################################################
import tensorflow as tf
from tensorpack.models.common import layer_register, VariableHolder
from tensorpack.tfutils.common import get_tf_version_number
from tensorpack.utils.argtools import shape2d, shape4d, get_data_format
from tensorpack.models.tflayer import rename_get_variable, convert_to_tflayer_args
###################################################################################################
np.warnings.filterwarnings('ignore')

DIMX  = 256
DIMY  = 256
DIMZ  = 256
SIZE  = 256 # For resize

EPOCH_SIZE = 400
BATCH_SIZE = 1
NB_FILTERS = 32

VGG19_MEAN = np.array([123.68, 116.779, 103.939])  # RGB
VGG19_MEAN_TENSOR = tf.constant(VGG19_MEAN, dtype=tf.float32)
###############################################################################
# Utility function for scaling 
def tf_2tanh(x, maxVal=255.0, name='ToRangeTanh'):
	with tf.variable_scope(name):
		return (x / maxVal - 0.5) * 2.0
###############################################################################
def tf_2imag(x, maxVal=255.0, name='ToRangeImag'):
	with tf.variable_scope(name):
		return (x / 2.0 + 0.5) * maxVal

# Utility function for scaling 
def np_2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
	return (x / maxVal - 0.5) * 2.0
###############################################################################
def np_2imag(x, maxVal = 255.0, name='ToRangeImag'):
	return (x / 2.0 + 0.5) * maxVal

def normalize(v):
	assert isinstance(v, tf.Tensor)
	v.get_shape().assert_has_rank(4)
	return v / tf.reduce_mean(v, axis=[1, 2, 3], keepdims=True)
###################################################################################################
def INReLU(x, name=None):
	x = InstanceNorm('inorm', x)
	return tf.nn.relu(x, name=name)


def INLReLU(x, name=None):
	x = InstanceNorm('inorm', x)
	return tf.nn.leaky_relu(x, name=name)
	
def BNLReLU(x, name=None):
	x = BatchNorm('bn', x)
	return tf.nn.leaky_relu(x, name=name)
###############################################################################
@layer_register(log_shape=True)
def Subpix2D(inputs, chan, scale=2, stride=1, nl=tf.nn.leaky_relu):
	with argscope([Conv2D], nl=nl, stride=stride, kernel_shape=3):
		results = Conv2D('conv0', inputs, chan* scale**2, padding='SAME')
		if scale>1:
			results = tf.depth_to_space(results, scale, name='depth2space', data_format='NHWC')
		return results

###############################################################################
###############################################################################
class ChainModelDesc(ModelDescBase):
	def collect_variables(self, scope_1='scope_1', scope_2='scope_2'):
		"""
		Assign self.scope_1_vars to the parameters under scope `g_scope`,
		and same with self.d_vars.
		"""
		self.scope_1_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_1)
		assert self.scope_1_vars
		self.scope_2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_2)
		assert self.scope_2_vars

	def build_losses(self, scope_1_loss, scope_2_loss):
		"""D and G play two-player minimax game with value function V(G,D)
		  min_G max _D V(D, G) = IE_{x ~ p_data} [log D(x)] + IE_{z ~ p_fake} [log (1 - D(G(z)))]
		Args:
			logits_real (tf.Tensor): discrim logits from real samples
			logits_fake (tf.Tensor): discrim logits from fake samples produced by generator
		"""
		with tf.name_scope("losses"):

			with tf.name_scope("scope_1"):
				self.scope_1_loss = scope_1_loss

			with tf.name_scope("scope_2"):
				self.scope_2_loss = scope_2_loss

			add_moving_summary(self.scope_1_loss, self.scope_2_loss)

	@memoized
	def get_optimizer(self):
		return self._get_optimizer()


class ChainTrainer(TowerTrainer):
	def __init__(self, input, model):
		super(ChainTrainer, self).__init__()
		assert isinstance(model, ChainModelDesc), model
		inputs_desc = model.get_inputs_desc()
		cbs = input.setup(inputs_desc)

		# we need to set towerfunc because it's a TowerTrainer,
		# and only TowerTrainer supports automatic graph creation for inference during training.
		tower_func = TowerFuncWrapper(model.build_graph, inputs_desc)
		with TowerContext('', is_training=True):
			tower_func(*input.get_input_tensors())
		opt = model.get_optimizer()

		# by default, run one scope_2_min after one scope_1_min
		with tf.name_scope('optimize'):
			scope_1_min = opt.minimize(model.scope_1_loss, var_list=model.scope_1_vars, name='g_op')
			with tf.control_dependencies([scope_1_min]):
				scope_2_min = opt.minimize(model.scope_2_loss, var_list=model.scope_2_vars, name='d_op')
		self.train_op = scope_2_min
		self.set_tower_func(tower_func)

		for cb in cbs:
			self.register_callback(cb)

###############################################################################
@layer_register(log_shape=True)
@convert_to_tflayer_args(
	args_names=['filters', 'kernel_size'],
	name_mapping={
		'out_channel': 'filters',
		'kernel_shape': 'kernel_size',
		'stride': 'strides',
	})
def Conv3D(
		inputs,
		filters,
		kernel_size,
		strides=(1, 1, 1),
		padding='same',
		data_format='channels_last',
		dilation_rate=(1, 1, 1),
		activation=None,
		use_bias=False,
		kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0), #tf.contrib.layers.xavier_initializer(), #tf.initializers.variance_scaling(distribution='uniform'),
		bias_initializer=tf.zeros_initializer(),
		kernel_regularizer=None,
		bias_regularizer=None,
		activity_regularizer=None,
		split=1):
	"""
	A wrapper around `tf.layers.Conv2D`.
	Some differences to maintain backward-compatibility:
	1. Default kernel initializer is variance_scaling_initializer(2.0),.
	2. Default padding is 'same'.
	3. Support 'split' argument to do group conv.
	Variable Names:
	* ``W``: weights
	* ``b``: bias
	"""
	if split == 1:
		with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
			layer = tf.layers.Conv3D(
				filters,
				kernel_size,
				strides=strides,
				padding=padding,
				data_format=data_format,
				dilation_rate=dilation_rate,
				activation=activation,
				use_bias=use_bias,
				kernel_initializer=kernel_initializer,
				bias_initializer=bias_initializer,
				kernel_regularizer=kernel_regularizer,
				bias_regularizer=bias_regularizer,
				activity_regularizer=activity_regularizer)
			ret = layer.apply(inputs, scope=tf.get_variable_scope())
			ret = tf.identity(ret, name='output')

		ret.variables = VariableHolder(W=layer.kernel)
		if use_bias:
			ret.variables.b = layer.bias

	else:
		# group conv implementation
		data_format = get_data_format(data_format, tfmode=False)
		in_shape = inputs.get_shape().as_list()
		channel_axis = 3 if data_format == 'NHWC' else 1
		in_channel = in_shape[channel_axis]
		assert in_channel is not None, "[Conv3D] Input cannot have unknown channel!"
		assert in_channel % split == 0

		assert kernel_regularizer is None and bias_regularizer is None and activity_regularizer is None, \
			"Not supported by group conv now!"

		out_channel = filters
		assert out_channel % split == 0
		assert dilation_rate == (1, 1) or get_tf_version_number() >= 1.5, 'TF>=1.5 required for group dilated conv'

		kernel_shape = shape2d(kernel_size)
		filter_shape = kernel_shape + [in_channel / split, out_channel]
		stride = shape4d(strides, data_format=data_format)

		kwargs = dict(data_format=data_format)
		if get_tf_version_number() >= 1.5:
			kwargs['dilations'] = shape4d(dilation_rate, data_format=data_format)

		W = tf.get_variable(
			'W', filter_shape, initializer=kernel_initializer)

		if use_bias:
			b = tf.get_variable('b', [out_channel], initializer=bias_initializer)

		inputs = tf.split(inputs, split, channel_axis)
		kernels = tf.split(W, split, 3)
		outputs = [tf.nn.conv2d(i, k, stride, padding.upper(), **kwargs)
				   for i, k in zip(inputs, kernels)]
		conv = tf.concat(outputs, channel_axis)
		if activation is None:
			activation = tf.identity
		ret = activation(tf.nn.bias_add(conv, b, data_format=data_format) if use_bias else conv, name='output')

		ret.variables = VariableHolder(W=W)
		if use_bias:
			ret.variables.b = b
	return ret
###############################################################################
@layer_register(log_shape=True)
@convert_to_tflayer_args(
	args_names=['filters', 'kernel_size', 'strides'],
	name_mapping={
		'out_channel': 'filters',
		'kernel_shape': 'kernel_size',
		'stride': 'strides',
	})
def Conv3DTranspose(
		inputs,
		filters,
		kernel_size,
		strides=(1, 1, 1),
		padding='same',
		data_format='channels_last',
		activation=None,
		use_bias=False,
		kernel_initializer=tf.contrib.layers.variance_scaling_initializer(2.0), #tf.contrib.layers.xavier_initializer(), #tf.initializers.variance_scaling(distribution='uniform'),
		bias_initializer=tf.zeros_initializer(),
		kernel_regularizer=None,
		bias_regularizer=None,
		activity_regularizer=None):
	"""
	A wrapper around `tf.layers.Conv2DTranspose`.
	Some differences to maintain backward-compatibility:
	1. Default kernel initializer is variance_scaling_initializer(2.0),.
	2. Default padding is 'same'
	Variable Names:
	* ``W``: weights
	* ``b``: bias
	"""

	with rename_get_variable({'kernel': 'W', 'bias': 'b'}):
		layer = tf.layers.Conv3DTranspose(
			filters,
			kernel_size,
			strides=strides,
			padding=padding,
			data_format=data_format,
			activation=activation,
			use_bias=use_bias,
			kernel_initializer=kernel_initializer,
			bias_initializer=bias_initializer,
			kernel_regularizer=kernel_regularizer,
			bias_regularizer=bias_regularizer,
			activity_regularizer=activity_regularizer)
		ret = layer.apply(inputs, scope=tf.get_variable_scope())

	ret.variables = VariableHolder(W=layer.kernel)
	if use_bias:
		ret.variables.b = layer.bias
	return tf.identity(ret, name='output')


Deconv3D = Conv3DTranspose
###################################################################################################
@layer_register(log_shape=True)
def tf_batch_to_space(inputs, crops=[[0,0],[0,0]], block_size=2, name=None):
	return tf.batch_to_space(inputs, crops, block_size, name)

@layer_register(log_shape=True)
def tf_space_to_batch(inputs, paddings=[[0,0],[0,0]], block_shape=2, name=None):
	return tf.space_to_batch(inputs, paddings, block_shape, name)

@layer_register(log_shape=True)
def tf_tranpose(inputs, perm=[0, 2, 3, 1, 4], name=None):
	return tf.tranpose(inputs, perm, name)
###################################################################################################
class Model(ModelDesc):
	def _build_adain_layers(self, content, style, eps=1e-6, name='adain', is_normalized=True):
		with varreplace.freeze_variables():
			# if is_normalized:
			# 	content = normalize(content)
			# 	style   = normalize(style)
			c_mean, c_var = tf.nn.moments(content, axes=[0,1,2], keep_dims=True)
			s_mean, s_var = tf.nn.moments(style,   axes=[0,1,2], keep_dims=True)
			c_std, s_std  = tf.sqrt(c_var + eps), tf.sqrt(s_var + eps)

			results = (s_std * (content - c_mean) / c_std + s_mean)
			if is_normalized:
				results = normalize(results)
			return results

	

	@auto_reuse_variable_scope
	def vol3d_encoder(self, x, name='Vol3D_Encoder'):
		with argscope([Conv3D], kernel_shape=3, padding='SAME', nl=tf.nn.leaky_relu):
			x = tf_2tanh(x)
			x = tf.expand_dims(x, axis=0) # to 1 256 256 256 3
			x = tf.transpose(x, [4, 1, 2, 3, 0]) # here 3x256x256x256x1
			x = (LinearWrap(x)
				.Conv3D('conv1a',   8, strides = 2, padding='SAME') #here 3x128x128x128x8
				.Conv3D('conv2a',  16, strides = 2, padding='SAME') #here 3x64x64x64x16
				.Conv3D('conv3a',  32, strides = 2, padding='SAME') #here 3x32x32x32x32
				.Conv3D('conv4a',  64, strides = 2, padding='SAME') #here 3x16x16x16x64
				.Conv3D('conv5a', 128, strides = 2, padding='SAME', nl=tf.tanh) #here 3x8x8x8x128
				()) 
			x = tf_2imag(x)
			x = tf.transpose(x, [4, 1, 2, 3, 0]) ##here 128x8x8x8x3
			# x = tf.reshape(x, [-1, DIMY, DIMX, 3]) #here 256x256x3
			x = tf.reshape(x, [-1, 8, 8, 3]) #here 256x256x3
			x = tf.batch_to_space(x, crops=[[0,0],[0,0]], block_size=32,name='b2s')
			return x

	@auto_reuse_variable_scope
	def vol3d_decoder(self, x, name='Vol3D_Decoder'):
		with argscope([Conv3DTranspose], kernel_shape=3, padding='SAME', nl=tf.nn.leaky_relu):
			x = tf_2tanh(x)
			x = tf.space_to_batch(x, paddings=[[0,0],[0,0]], block_size=32 ,name='s2b')
			x = tf.reshape(x, [-1, 8, 8, 8, 3]) 
			x = tf.transpose(x, [4, 1, 2, 3, 0]) # #here 3x8x8x8x128
			x = (LinearWrap(x)
				.Conv3DTranspose('conv5b',  64, strides = 2, padding='SAME') #here 3x16x16x16x64
				.Conv3DTranspose('conv4b',  32, strides = 2, padding='SAME') #here 3x32x32x32x32
				.Conv3DTranspose('conv3b',  16, strides = 2, padding='SAME') #here 3x64x64x64x16
				.Conv3DTranspose('conv2b',   8, strides = 2, padding='SAME') #here 3x128x128x128x8
				.Conv3DTranspose('conv1b',   1, strides = 2, padding='SAME', nl=tf.tanh) #here 3x256x256x256x1
				()) 
			x = tf_2imag(x)
			x = tf.transpose(x, [4, 1, 2, 3, 0]) # here 1x256x256x256x3
			x = tf.squeeze(x)
			return x

	@auto_reuse_variable_scope
	def vgg19_encoder(self, inputs, name='VGG19_Encoder'):
		with varreplace.freeze_variables():
			with argscope([Conv2D], kernel_shape=3, nl=tf.nn.relu):
				# print(inputs.get_shape())
				inputs  = inputs - VGG19_MEAN_TENSOR
				conv1_1 = Conv2D('conv1_1', inputs,  64)
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
				
				return normalize(conv4_1)

	@auto_reuse_variable_scope
	def vgg19_decoder(self, inputs, name='VGG19_Decoder'):
		# with varreplace.freeze_variables():
			with argscope([Conv2D], kernel_shape=3, nl=tf.nn.relu):	
				with argscope([Deconv2D], kernel_shape=3, strides=(2,2), nl=tf.nn.relu):
					pool3 = Subpix2D('pool3',   inputs,  256)  # 16
					conv3_4 = Conv2D('conv3_4', pool3,   256)
					conv3_3 = Conv2D('conv3_3', conv3_4, 256)
					conv3_2 = Conv2D('conv3_2', conv3_3, 256)
					conv3_1 = Conv2D('conv3_1', conv3_2, 256)
					pool2 = Subpix2D('pool2',   conv3_1, 128)  # 8
					conv2_2 = Conv2D('conv2_2', pool2, 	 128)
					conv2_1 = Conv2D('conv2_1', conv2_2, 128)
					pool1 = Subpix2D('pool1',   conv2_1, 64)  # 64
					conv1_2 = Conv2D('conv1_2', pool1, 	 64)
					conv1_1 = Conv2D('conv1_1', conv1_2, 64)
					conv1_0 = Conv2D('conv1_0', conv1_1, 3)
					conv1_0 = conv1_0 + VGG19_MEAN_TENSOR
					return conv1_0 # List of feature maps
		
	def _get_inputs(self):
		return [
			InputDesc(tf.float32, (DIMZ, DIMY, DIMX, 3), 'image'),
			InputDesc(tf.float32, (   1, DIMY, DIMX, 3), 'style'),
			InputDesc(tf.int32,                  (1, 1), 'condition'),
			]

									
	def _build_graph(self, inputs):
		# sImg2d # sImg the projection 2D, reshape from 
		

		vol3d, img2d, condition = inputs # Split the input

		with tf.variable_scope('gen'):
			with tf.variable_scope('encoder_3d'):
				vol2d = self.vol3d_encoder(vol3d)

			with tf.variable_scope('style_transfer'):
				with varreplace.freeze_variables():
					merge = self._build_adain_layers(vol2d, img2d)
					condition = tf.reshape(condition, []) # Make 0 rank for condition
					chose = tf.cond(condition > 0, # if istest turns on, perform statistical transfering
											lambda: tf.identity(merge), 
											lambda: tf.identity(vol2d)) #else get the img2d
					img2d = tf.identity(img2d)

			with tf.variable_scope('decoder_3d'):
				vol3d_decoded = self.vol3d_decoder(vol2d)
				img3d   	  = self.vol3d_decoder(img2d)


			with tf.variable_scope('encoder_3d'):
				img2d_decoded = self.vol3d_encoder(img3d)




		#
		# Build losses 
		#
		with tf.name_scope('losses'):
			losses = []
			# Content loss between t and f(g(t))
			loss_vol3d = tf.reduce_mean(tf.abs(vol3d - vol3d_decoded), name='loss_vol3d')
			loss_img2d = tf.reduce_mean(tf.abs(img2d - img2d_decoded), name='loss_img2d')


			add_moving_summary(loss_vol3d)
			add_moving_summary(loss_img2d)


			losses.append(1e0*loss_vol3d)
			losses.append(1e0*loss_img2d)
		
		self.cost = tf.reduce_sum(losses, name='self.cost')
		add_moving_summary(self.cost)
		# self.scope_1_loss =  tf.reduce_sum([2e1*loss_vol3d, 1e0*loss_img3d], name='scope_1_loss')
		# self.scope_2_loss =  tf.reduce_sum([1e1*loss_img2d, 1e0*loss_vol2d], name='scope_2_loss')
		# self.build_losses(self.scope_1_loss, self.scope_2_loss)
		# self.collect_variables(scope_1='scope_1', scope_2='scope_2')
		

		out_vol3d 			= tf.identity(vol3d, 		 name='out_vol3d')
		out_vol3d_decoded 	= tf.identity(vol3d_decoded, name='out_vol3d_decoded')
		with tf.name_scope('visualization'):
			def tf_squeeze(any_tensor):
				return tf.reshape(tf.squeeze(any_tensor), [1, DIMY, DIMX, 3])
			mid=128
			# viz_vol_0 = vol3d[mid-2:mid-1,...]
			# viz_vol_1 = vol3d[mid-1:mid-0,...]
			# viz_vol_2 = vol3d[mid+0:mid+1,...]
			# viz_vol_3 = vol3d[mid+1:mid+2,...]

			# viz_vol_4 = vol3d_decoded[mid-2:mid-1,...]
			# viz_vol_5 = vol3d_decoded[mid-1:mid-0,...]
			# viz_vol_6 = vol3d_decoded[mid+0:mid+1,...]
			# viz_vol_7 = vol3d_decoded[mid+1:mid+2,...]
			viz_vol_1 = tf_squeeze(vol3d[mid:mid+1,...])
			viz_vol_2 = tf_squeeze(vol3d[:,mid:mid+1,...])
			viz_vol_3 = tf_squeeze(vol3d[:,:,mid:mid+1,...])
			viz_vol_0 = tf_squeeze(tf.zeros_like(viz_vol_1))
			
			viz_vol_5 = tf_squeeze(vol3d_decoded[mid:mid+1,...])
			viz_vol_6 = tf_squeeze(vol3d_decoded[:,mid:mid+1,...])
			viz_vol_7 = tf_squeeze(vol3d_decoded[:,:,mid:mid+1,...])
			viz_vol_4 = tf_squeeze(tf.zeros_like(viz_vol_5))


			viz_vol_8 = vol2d
			# viz_vol_9 = vol2d_decoded
			####
			# viz_img_0 = img3d_decoded[mid-2:mid-1,...]
			# viz_img_1 = img3d_decoded[mid-1:mid-0,...]
			# viz_img_2 = img3d_decoded[mid+0:mid+1,...]
			# viz_img_3 = img3d_decoded[mid+1:mid+2,...]

			viz_img_1 = tf_squeeze(img3d[mid:mid+1,...])
			viz_img_2 = tf_squeeze(img3d[:,mid:mid+1,...])
			viz_img_3 = tf_squeeze(img3d[:,:,mid:mid+1,...])
			viz_img_0 = tf_squeeze(tf.zeros_like(viz_img_1))

			viz_img_4 = img2d
			viz_img_5 = img2d_decoded
			# viz_img_6 = img3d_encoded


			viz_zeros = tf.zeros_like(img2d)
			# Visualization
			viz = tf.concat([tf.concat([viz_vol_1, viz_vol_2, viz_vol_3, viz_img_4], 2), 
							 tf.concat([viz_vol_5, viz_vol_6, viz_vol_7, viz_vol_8], 2), 
							 tf.concat([viz_img_1, viz_img_2, viz_img_3, viz_img_5], 2), 
							 ], 1)

			viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
			tf.summary.image('colorized', viz, max_outputs=50)

	def _get_optimizer(self):
		lr  = tf.get_variable('learning_rate', initializer=2e-4, trainable=False)
		opt = tf.train.AdamOptimizer(lr)
		# opt = tf.train.RMSPropOptimizer(lr)

		return opt
####################################################################################################
class ImageDataFlow(RNGDataFlow):
	def __init__(self, image_path, style_path, size, dtype='float32', isTrain=False, isValid=False, isTest=False):
		self.dtype      	= dtype
		self.image_path   	= image_path
		self.style_path   	= style_path
		self._size      	= size
		self.isTrain    	= isTrain
		self.isValid    	= isValid
		self.isTest   	 	= isTest

		#
		# Read and store into pairs of images and styles
		#
		images = natsorted(glob.glob(self.image_path + '/*.tif'))
		styles = natsorted(glob.glob(self.style_path + '/*.jpg'))
		self.images = []
		self.styles = []
		# self.data_seed = time_seed ()
		# self.data_rand = np.random.RandomState(self.data_seed)
		self.rng = np.random.RandomState(999)

		for i in range (len (images)):
			image = images[i]
			image = skimage.io.imread (image)
			self.images.append(image)
		
		for i in range (len (styles)):
			style = styles[i]
			style = cv2.imread(style, cv2.IMREAD_COLOR)
			style = cv2.cvtColor(style, cv2.COLOR_BGR2RGB) # BGR to RGB
			self.styles.append (style)

		# self._size = 0
		# for i in range (len (self.images)):
		# 	self._size += self.images[i].shape[0] * self.images[i].shape[1] * self.images[i].shape[2] \
		# 	        / (input_shape[0] * input_shape[1] * input_shape[2])

	def size(self):
		return self._size

	def reset_state(self):
		self.rng = get_rng(self)

	def get_data(self, shuffle=True):
		
	
		#
		# Pick the image over size 
		#
		for k in range(self._size):
			#
			# Pick randomly a tuple of training instance
			#
			

			if self.isTrain:
				# Read image
				rand_image = self.rng.randint(0, len(self.images))
				rand_style = self.rng.randint(0, len(self.styles))
				image = self.images[rand_image].copy ()
				style = self.styles[rand_style].copy ()

				# Image augmentation
				image = self.random_flip(image)
				image = self.random_square_rotate(image)
				image = self.random_reverse(image)
				image = self.random_permute(image)
				image = self.random_pad(image)

				# Style augmentation
				style = self.central_crop(style)
				is_random_crop = self.rng.randint(0, 2) # 50% of cropping
				if is_random_crop:
					style = self.resize_image(style, size=512)
					style = self.random_crop (style, size=256)
				else: 
					style = self.resize_image(style, size=256)

			elif self.isValid:
				# Read image
				rand_image = self.rng.randint(0, len(self.images))
				rand_style = self.rng.randint(0, len(self.styles))
				image = self.images[rand_image].copy ()
				style = self.styles[rand_style].copy ()


				image = self.random_pad(image, symmetry=True)
				# Style augmentation
				style = self.central_crop(style)
				style = self.resize_image(style, size=256)
			elif self.isTest:
				for i in range(len(self.images)):
					for s in range(len(self.styles)):
						# Read image
						image = self.images[i].copy ()
						style = self.styles[s].copy ()


						image = self.random_pad(image, symmetry=True)
						# Style augmentation
						style = self.central_crop(style)
						style = self.resize_image(style, size=256)
						
						# Make RGB volume
						image = np.stack([image,image,image], axis=3) #012 to 0123
						image = np.squeeze(image)
						condition = 0 if self.isTrain else 1
						condition = np.array(condition)
						style     = np.expand_dims(style, axis=0)
						condition = np.expand_dims(condition, axis=0)
						condition = np.expand_dims(condition, axis=0)
						# print(image.shape)
						# print(style.shape)
						# print(condition.shape)
						yield [
							   image.astype(np.float32), 
							   style.astype(np.float32), 
							   condition.astype(np.int32)
							   ]

			if not self.isTest:
				# Make RGB volume
				image = np.stack([image,image,image], axis=3) #012 to 0123
				image = np.squeeze(image)
				condition = 0 if self.isTrain else 1
				condition = np.array(condition)
				style     = np.expand_dims(style, axis=0)
				condition = np.expand_dims(condition, axis=0)
				condition = np.expand_dims(condition, axis=0)
				yield [
					   image.astype(np.float32), 
					   style.astype(np.float32), 
					   condition.astype(np.int32)
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
		if dimy > dimx:
			image = skimage.transform.resize(image, output_shape=[dimy*size//dimx, size], preserve_range=True)
		else:
			image = skimage.transform.resize(image, output_shape=[size, dimx*size//dimy], preserve_range=True)
		return image

	def random_crop(self, image, size=256):
		shape = image.shape
		dimy, dimx = shape[0], shape[1]
		assert size<=dimx and size<=dimy
		randy = self.rng.randint(0, dimy-size+1)
		randx = self.rng.randint(0, dimx-size+1)
		image = image[randy:randy+size,randx:randx+size,...]
		return image

	
	def random_pad(self, image, target_shape=[DIMZ, DIMY, DIMX], seed=None, symmetry=False):
		assert ((image.ndim == 3))
		if seed:
			self.rng.seed(seed)
		
		dimz, dimy, dimx = image.shape
		assert (dimz <= DIMZ)
		assert (dimy <= DIMY)
		assert (dimx <= DIMX)
		padded = np.zeros([DIMZ, DIMY, DIMX], dtype=np.float32)
		if not symmetry:
			offset = [self.rng.randint(0, DIMZ-dimz+1), 
					  self.rng.randint(0, DIMY-dimy+1), 
					  self.rng.randint(0, DIMX-dimx+1), 
					  ]
		else:
			offset = [int((DIMZ-dimz)/2), 
					  int((DIMY-dimy)/2), 
					  int((DIMX-dimx)/2), 
					  ]
		padded[offset[0]:offset[0]+dimz,
			   offset[1]:offset[1]+dimy,
			   offset[2]:offset[2]+dimx]	= image
		return padded

	def random_flip(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			self.rng.seed(seed)
		random_flip = self.rng.randint(1,5)
		if random_flip==1:
			flipped = image[...,::1,::-1]
			image = flipped
		elif random_flip==2:
			flipped = image[...,::-1,::1]
			image = flipped
		elif random_flip==3:
			flipped = image[...,::-1,::-1]
			image = flipped
		elif random_flip==4:
			flipped = image
			image = flipped
		return image

	def random_reverse(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			self.rng.seed(seed)
		random_reverse = self.rng.randint(1,3)
		if random_reverse==1:
			reverse = image[::1,...]
		elif random_reverse==2:
			reverse = image[::-1,...]
		image = reverse
		return image

	def random_square_rotate(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			self.rng.seed(seed)        
		random_rotatedeg = 90*self.rng.randint(0,4)
		rotated = image.copy()
		from scipy.ndimage.interpolation import rotate
		if image.ndim==2:
			rotated = rotate(image, random_rotatedeg, axes=(0,1))
		elif image.ndim==3:
			rotated = rotate(image, random_rotatedeg, axes=(1,2))
		image = rotated
		return image

	def random_permute(self, image, seed=None):
		assert ((image.ndim == 2) | (image.ndim == 3))
		if seed:
			self.rng.seed(seed)
		permuted_image = np.transpose(image.copy(), self.rng.permutation(image.ndim))
		return permuted_image
	
				
	
####################################################################################################
def get_data(image_path, style_path, size=EPOCH_SIZE):
	ds_train = ImageDataFlow(image_path=image_path,
							 style_path=style_path, 
							 size=size, 
							 isTrain=True
							 )

	ds_valid = ImageDataFlow(image_path=image_path.replace('train','valid'),
							 style_path=style_path.replace('train','valid'), 
							 size=20, 
							 isValid=True
							 )
	ds_test2 = ImageDataFlow(image_path=image_path.replace('train','valid'),
						 style_path=style_path.replace('train','valid'), 
						 size=1, 
						 isTest=True
						 )

	ds_train.reset_state()
	ds_valid.reset_state() 
	ds_test2.reset_state() 

	# ds_train = BatchData(ds_train, BATCH_SIZE)
	# ds_valid = BatchData(ds_valid, BATCH_SIZE)

	ds_train = PrefetchDataZMQ(ds_train, 2)
	# ds_valid = PrefetchDataZMQ(ds_valid, 2)
	return ds_train, ds_valid, ds_test2

###################################################################################################
def render(model_path, image_path, style_path):
	pass
	

###################################################################################################
class VisualizeRunner(Callback):
	def __init__(self, input, tower_name='InferenceTower', device=0):
		self.dset = input 
		self._tower_name = tower_name
		self._device = device

	def _setup_graph(self):
		self.pred = self.trainer.get_predictor(
			['image', 'style', 'condition'], ['visualization/viz'])

	def _before_train(self):
		pass 

	def _trigger(self):
		for lst in self.dset.get_data():
			viz_valid = self.pred(lst)
			viz_valid = np.squeeze(np.array(viz_valid))

			#print viz_valid.shape

			self.trainer.monitors.put_image('viz_valid', viz_valid[...,::1])
###################################################################################################
class RenderingRunner(Callback):
	def __init__(self, input, tower_name='InferenceTower', device=0):
		self.dset = input 
		self._tower_name = tower_name
		self._device = device

	def _setup_graph(self):
		self.pred = self.trainer.get_predictor(
			['image', 'style', 'condition'], ['out_vol3d', 'out_vol3d_decoded'])

	def _before_train(self):
		pass 

	def _trigger(self):
		image_path = args.image_path
		style_path = args.style_path

		# shutil.move('result-*', 'result-latest')
		# resultDir = time.strftime("result-%Y-%m-%d-%H-%m-%S")
		resultDir = 'result_stylish_3d_vgg_cycle_888/'
		shutil.rmtree(resultDir, ignore_errors=True)
		os.makedirs(resultDir)

		images = glob.glob(image_path + '/*.tif')
		images = natsorted(images)
		styles = glob.glob(style_path + '/*.jpg')
		styles = natsorted(styles)

		filename_vol = ''
		for idx, lst in enumerate(self.dset.get_data()):
			# viz_valid = self.pred(lst)
			# viz_valid = np.squeeze(np.array(viz_valid))

			
			if True:
				## Extract stack of images with SimpleDatasetPredictor
				o_vol3d, o_vol3d_decoded = self.pred(lst)
				
				# for idx, outs in enumerate(pred.get_result()):
				# 	o_vol3d 		= np.array(outs[0][:, :, :, :])
				# 	o_vol3d_decoded = np.array(outs[1][:, :, :, :])

				# Calculate the index for result
				idx_image = idx//len(images) 
				idx_style = idx%len(images)

				# head, tail = os.path.split(images[idx_image])

				# filename_old = os.path.join(resultDir, tail.replace('.tif', '')+ '_style_' + 'intensity' +'.tif')
				# filename_new = os.path.join(resultDir, tail.replace('.tif', '')+ '_style_' + str(idx_style).zfill(2) +'.tif')
				head, tail = os.path.split(images[idx_image])
				tail= tail.split('.', 1)[0] # Split the first ocurrent
				filename_old = os.path.join(resultDir, tail+'99_256_256_256.raw')
				filename_new = os.path.join(resultDir, tail+str(idx_style).zfill(2)+'_256_256_256.raw')
				
				# Save the style:
				if idx_style == idx:
					filename_img = os.path.join(resultDir, 'style'+str(idx).zfill(2)+'.jpg')
					print("Saving "+filename_img)
					skimage.io.imsave(filename_img, skimage.io.imread(styles[idx]))

				if filename_old!=filename_vol: # If different from base then save
					filename_vol = filename_old
					# skimage.io.imsave(filename_old, np.squeeze(o_vol3d[...,1].astype(np.uint8)))
					np.squeeze(o_vol3d[...,1].astype(np.uint8)).tofile(filename_old)
					print("Saving "+filename_old)
				#skimage.io.imsave(filename_new, np.squeeze(o_vol3d_decoded.astype(np.uint8)))
				np.squeeze(o_vol3d_decoded.astype(np.uint8)).tofile(filename_new)
				print("Saving "+filename_new)
###################################################################################################
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', 		help='comma separated list of GPU(s) to use.')
	parser.add_argument('--load', 		help='load model')
	parser.add_argument('--vgg19', 		help='load model', 		   default='data/model/vgg19.npz') #vgg19.npz')
	parser.add_argument('--image_path', help='path to the image.', default='data/train/image/') 
	parser.add_argument('--style_path', help='path to the style.', default='data/train/style/') 
	parser.add_argument('--alpha',      help='Between 0 and 1',    default=1.0,  type=float)
	parser.add_argument('--lambda',     help='Between 0 and 1',    default=1e-0, type=float)
	parser.add_argument('--weight_c',   help='Between 0 and 1',    default=1e-0, type=float)
	parser.add_argument('--weight_s',   help='Between 0 and 1',    default=1e-2, type=float)
	parser.add_argument('--weight_tv',  help='Between 0 and 1',    default=1e-4, type=float)
	parser.add_argument('--render', 	action='store_true')
	
	global args
	args = parser.parse_args()

	if args.gpu:
		os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

	if args.render:
		render(args.load, args.image_path, args.style_path)
	else:
		# Set the logger directory
		logger.auto_set_dir()

		nr_tower = max(get_nr_gpu(), 1)

		ds_train, ds_valid, ds_test2 = get_data(args.image_path, args.style_path)

		model = Model()

		if args.load:
			session_init = SaverRestore(args.load)
		else: # For training from scratch, read the vgg model from args.vgg19
			assert os.path.isfile(args.vgg19)

			weight = dict(np.load(args.vgg19))
			param_dict = {}
			param_dict.update({'gen/encoder_vgg19/' + name: value for name, value in six.iteritems(weight)})
			session_init = DictRestore(param_dict)



		
		# Set up configuration
		config = TrainConfig(
			model           =   model, 
			dataflow        =   ds_train,
			callbacks       =   [
				PeriodicTrigger(ModelSaver(), every_k_epochs=10),
				PeriodicTrigger(VisualizeRunner(ds_valid), every_k_epochs=1),
				PeriodicTrigger(RenderingRunner(ds_test2), every_k_epochs=100),
				PeriodicTrigger(InferenceRunner(ds_valid, [ScalarStats('losses/loss_img2d'), 
														   # ScalarStats('losses/loss_img3d'), 
														   # ScalarStats('losses/loss_vol2d'), 
														   ScalarStats('losses/loss_vol3d'), 
														   ScalarStats('self.cost'), 
														   ]), every_k_epochs=1),
				ScheduledHyperParamSetter('learning_rate', [(0, 2e-5), (100, 1e-5), (200, 1e-5), (300, 1e-6)], interp='linear')
				#ScheduledHyperParamSetter('learning_rate', [(30, 6e-6), (45, 1e-6), (60, 8e-7)]),
				#HumanHyperParamSetter('learning_rate'),
				],
			max_epoch       =   500, 
			session_init    =   session_init,
			nr_tower        =   max(get_nr_gpu(), 1)
			)
	
		# Train the model
		# SyncMultiGPUTrainer(config).train()
		launch_train_with_config(
			config,
			QueueInputTrainer())
		# trainer = ChainTrainer(QueueInput(ds_train), model)
		# trainer.train_with_defaults(
		# 	max_epoch       =   500, 
		# 	steps_per_epoch =	ds_train.size(),
		# 	session_init    =   SaverRestore(args.load) if args.load else session_init,
		# 	callbacks       =   [
		# 		PeriodicTrigger(ModelSaver(), every_k_epochs=10),
		# 		PeriodicTrigger(VisualizeRunner(ds_valid), every_k_epochs=1),
		# 		PeriodicTrigger(RenderingRunner(ds_test2), every_k_epochs=100),
		# 		PeriodicTrigger(InferenceRunner(ds_valid, 
		# 										[ScalarStats('losses/loss_img2d'), 
		# 										 ScalarStats('losses/loss_img3d'), 
		# 										 ScalarStats('losses/loss_vol2d'), 
		# 										 ScalarStats('losses/loss_vol3d'), 
		# 									   # ScalarStats('self.cost'), 
		# 												   ]), every_k_epochs=1),
		# 		ScheduledHyperParamSetter('learning_rate', [(0, 2e-5), (100, 1e-5), (200, 1e-5), (300, 1e-6)], interp='linear')
		# 		#ScheduledHyperParamSetter('learning_rate', [(30, 6e-6), (45, 1e-6), (60, 8e-7)]),
		# 		#HumanHyperParamSetter('learning_rate'),
		# 		],
		# )