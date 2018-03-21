#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Tran Minh Quan, quantm@unist.ac.kr
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import os, sys, argparse, glob, cv2, six, h5py



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
from tensorpack.tfutils.summary import add_moving_summary, add_param_summary, add_tensor_summary
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.utils import logger

###################################################################################################
np.warnings.filterwarnings('ignore')

DIMX  = 256
DIMY  = 256
DIMZ  = 256
SIZE  = 256 # For resize

EPOCH_SIZE = 200
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
###################################################################################################
class Model(ModelDesc):
	def _build_adain_layers(self, content, style, eps=1e-6, name='adain', is_normalized=True):
		with tf.variable_scope(name):
			# if is_normalized:
			# 	content = normalize(content)
			# 	style   = normalize(style)
			c_mean, c_var = tf.nn.moments(content, axes=[1,2], keep_dims=True)
			s_mean, s_var = tf.nn.moments(style,   axes=[1,2], keep_dims=True)
			c_std, s_std  = tf.sqrt(c_var + eps), tf.sqrt(s_var + eps)

			results = (s_std * (content - c_mean) / c_std + s_mean)
			if is_normalized:
				results = normalize(results)
			return results

	
	# @auto_reuse_variable_scope
	# def vol3d_encoder(self, x, name='Vol3D_Encoder'):
	# 	# with varreplace.freeze_variables():
	# 		with argscope([Conv2D], kernel_shape=3, nl=INLReLU):
	# 			x = tf.transpose(x, [3, 1, 2, 0]) # from z y x c to c y x z
	# 			x = tf_2tanh(x)
	# 			x = (LinearWrap(x)
	# 					.Conv2D('conv1', DIMZ/2,   padding='SAME') # 128
	# 					.Conv2D('conv2', DIMZ/4,   padding='SAME') # 64				
	# 					.Conv2D('conv3', DIMZ/8,   padding='SAME') # 32
	# 					.Conv2D('conv4', DIMZ/16,  padding='SAME') # 16
	# 					.Conv2D('conv5', DIMZ/32,  padding='SAME') # 8
	# 					.Conv2D('conv6', DIMZ/64,  padding='SAME') # 4
	# 					.Conv2D('conv7', DIMZ/128, padding='SAME') # 2
	# 					.Conv2D('conv8', DIMZ/256, padding='SAME', nl=tf.nn.tanh) # 1
	# 					())
	# 			x = tf.transpose(x, [3, 1, 2, 0]) # from c y x 1 to 1 y x c
	# 			x = tf_2imag(x, maxVal=255.0)
	# 			return x
	
	# @auto_reuse_variable_scope
	# def vol3d_decoder(self, x, name='Vol3D_Decoder'):
	# 	# with varreplace.freeze_variables():
	# 		with argscope([Conv2D], kernel_shape=3, nl=BNLReLU):
	# 			x = tf.transpose(x, [3, 1, 2, 0]) # from 1 y x c to c y x 1
	# 			x = tf_2tanh(x)
	# 			x = (LinearWrap(x)
	# 					.Subpix2D('conv7', DIMZ/128,scale=1, ) # 2
	# 					.Subpix2D('conv6', DIMZ/64, scale=1, ) # 4
	# 					.Subpix2D('conv5', DIMZ/32, scale=1, ) # 8
	# 					.Subpix2D('conv4', DIMZ/16, scale=1, ) # 16
	# 					.Subpix2D('conv3', DIMZ/8,  scale=1, ) # 32
	# 					.Subpix2D('conv2', DIMZ/4,  scale=1, ) # 64				
	# 					.Subpix2D('conv1', DIMZ/2,  scale=1, ) # 128
	# 					.Subpix2D('conv0', DIMZ/1,  scale=1, nl=tf.nn.tanh) # 128
	# 					())
	# 			x = tf.transpose(x, [3, 1, 2, 0]) # from c y x z to z y x c
	# 			x = tf_2imag(x)
	# 			return x


	@auto_reuse_variable_scope
	def vgg19_encoder(self, inputs, name='VGG19_Encoder'):
		inputs  = inputs - VGG19_MEAN_TENSOR
		with argscope([Conv2D], kernel_shape=3, nl=tf.nn.relu):
			with varreplace.freeze_variables():
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
				conv4_2 = Conv2D('conv4_2', conv4_1, 512)
				conv4_3 = Conv2D('conv4_3', conv4_2, 512)
				conv4_4 = Conv2D('conv4_4', conv4_3, 512)
				pool4 = MaxPooling('pool4', conv4_4, 2)  # 8
				conv5_1 = Conv2D('conv5_1', pool4,   512)
				conv5_2 = Conv2D('conv5_2', conv5_1, 512)
				conv5_3 = Conv2D('conv5_3', conv5_2, 512)
				conv5_4 = Conv2D('conv5_4', conv5_3, 512)
				pool5 = MaxPooling('pool5', conv5_4, 2)  # 4
				return normalize(conv4_1), [conv1_1, conv2_1, conv3_1, conv4_1]

	@auto_reuse_variable_scope
	def vgg19_decoder(self, inputs, nl=INLReLU, last_dim=3, name='VGG19_Decoder'):
		with tf.variable_scope(name):
			with argscope([Conv2D], kernel_shape=3, nl=nl):	
				with argscope([Deconv2D], kernel_shape=3, strides=(2,2), nl=nl):
					# conv4_1 = Conv2D('conv4_1', conv4_2, 512)
					pool3 = Subpix2D('pool3',   inputs,  256)  # 16
					conv3_4 = Conv2D('conv3_4', pool3,   256)
					conv3_3 = Conv2D('conv3_3', conv3_4, 256)
					conv3_2 = Conv2D('conv3_2', conv3_3, 256)
					conv3_1 = Conv2D('conv3_1', conv3_2, 256)
					pool2 = Subpix2D('pool2',   conv3_1, 128)  # 32
					conv2_2 = Conv2D('conv2_2', pool2, 	 128)
					conv2_1 = Conv2D('conv2_1', conv2_2, 128)
					pool1 = Subpix2D('pool1',   conv2_1, 64)  # 64
					conv1_2 = Conv2D('conv1_2', pool1, 	 64)
					conv1_1 = Conv2D('conv1_1', conv1_2, 64)
					conv1_0 = Conv2D('conv1_0', conv1_1, last_dim)
					if last_dim != 3:
						conv1_0 = tf.reshape(conv1_0, [-1, DIMY, DIMX, 3])
					conv1_0 = conv1_0 + VGG19_MEAN_TENSOR
					return conv1_0 # List of feature maps

	@auto_reuse_variable_scope
	def vol3d_encoder(self, x, name='Vol3D_Encoder'):
		x, _ = self.vgg19_encoder(x) # from 256x256x256x3 to 256x32x32x256
		x  = tf.reduce_mean(x, axis=0, keepdims=True)
		x  = self.vgg19_decoder(x, last_dim=3, name=name, nl=BNLReLU)
		return x

	@auto_reuse_variable_scope
	def vol3d_decoder(self, x, name='Vol3D_Decoder'):
		x  = tf.tile(x, [DIMZ, 1, 1, 1])
		x, _ = self.vgg19_encoder(x) # from 256x256x256x3 to 256x32x32x256
		x  = self.vgg19_decoder(x, last_dim=3, name=name, nl=BNLReLU)
		return x

	def _get_inputs(self):
		return [
			InputDesc(tf.float32, (DIMZ, DIMY, DIMX, 3), 'image'),
			InputDesc(tf.float32, (   1, DIMY, DIMX, 3), 'style'),
			InputDesc(tf.int32,                  (1, 1), 'condition'),
			]
									
	def _build_graph(self, inputs):
		# sImg2d # sImg the projection 2D, reshape from 
		

		vol3d, img2d, condition = inputs # Split the input

		# Step 0; run thru 3d encoder
		with tf.variable_scope('encoder_3d'):
			vol2d = self.vol3d_encoder(vol3d)
		# Step 1: Run thru the encoder
		with tf.variable_scope('encoder_vgg19_2d'):
			vol2d_encoded, vol2d_feature = self.vgg19_encoder(vol2d)
			img2d_encoded, img2d_feature = self.vgg19_encoder(img2d)
		# Step 2: Run thru the adain block to get t=AdIN(f(c), f(s))
		with tf.variable_scope('style_transfer'):
			merge_encoded = self._build_adain_layers(vol2d_encoded, img2d_encoded)
			condition = tf.reshape(condition, []) # Make 0 rank for condition
			chose_encoded = tf.cond(condition > 0, # if istest turns on, perform statistical transfering
									lambda: tf.identity(merge_encoded), 
									lambda: tf.identity(vol2d_encoded)) #else get the img2d_encoded
			img2d_encoded = tf.identity(img2d_encoded)

		# Step 3: Run thru the decoder to get the paint image
		with tf.variable_scope('decoder_vgg19_2d'):
			vol2d_decoded = self.vgg19_decoder(chose_encoded)
			img2d_decoded = self.vgg19_decoder(img2d_encoded)

		with tf.variable_scope('decoder_3d'):
			vol3d_decoded = self.vol3d_decoder(vol2d_decoded)
			img3d_decoded = self.vol3d_decoder(img2d_decoded)

		# Step 0; run thru 3d encoder
		with tf.variable_scope('encoder_3d'):
			img3d_encoded = self.vol3d_encoder(img3d_decoded)

		#
		# Build losses here
		#
		with tf.name_scope('losses'):
			losses = []
			# Content loss between t and f(g(t))
			# loss_vol2d = tf.reduce_mean(tf.abs(vol2d - vol2d_decoded), name='loss_vol2d')
			loss_vol3d = tf.reduce_mean(tf.abs(vol3d - vol3d_decoded), name='loss_vol3d')
			loss_vol2d = tf.reduce_mean(tf.abs(vol2d - vol2d_decoded), name='loss_vol2d')
			loss_img2d = tf.reduce_mean(tf.abs(img2d - img2d_decoded), name='loss_img2d')
			loss_img3d = tf.reduce_mean(tf.abs(img2d - img3d_encoded), name='loss_img3d')
			# loss_img3d = tf.reduce_mean(tf.abs(img3d - img3d_decoded), name='loss_img3d')


			add_moving_summary(loss_vol3d)
			add_moving_summary(loss_vol2d)
			add_moving_summary(loss_img2d)
			add_moving_summary(loss_img3d)


			losses.append(1e0*loss_vol3d)
			losses.append(2e0*loss_vol2d)
			losses.append(2e0*loss_img2d)
			losses.append(1e0*loss_img3d)
		self.cost = tf.reduce_sum(losses, name='self.cost')
		add_moving_summary(self.cost)

		out_vol3d 			= tf.identity(vol3d, 		 name='out_vol3d')
		out_vol3d_decoded 	= tf.identity(vol3d_decoded, name='out_vol3d_decoded')
		with tf.name_scope('visualization'):
			mid=128
			viz_vol_0 = vol3d[mid-2:mid-1,...]
			viz_vol_1 = vol3d[mid-1:mid-0,...]
			viz_vol_2 = vol3d[mid+0:mid+1,...]
			viz_vol_3 = vol3d[mid+1:mid+2,...]

			viz_vol_4 = vol3d_decoded[mid-2:mid-1,...]
			viz_vol_5 = vol3d_decoded[mid-1:mid-0,...]
			viz_vol_6 = vol3d_decoded[mid+0:mid+1,...]
			viz_vol_7 = vol3d_decoded[mid+1:mid+2,...]

			viz_vol_8 = vol2d
			viz_vol_9 = vol2d_decoded
			####
			viz_img_0 = img3d_decoded[mid-2:mid-1,...]
			viz_img_1 = img3d_decoded[mid-1:mid-0,...]
			viz_img_2 = img3d_decoded[mid+0:mid+1,...]
			viz_img_3 = img3d_decoded[mid+1:mid+2,...]


			viz_img_4 = img2d
			viz_img_5 = img2d_decoded
			viz_img_6 = img3d_encoded


			viz_zeros = tf.zeros_like(img2d)
			# Visualization
			viz = tf.concat([tf.concat([viz_vol_0, viz_vol_1, viz_vol_2, viz_vol_3, viz_vol_8, viz_img_4], 2), 
							 tf.concat([viz_vol_4, viz_vol_5, viz_vol_6, viz_vol_7, viz_vol_9, viz_img_5], 2), 
							 tf.concat([viz_img_0, viz_img_1, viz_img_2, viz_img_3, viz_img_6, viz_img_4], 2), 
							 ], 1)

			viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
			tf.summary.image('colorized', viz, max_outputs=50)

	def _get_optimizer(self):
		lr  = tf.get_variable('learning_rate', initializer=2e-4, trainable=False)
		opt = tf.train.AdamOptimizer(lr)
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

	def size(self):
		return self._size

	def reset_state(self):
		self.rng = get_rng(self)

	def get_data(self, shuffle=True):
		#
		# Read and store into pairs of images and labels
		#
		images = glob.glob(self.image_path + '/*.tif')
		styles = glob.glob(self.style_path + '/*.jpg')

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
			

			if self.isTrain:
				# Read image
				rand_image = self.rng.randint(0, len(images))
				rand_style = self.rng.randint(0, len(styles))
				image = skimage.io.imread(images[rand_image])
				style = cv2.imread(styles[rand_style], cv2.IMREAD_COLOR)

				# Image augmentation
				image = self.random_flip(image)
				image = self.random_square_rotate(image)
				image = self.random_reverse(image)
				image = self.random_permute(image)
				image = self.random_pad(image)

				# Style augmentation
				style = self.central_crop(style)
				is_random_crop = self.rng.randint(0, 2)
				if is_random_crop:
					style = self.resize_image(style, size=512)
					style = self.random_crop (style, size=256)
				else: 
					style = self.resize_image(style, size=256)
				
			elif self.isValid:
				# Read image
				rand_image = self.rng.randint(0, len(images))
				rand_style = self.rng.randint(0, len(styles))
				image = skimage.io.imread(images[rand_image])
				style = cv2.imread(styles[rand_style], cv2.IMREAD_COLOR)

				image = self.random_pad(image, symmetry=True)
				# Style augmentation
				style = self.central_crop(style)
				style = self.resize_image(style, size=256)
			elif self.isTest:
				# Read image
				rand_image = self.rng.randint(0, len(images))
				rand_style = self.rng.randint(0, len(styles))
				image = skimage.io.imread(images[rand_image])
				style = cv2.imread(styles[rand_style], cv2.IMREAD_COLOR)

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

	ds_train.reset_state()
	ds_valid.reset_state() 

	# ds_train = BatchData(ds_train, BATCH_SIZE)
	# ds_valid = BatchData(ds_valid, BATCH_SIZE)

	ds_train = PrefetchDataZMQ(ds_train, 2)
	# ds_valid = PrefetchDataZMQ(ds_valid, 2)
	return ds_train, ds_valid

###################################################################################################
def render(model_path, volume_path, style_path):
	pass


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

			self.trainer.monitors.put_image('viz_valid', viz_valid[...,::-1])
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

			weight = dict(np.load(args.vgg19))
			param_dict = {}
			param_dict.update({'encoder_vgg19_2d/' + name: value for name, value in six.iteritems(weight)})
			session_init = DictRestore(param_dict)



		
		# Set up configuration
		config = TrainConfig(
			model           =   model, 
			dataflow        =   ds_train,
			callbacks       =   [
				PeriodicTrigger(ModelSaver(), every_k_epochs=10),
				PeriodicTrigger(VisualizeRunner(ds_valid), every_k_epochs=1),
				PeriodicTrigger(InferenceRunner(ds_valid, [ScalarStats('losses/loss_img3d'), 
														   ScalarStats('losses/loss_img2d'), 
														   ScalarStats('losses/loss_vol2d'), 
														   ScalarStats('losses/loss_vol3d'), 
														   ScalarStats('self.cost'), 
														   ]), every_k_epochs=1),
				ScheduledHyperParamSetter('learning_rate', [(0, 2e-4), (100, 1e-4), (200, 1e-5), (300, 1e-6)], interp='linear')
				#ScheduledHyperParamSetter('learning_rate', [(30, 6e-6), (45, 1e-6), (60, 8e-7)]),
				#HumanHyperParamSetter('learning_rate'),
				],
			max_epoch       =   500, 
			session_init    =   session_init,
			nr_tower        =   max(get_nr_gpu(), 1)
			)
	
		# Train the model
		SyncMultiGPUTrainerReplicated(config).train()