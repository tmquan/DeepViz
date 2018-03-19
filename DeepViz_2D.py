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
SIZE  = 256 # For resize

EPOCH_SIZE = 2000
BATCH_SIZE = 10
###############################################################################
# Utility function for scaling 
def tf_2tanh(x, maxVal = 255.0, name='ToRangeTanh'):
	with tf.variable_scope(name):
		return (x / maxVal - 0.5) * 2.0
###############################################################################
def tf_2imag(x, maxVal = 255.0, name='ToRangeImag'):
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

@layer_register(log_shape=True)
def Subpix2D(inputs, chan, scale=2, stride=1):
	with argscope([Conv2D], nl=INReLU, stride=stride, kernel_shape=3):
		results = Conv2D('conv0', inputs, chan* scale**2, padding='SAME')
		if scale>1:
			results = tf.depth_to_space(results, scale, name='depth2space', data_format='NHWC')
		return results
###################################################################################################
class Model(ModelDesc):
	def _get_inputs(self):
		return [
			InputDesc(tf.float32, (BATCH_SIZE, DIMY, DIMX, 3), 'image'),
			InputDesc(tf.float32, (BATCH_SIZE, DIMY, DIMX, 3), 'style'),
			]
			
	def _build_adain_layers(self, content, style, eps=1e-6, name='adain'):
		with tf.variable_scope(name):
			c_mean, c_var = tf.nn.moments(content, axes=[1,2], keep_dims=True)
			s_mean, s_var = tf.nn.moments(style, axes=[1,2], keep_dims=True)
			c_std, s_std = tf.sqrt(c_var + eps), tf.sqrt(s_var + eps)

			return s_std * (content - c_mean) / c_std + s_mean

	def _build_content_loss(self, current, target, weight=1.0):
		loss = tf.reduce_mean(tf.squared_difference(current, target))
		return loss*weight

	def _build_style_losses(self, current_layers, target_layers, weight=1.0, eps=1e-6):
		losses = []
		# for layer in current_layers:
		for current, target in zip(current_layers, target_layers):
			# current, target = current_layers[layer], target_layers[layer]
			axes = [1,2]
			current_mean, current_var = tf.nn.moments(current, axes=[1,2], keep_dims=True) # Normalize to 2,3 is for NCHW; 1,2 is for NHWC
			current_std = tf.sqrt(current_var + eps)

			target_mean, target_var   = tf.nn.moments(target, axes=[1,2], keep_dims=True) # Normalize to 2,3 is for NCHW; 1,2 is for NHWC
			target_std = tf.sqrt(target_var + eps)

			mean_loss = tf.reduce_sum(tf.squared_difference(current_mean, target_mean))
			std_loss  = tf.reduce_sum(tf.squared_difference(current_std, target_std))

			# normalize w.r.t batch size
			n = tf.cast(tf.shape(current)[0], dtype=tf.float32)
			mean_loss /= n
			std_loss  /= n

			# losses[layer] = (mean_loss + std_loss) * weight
			losses.append((mean_loss + std_loss) * weight)
		return losses


	def _build_graph(self, inputs):
		# sImg2d # sImg the projection 2D, reshape from 
		VGG19_MEAN = np.array([123.68, 116.779, 103.939])  # RGB
		VGG19_MEAN_TENSOR = tf.constant(VGG19_MEAN, dtype=tf.float32)

		image, style = inputs # Split the input
		
		image = image - VGG19_MEAN_TENSOR
		style = style - VGG19_MEAN_TENSOR
		
		@auto_reuse_variable_scope
		def vgg19_encoder(source, name='VGG19_Encoder'):
			with tf.variable_scope(name):
				with varreplace.freeze_variables():
					with argscope([Conv2D], kernel_shape=3, nl=tf.nn.relu):
						conv1_1 = Conv2D('conv1_1', source,  64)
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
						return conv4_1, [conv1_1, conv2_1, conv3_1, conv4_1]

		@auto_reuse_variable_scope
		def vgg19_decoder(source, name='VGG19_Decoder'):
			with tf.variable_scope(name):
				# with varreplace.freeze_variables():
					with argscope([Conv2D], kernel_shape=3, nl=INReLU):	
						with argscope([Deconv2D], kernel_shape=3, strides=(2,2), nl=INReLU):
							# conv5_4 = Conv2D('conv5_4', input,   512)
							# conv5_3 = Conv2D('conv5_3', conv5_4, 512)
							# conv5_2 = Conv2D('conv5_2', conv5_3, 512)
							# conv5_1 = Conv2D('conv5_1', conv5_2, 512)
							# pool4 = Deconv2D('pool4',   input,   512)  # 8
							# conv4_4 = Conv2D('conv4_4', pool4,   512)
							# conv4_3 = Conv2D('conv4_3', conv4_4, 512)
							# conv4_2 = Conv2D('conv4_2', conv4_3, 512)
							# conv4_1 = Conv2D('conv4_1', conv4_2, 512)
							pool3 = Deconv2D('pool3',   source,  256)  # 16
							conv3_4 = Conv2D('conv3_4', pool3,   256)
							conv3_3 = Conv2D('conv3_3', conv3_4, 256)
							conv3_2 = Conv2D('conv3_2', conv3_3, 256)
							conv3_1 = Conv2D('conv3_1', conv3_2, 256)
							pool2 = Deconv2D('pool2',   conv3_1, 128)  # 32
							conv2_2 = Conv2D('conv2_2', pool2, 	 128)
							conv2_1 = Conv2D('conv2_1', conv2_2, 128)
							pool1 = Deconv2D('pool1',   conv2_1, 64)  # 64
							conv1_2 = Conv2D('conv1_2', pool1, 	 64)
							conv1_1 = Conv2D('conv1_1', conv1_2, 64)
							conv1_0 = Conv2D('conv1_0', conv1_1, 3, nl=tf.tanh)
							conv1_0 = tf_2tanh(conv1_0, maxVal=255.0)
							# conv1_0 = tf_2imag(conv1_0, maxVal=255.0)
							# conv1_0 = conv1_0 - VGG19_MEAN_TENSOR
							return conv1_0 # List of feature maps

		@auto_reuse_variable_scope				
		def vgg19_feature(source, name='VGG19_Feature'):
			with tf.variable_scope(name):
				with varreplace.freeze_variables():
					with argscope([Conv2D], kernel_shape=3, nl=tf.nn.relu):
						conv1_1 = Conv2D('conv1_1', source,  64)
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
						return conv4_1, [conv1_1, conv2_1, conv3_1, conv4_1]
						# return normalize(conv4_1), 
						# 	   [normalize(conv1_1), normalize(conv2_1), normalize(conv3_1), normalize(conv4_1)] # List of returned feature maps


		# Step 1: Run thru the encoder
		image_encoded, image_feature = vgg19_encoder(image)
		style_encoded, style_feature = vgg19_encoder(style)

		# Step 2: Run thru the adain block to get t=AdIN(f(c), f(s))
		merge_encoded = self._build_adain_layers(image_encoded, style_encoded)

		# Step 3: Run thru the decoder to get the paint image
		paint = vgg19_decoder(merge_encoded)

		# Actually, vgg19_feature and vgg19_encoder are identical
		# Splitting them to improve the programmability 
		paint_encoded, paint_feature = vgg19_feature(paint)		
		style_encoded, style_feature = vgg19_feature(style)

		# print(merge_encoded.get_shape())
		# print(paint_encoded.get_shape())
		#
		# Build losses here
		#
		with tf.name_scope('losses'):
			losses = []
			# Content loss between t and f(g(t))
			content_loss = self._build_content_loss(merge_encoded, paint_encoded, weight=args.weight_c)
			# add_moving_summary(content_loss)
			add_tensor_summary(content_loss, types=['scalar'], name='content_loss')
			losses.append(content_loss)

			# Style losses between paint and style
			style_losses = self._build_style_losses(paint_feature, style_feature, weight=args.weight_s)
			for idx, style_loss in enumerate(style_losses):
				add_tensor_summary(style_loss, types=['scalar'], name='style_loss')
				losses.append(style_loss)

			# Total variation loss
			smoothness = tf.reduce_sum(tf.image.total_variation(paint)) 
			add_tensor_summary(smoothness, types=['scalar'], name='smoothness')
			losses.append(smoothness*args.weight_tv)
			
			# Total loss
			self.cost = tf.reduce_sum(losses, name='self.cost') # this one goes to the optimizer
			add_tensor_summary(self.cost, types=['scalar'], name='self.cost')


		# Reconstruct img
		image = image + VGG19_MEAN_TENSOR
		style = style + VGG19_MEAN_TENSOR
		paint = tf.identity(paint + VGG19_MEAN_TENSOR, name='paint')
		# Build loss in here
		

		# Visualization
		viz = tf.concat([image, style, paint], axis=2)
		viz = tf.cast(tf.clip_by_value(viz, 0, 255), tf.uint8, name='viz')
		tf.summary.image('colorized', viz, max_outputs=50)

	def _get_optimizer(self):
		lr  = tf.get_variable('learning_rate', initializer=2e-5, trainable=False)
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
		images = glob.glob(self.image_path + '/*.jpg')
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
			rand_image = np.random.randint(0, len(images))
			rand_style = np.random.randint(0, len(styles))

			if self.isTrain:
				# Read image
				image = cv2.imread(images[rand_image], cv2.IMREAD_COLOR)
				image = self.central_crop(image)
				image = self.resize_image(image, size=256)
				image = self.random_crop (image, size=256)

				# Read style
				style = cv2.imread(styles[rand_style], cv2.IMREAD_COLOR)
				style = self.central_crop(style)
				style = self.resize_image(style, size=512)
				style = self.random_crop (style, size=256)
			else:
				pass

			yield [
				   image.astype(np.float32), 
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

	# ds_train = PrefetchDataZMQ(ds_train, 2)
	return ds_train, ds_valid

###################################################################################################
def render(model_path, volume_path, style_path):
	pass

###################################################################################################
if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--gpu', 		help='comma separated list of GPU(s) to use.')
	parser.add_argument('--load', 		help='load model')
	parser.add_argument('--vgg19', 		help='load model', 		   default='data/model/vgg19_weights_normalized.h5') #vgg19.npz')
	parser.add_argument('--image_path', help='path to the image.', default='data/train/image/') 
	parser.add_argument('--style_path', help='path to the style.', default='data/train/style/') 
	parser.add_argument('--alpha',      help='Between 0 and 1',    default=1.0,  type=float)
	parser.add_argument('--lambda',     help='Between 0 and 1',    default=1e-0, type=float)
	parser.add_argument('--weight_c',   help='Between 0 and 1',    default=1e-0, type=float)
	parser.add_argument('--weight_s',   help='Between 0 and 1',    default=1e-2, type=float)
	parser.add_argument('--weight_tv',  help='Between 0 and 1',    default=1e-5, type=float)
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
			# param_dict = dict(np.load(args.vgg19))
			# param_dict = {'VGG19/' + name: value for name, value in six.iteritems(param_dict)} 
			
			weight = h5py.File(args.vgg19, 'r')
			param_dict = {}
			param_dict.update({'VGG19_Encoder/' + name: value for name, value in six.iteritems( weight)})
			param_dict.update({'VGG19_Feature/' + name: value for name, value in six.iteritems( weight)})
			weight.close()

			# weight = dict(np.load(args.vgg19))
			# param_dict = {}
			# param_dict.update({'VGG19_Encoder/' + name: value for name, value in six.iteritems(weight)})
			# param_dict.update({'VGG19_Feature/' + name: value for name, value in six.iteritems(weight)})
			# print(param_dict)
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
