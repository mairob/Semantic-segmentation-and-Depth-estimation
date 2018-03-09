#! /usr/bin/python3
#############################################################
### Helper File for Fully Conv. Building Blocks ############# 
#############################################################
import tensorflow as tf
import numpy as np
import sys
sys.path.append(r'Path/To/tf_slim/slim')
slim = tf.contrib.slim
L2_REG_SCALE = 5e-5



def global_conv_module(input_data, n_filters,  name, k=9, padding='same'):
	"""Global convolution network [https://arxiv.org/abs/1703.02719]
    
    Args:
      input_data: tensor, incoming feautre maps to process
      n_filters: nnumber of output feature maps
      name: string, name for graph node
	  k: size of convolution kernel. Should be around size  of incoming feature maps
	  padding: strategy for convolution padding

    Returns:
      Convolved feature maps. 
    """

    with tf.name_scope("GCM"):
	    with tf.variable_scope("global_conv_module_{}".format(name)):
	        branch_a = tf.layers.conv2d(input_data, n_filters, (k, 1), activation=None, kernel_regularizer=slim.l2_regularizer(L2_REG_SCALE),padding=padding, name='conv_1a')
	        branch_a = tf.layers.conv2d(branch_a, n_filters, (1, k), activation=None, kernel_regularizer=slim.l2_regularizer(L2_REG_SCALE), padding=padding, name='conv_2a')

	        branch_b = tf.layers.conv2d(input_data, n_filters, (1, k), activation=None, kernel_regularizer=slim.l2_regularizer(L2_REG_SCALE),padding=padding, name='conv_1b')
	        branch_b = tf.layers.conv2d(branch_b, n_filters, (k, 1), activation=None, kernel_regularizer=slim.l2_regularizer(L2_REG_SCALE),padding=padding, name='conv_2b')

	        output_data = tf.add(branch_a, branch_b, name='sum')

    return tf.nn.relu(output_data)

def boundary_refine(input_data, name, training=True):
	"""Boundary refinement network [https://arxiv.org/abs/1703.02719]
    
    Args:
      input_data: tensor, incoming feautre maps to process
      name: string, name for graph node
	  training: boolean, sets training mode for batch normalization

    Returns:
      Convolved feature maps. 
    """

	with tf.name_scope("BR"):
		with tf.variable_scope("boundary_refine_module_{}".format(name)):

			n_filters = input_data.get_shape()[3].value

			output_data	 = tf.layers.conv2d(input_data, n_filters, (3, 3), activation=None, kernel_regularizer=slim.l2_regularizer(L2_REG_SCALE), padding='SAME', name='conv_1')
			output_data = tf.layers.batch_normalization(output_data, training=training, name='bn_1', renorm=True)
			output_data = tf.nn.relu(output_data, name='relu_1')

			output_data = tf.layers.conv2d(output_data, n_filters, (3, 3), activation=None, kernel_regularizer=slim.l2_regularizer(L2_REG_SCALE), padding='SAME', name='conv_2')
			output_data = tf.add(output_data, input_data, name='sum')

	return tf.nn.relu(output_data)

def Transpose(input_data, outShape, name, FilterKernelSize=3 , stride = [1, 2, 2, 1]):
	"""Conveniance wrapper for transpose convolution.
    
    Args:
      input_data: tensor, incoming feautre maps to process
      outShape: list, shape of output
	  name: string, name for graph node
	  FilterKernelSize: size of convolution kernel
	  stride: list. Striding strategy for convolution

    Returns:
      Convolved feature maps. 
    """

	with tf.name_scope("Transpose"):
		with tf.variable_scope("transpose_{}".format(name)):
			inputChannels = input_data.get_shape()[3]
			filterShape = [FilterKernelSize, FilterKernelSize , outShape[3], inputChannels]
			W = tf.get_variable(name + "_Weight", shape= filterShape, initializer= tf.contrib.layers.xavier_initializer())
			up = tf.nn.conv2d_transpose(input_data, W , outShape, strides=stride, padding="SAME")

	return tf.nn.relu(up)
	

def atrousPyramid_small(input_data, name, allow_r16=False, training=True, power2output=True):
	"""inspired by Spatial Pyramid Pooling [arXiv:1706.05587v2]. Channel reduction + different atrous convolutions to enlarge effective receptive field
    
    Args:
      input_data: tensor, incoming feautre maps to process
	  name: string, name for graph node
	  allow_r16: boolean, allow additional convolution with dilation rate r=16
	  training: boolean, sets training mode for batch normalization
	  power2output: boolean, whether or not to reduce output to power of 2. e.g: Input=1536 f.maps -> Output: 1024/2 f.maps.

    Returns:
      Convolved feature maps. 
    """
	
	with tf.name_scope("Atrous_Pyramid"):
		with tf.variable_scope("atrous_small_{}".format(name)):
			inputChannels = input_data.get_shape()[3] 

			outputChannels = inputChannels
			#power of 2
			if power2output:
				outputChannels = 1<<(int(inputChannels)-1).bit_length()
				if outputChannels != 0:
						outputChannels /= 4
				else:
					outputChannels += 2 
			
			
			#reduce input for follow-up atrous convs.
			#to minimize memory footprint
			input_red = tf.layers.conv2d(input_data, outputChannels, (1,1), activation=None,  kernel_regularizer=slim.l2_regularizer(L2_REG_SCALE), padding='SAME')

			filterShape = [3,3, input_red.get_shape()[3] , outputChannels]
			W2 = tf.get_variable(name+"_W_2", shape=filterShape,initializer=tf.contrib.layers.xavier_initializer())
			W4 = tf.get_variable(name+"_W_4", shape=filterShape,initializer=tf.contrib.layers.xavier_initializer())
			W8 = tf.get_variable(name+"_W_8", shape=filterShape,initializer=tf.contrib.layers.xavier_initializer())

			ac2 = tf.nn.atrous_conv2d(input_red, W2, rate=2, padding="SAME")
			ac4 = tf.nn.atrous_conv2d(input_red, W4, rate=4, padding="SAME")
			ac8 = tf.nn.atrous_conv2d(input_red, W8, rate=8, padding="SAME")


			if allow_r16:
				W16 = tf.get_variable(name+"_W_16", shape=filterShape,initializer=tf.contrib.layers.xavier_initializer())
				ac16 = tf.nn.atrous_conv2d(input_red, W16, rate=16, padding="SAME")
				
				ccat =  tf.concat([input_red, ac2, ac4, ac8, ac16,], 3)

			else:
				ccat =  tf.concat([input_red, ac2, ac4, ac8], 3)

			output_data = tf.layers.conv2d(ccat, outputChannels, (1,1), activation=None,  kernel_regularizer=slim.l2_regularizer(L2_REG_SCALE), padding='SAME')
			output_data = tf.layers.batch_normalization(output_data, training=training, renorm=True)

			return tf.nn.relu(output_data)


def get_incoming_shape(incoming):
	"""Wrapper to determin shape of a tensor
    Returns:
      Shape of  tensor
    """


	if isinstance(incoming, tf.Tensor):
		return incoming.get_shape().as_list()
	elif type(incoming) in [np.array, list, tuple]:
		return np.shape(incoming)
	else:
		raise Exception("Invalid incoming layer.")


def interleave(tensors, axis):
	"""https://arxiv.org/abs/1606.00373
    Returns:
      interleaved tensor
    """
	old_shape = get_incoming_shape(tensors[0])[1:]
	new_shape = [-1] + old_shape
	new_shape[axis] *= len(tensors)
	return tf.reshape(tf.stack(tensors, axis + 1), new_shape)

		
def unpool_as_conv(size, input_data, id, stride = 1, ReLU = False, training=True):
	"""https://arxiv.org/abs/1606.00373
		# Model upconvolutions (unpooling + convolution) as interleaving feature
		# maps of four convolutions (A,B,C,D). Building block for up-projections. 
    """


	with tf.name_scope("Unpool"):
		with tf.variable_scope("unpool_{}".format(id)):
			# Convolution A (3x3)
			# --------------------------------------------------
			layerName = "layer%s_ConvA" % (id)
			outputA = tf.contrib.layers.conv2d(input_data, size[3], (3,3), (stride, stride), weights_regularizer=slim.l2_regularizer(L2_REG_SCALE),  padding='SAME', activation_fn = None)


			# Convolution B (2x3)
			# --------------------------------------------------
			layerName = "layer%s_ConvB" % (id)
			padded_input_B = tf.pad(input_data, [[0, 0], [1, 0], [1, 1], [0, 0]], "CONSTANT")
			outputB = tf.contrib.layers.conv2d(padded_input_B, size[3], (2, 3), (stride, stride),  weights_regularizer=slim.l2_regularizer(L2_REG_SCALE), padding='VALID', activation_fn = None)
			
			

			# Convolution C (3x2)
			# --------------------------------------------------
			layerName = "layer%s_ConvC" % (id)
			padded_input_C = tf.pad(input_data, [[0, 0], [1, 1], [1, 0], [0, 0]], "CONSTANT")
			outputC = tf.contrib.layers.conv2d(padded_input_C, size[3], (3, 2), (stride, stride), weights_regularizer=slim.l2_regularizer(L2_REG_SCALE),  padding='VALID', activation_fn = None)


			# Convolution D (2x2)
			# --------------------------------------------------
			layerName = "layer%s_ConvD" % (id)
			padded_input_D = tf.pad(input_data, [[0, 0], [1, 0], [1, 0], [0, 0]], "CONSTANT")
			outputD = tf.contrib.layers.conv2d(padded_input_D, size[3], (2, 2), (stride, stride), weights_regularizer=slim.l2_regularizer(L2_REG_SCALE), padding='VALID', activation_fn = None)


			# Interleaving elements of the four feature maps
			# --------------------------------------------------
			left = interleave([outputA, outputB], axis=1)  # columns
			right = interleave([outputC, outputD], axis=1)  # columns
			Y = interleave([left, right], axis=2) # rows

			#layerName = "layer%s_BN" % (id)
			Y = tf.layers.batch_normalization(Y, training=training,  renorm=True)

			if ReLU:
				Y = tf.nn.relu(Y, name = layerName)

	return Y
	
def up_project(input_data, size, id, stride = 1, training=True):
	"""https://arxiv.org/abs/1606.00373
		# Create residual upsampling layer (UpProjection)
    """
	with tf.name_scope("up_projection"):
		with tf.variable_scope("up_projection_{}".format(id)):
			# Branch 1
			id_br1 = "%s_br1" % (id)

			# Interleaving Convs of 1st branch
			branch1_output = unpool_as_conv(size, input_data, id_br1, stride, ReLU=True, training=True)

			# Convolution following the upProjection on the 1st branch
			layerName = "layer%s_Conv" % (id)
			branch1_output = tf.contrib.layers.conv2d(branch1_output, size[3], (size[0],size[1]), (stride, stride), weights_regularizer=slim.l2_regularizer(L2_REG_SCALE))


			layerName = "layer%s_BN" % (id)
			branch1_output = tf.layers.batch_normalization(branch1_output, training=training, renorm=True)


			# Branch 2
			id_br2 = "%s_br2" % (id)
			# Interleaving convolutions and output of 2nd branch
			branch2_output = unpool_as_conv(size, input_data, id_br2, stride, ReLU=False)


			# sum branches
			layerName = "layer%s_Sum" % (id)
			output = tf.add_n([branch1_output, branch2_output], name = layerName)
			# ReLU
			layerName = "layer%s_ReLU" % (id)
			output = tf.nn.relu(output, name=layerName)
		
	return output


