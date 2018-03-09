#! /usr/bin/python3
#from __future__ import absolute_import
##############################################################################################
#################### example for network-file ################################################
#################### contains definition of whole graph ######################################
#################### losses and training-definition (optimizer, lrates, accumulation...) #####
#################### queue-manager to fill pipeline from tf-record ###########################
##############################################################################################

import tensorflow as tf
import numpy as np
import sys
sys.path.append(r'Path/To/MyUtils_Folder')
sys.path.append(r'Path/To/tf_slim/slim')
slim = tf.contrib.slim

# nets.nasnet import nasnet  #(located in tf_slim/slim starting at tf 1.4)
from nets import inception
from MyUtils import LayerBuildingBlocks as LayerBlocks 
from MyUtils import ImageProcessing as ImageProcessing

import json
with open(r'Path\To\MyUtils\Global_Variables.json', encoding='utf-8') as data_file:
  meta = json.loads(data_file.read())


checkpoint_file = meta["Model"]["BACKBONE_CKPT"]
NUMBER_OF_CLASSES =  meta["Hyperparams"]["NR_OF_CLASSES"]  
BATCH_SIZE = meta["Hyperparams"]["BATCH_SIZE"]

IMAGE_HEIGHT = meta["Hyperparams"]["INPUT_HEIGHT"]
IMAGE_WIDTH = meta["Hyperparams"]["INPUT_WIDTH"]
DOWNSAMPLING_FACTOR =  meta["Hyperparams"]["DOWNSAMPLING_FACTOR"]
RESIZE_HEIGHT = int(IMAGE_HEIGHT // DOWNSAMPLING_FACTOR)  
RESIZE_WIDTH = int(IMAGE_WIDTH // DOWNSAMPLING_FACTOR) 

CROPPING_FRACTION = meta["Hyperparams"]["CROPPING_FRACTION"]
MIN_DEQUEUE = meta["Hyperparams"]["MIN_DEQUEUE"]
QUEUE_CAPACITY = meta["Hyperparams"]["QUEUE_CAPACITY"]
QUEUE_THREADS = meta["Hyperparams"]["QUEUE_THREADS"]
MAX_SUMMARY_IMAGES = meta["Hyperparams"]["MAX_SUMMARY_IMAGES"]



def berHu_loss(y_true, y_pred):

    y_true = tf.cast(y_true, tf.float32) 
    y_pred = tf.cast(y_pred, tf.float32) 

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    err = tf.subtract(y_pred, y_true)
    abs_err = tf.abs(err)


    c = 0.2 * tf.reduce_max(abs_err)
    fraction =  tf.add(abs_err**2, c**2)
    fraction =  tf.divide(fraction, 2*c)


    return  tf.reduce_mean(tf.where(abs_err <= c, abs_err  , fraction ))  # if, then, else)

#ADAPT WITH DEFENITIONS OF THE "NETWORKDEFINITIONS"--FOLDER
def inference(images, isTrainingRun = True, isTrainingFromScratch=True):

  images = tf.to_float(images)
 

  with slim.arg_scope(inception.inception_resnet_v2_arg_scope()):
    probabilities, end_points = inception.inception_resnet_v2(images , 1001 , is_training=False)  #isTrainingRun

  if isTrainingFromScratch:
    variables_to_restore = slim.get_variables_to_restore(exclude=['InceptionResnetV2/AuxLogits', 'InceptionResnetV2/Logits'])
    backbone_init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)


  with tf.name_scope('COMMON_ENTRANCE'):
    with tf.variable_scope('COMMON_ENTRANCE'):

      CME_GCN_1 = LayerBlocks.global_conv_module(end_points['Conv2d_7b_1x1'], end_points['Conv2d_7b_1x1'].get_shape()[3], name='CME_GCN_1', k=7) 
      CME_Atrous_1 = LayerBlocks.atrousPyramid_small(end_points['Conv2d_7b_1x1'], name= "CME_Atrous_1", allow_r16=False, training=isTrainingRun) 

      Mixed_6a_conv = tf.contrib.layers.conv2d(end_points['Mixed_6a'], 164, (2,2), padding='VALID', activation_fn = tf.nn.relu)
      Mixed_6a_dil = tf.contrib.layers.conv2d(Mixed_6a_conv, 64, kernel_size=3, rate=2, padding='SAME')

      Mixed_5b_conv = tf.contrib.layers.conv2d(end_points['Mixed_5b'], 156, (4,4), padding='VALID', activation_fn = tf.nn.relu)
      Mixed_5b_dil = tf.contrib.layers.conv2d(Mixed_5b_conv, 64, kernel_size=3, rate=2, padding='SAME') 


      CME_1 =  tf.concat([CME_Atrous_1, CME_GCN_1], 3)
      CME_2 =  tf.concat([Mixed_6a_conv, Mixed_6a_dil], 3)
      CME_3 =  tf.concat([Mixed_5b_conv, Mixed_5b_dil], 3)

      print("end_points['Conv2d_7b_1x1']  : " + str(end_points['Conv2d_7b_1x1'].get_shape()[:]))
      print("end_points['Mixed_6a']  : " + str(end_points['Mixed_6a'].get_shape()[:]))
      print("end_points['Mixed_5b']  : " + str(end_points['Mixed_5b'].get_shape()[:]))
      print("CME_GCN_1  : " + str(CME_GCN_1.get_shape()[:]))
      print("CME_Atrous_1  : " + str(CME_Atrous_1.get_shape()[:]))
      print("Mixed_6a_conv  : " + str(Mixed_6a_conv.get_shape()[:]))
      print("Mixed_6a_dil  : " + str(Mixed_6a_dil.get_shape()[:]))
      print("Mixed_5b_conv : " + str(Mixed_5b_conv.get_shape()[:]))
      print("Mixed_5b_dil  : " + str(Mixed_5b_dil.get_shape()[:]))
      print("CME_1  : " + str(CME_1.get_shape()[:]))
      print("CME_2  : " + str(CME_2.get_shape()[:]))
      print("CME_3  : " + str(CME_3.get_shape()[:]))
      
      CME_1 = tf.layers.conv2d(CME_1, 1024 , (1,1), activation= tf.nn.relu, padding='VALID')
      print("CME_1 red. : " + str(CME_1.get_shape()[:]))

  with tf.name_scope('NEUTRAL_Branch'):
    with tf.variable_scope('NEUTRAL_Branch_1'):
      n_up_1 = LayerBlocks.up_project(CME_1, size=[3, 3, 1024, 96], id = 'n_2x_a', stride = 1, training=isTrainingRun)
      n_up_2 = LayerBlocks.up_project(n_up_1, size=[3, 3, 96, 64], id = 'n_2x_b', stride = 1, training=isTrainingRun)
      n_up_3 = LayerBlocks.up_project(n_up_2, size=[3, 3, 64, 32], id = 'n_2x_c', stride = 1, training=isTrainingRun)
      n_up_4 = LayerBlocks.up_project(n_up_3, size=[3, 3, 32, 8], id = 'n_2x_d', stride = 1, training=isTrainingRun) 
      n_up_5 = LayerBlocks.up_project(n_up_4, size=[3, 3, 8, 4], id = 'n_2x_e', stride = 1, training=isTrainingRun) 

      print("n_up_1  : " + str(n_up_1.get_shape()[:]))
      print("n_up_2 : " + str(n_up_2.get_shape()[:]))
      print("n_up_3 : " + str(n_up_3.get_shape()[:]))
      print("n_up_4 : " + str(n_up_4.get_shape()[:]))
      print("n_up_5 : " + str(n_up_5.get_shape()[:]))

    with tf.variable_scope('NEUTRAL_Branch_mid'):
      n_up_1_mid = LayerBlocks.up_project(CME_2, size=[3, 3, 164, 16], id = 'n_2x_a_mid', stride = 1, training=isTrainingRun)
      n_up_2_mid  = LayerBlocks.up_project(n_up_1_mid, size=[3, 3, 16, 16], id = 'n_2x_b_mid', stride = 1, training=isTrainingRun)
      n_up_3_mid  = LayerBlocks.up_project(n_up_2_mid, size=[3, 3, 16, 8], id = 'n_2x_c_mid', stride = 1, training=isTrainingRun)
      n_up_4_mid  = LayerBlocks.up_project(n_up_3_mid, size=[3, 3, 8, 4], id = 'n_2x_d_mid', stride = 1, training=isTrainingRun) 


      print("n_up_1_mid  : " + str(n_up_1_mid.get_shape()[:]))
      print("n_up_2_mid  : " + str(n_up_2_mid.get_shape()[:]))
      print("n_up_3_mid  : " + str(n_up_3_mid.get_shape()[:]))
      print("n_up_4_mid  : " + str(n_up_4_mid.get_shape()[:]))

    with tf.variable_scope('NEUTRAL_Branch_high'):
      n_up_1_high = LayerBlocks.up_project(CME_3, size=[3, 3, 156, 16], id = 'n_2x_a_high', stride = 1, training=isTrainingRun)
      n_up_2_high = LayerBlocks.up_project(n_up_1_high , size=[3, 3, 16, 8], id = 'n_2x_b_high', stride = 1, training=isTrainingRun)
      n_up_3_high = LayerBlocks.up_project(n_up_2_high , size=[3, 3, 8, 4], id = 'n_2x_c_high', stride = 1, training=isTrainingRun)


      print("n_up_1_high   : " + str(n_up_1_high .get_shape()[:]))
      print("n_up_2_high  : " + str(n_up_2_high .get_shape()[:]))
      print("n_up_3_high  : " + str(n_up_3_high .get_shape()[:]))




  with tf.name_scope('DEPTH_Branch'):
    with tf.variable_scope('DEPTH_Branch'):
      s_up_0 = LayerBlocks.boundary_refine(CME_1, name='BR_s_up_0', training=isTrainingRun)    
      s_up_0 = slim.convolution2d_transpose(s_up_0, 384, [3,3], [2,2], activation_fn=tf.nn.relu)
      sem_mix_0 = tf.concat([s_up_0, CME_2], 3) 

      s_up_1 = LayerBlocks.boundary_refine(sem_mix_0, name='BR_s_up_1', training=isTrainingRun)    
      s_up_1 = slim.convolution2d_transpose(s_up_1, 64, [3,3], [2,2], activation_fn=tf.nn.relu) 

      dil_s_up_1 = tf.contrib.layers.conv2d(s_up_1, 48, kernel_size=3, rate=2)
      sem_mix_1 = tf.concat([s_up_1, n_up_2, dil_s_up_1, CME_3], 3)

      s_up_2= LayerBlocks.boundary_refine(sem_mix_1, name='BR_s_up_2', training=isTrainingRun)
      s_up_2 = slim.convolution2d_transpose(s_up_2, 64, [3,3], [2,2], activation_fn=tf.nn.relu) 

      dil_s_up_2 = tf.contrib.layers.conv2d(s_up_2, 32, kernel_size=3, rate=4)
      sem_mix_2 = tf.concat([s_up_2, n_up_3, dil_s_up_2], 3)

      s_up_3 = LayerBlocks.boundary_refine(sem_mix_2, name='BR_s_up_3' ,training=isTrainingRun)
      s_up_3 = slim.convolution2d_transpose(s_up_3, 64, [3,3], [2,2], activation_fn=tf.nn.relu)

      dil_s_up_3 = tf.contrib.layers.conv2d(s_up_3, 32, kernel_size=3, rate=8)
      sem_mix_3 = tf.concat([s_up_3, n_up_4, dil_s_up_3], 3)


      s_up_4 = LayerBlocks.boundary_refine(sem_mix_3, name='BR_s_up_4', training=isTrainingRun)
      s_up_4 = slim.convolution2d_transpose(s_up_4, 64, [3,3], [2,2], activation_fn=tf.nn.relu) 

      dil_s_up_4 = tf.contrib.layers.conv2d(s_up_4, 32, kernel_size=3, rate=8)
      sem_mix_4 = tf.concat([s_up_4, n_up_5, n_up_4_mid, n_up_3_high, dil_s_up_4], 3)
      print("sem_mix_4  : " + str(sem_mix_4.get_shape()[:]))
       
      depth_endpoint = tf.layers.conv2d(sem_mix_4, 1 , (1,1), activation=tf.nn.relu, padding='VALID')
      depth_endpoint = tf.minimum(LayerBlocks.boundary_refine(depth_endpoint, name='depth_endpoint' ,training=isTrainingRun), 16896)


      print("s_up_1  : " + str(s_up_1.get_shape()[:]))
      print("s_up_2 : " + str(s_up_2.get_shape()[:]))
      print("s_up_3 : " + str(s_up_3.get_shape()[:]))
      print("s_up_4 : " + str(s_up_4.get_shape()[:]))
      print("depth_endpoint: " + str(depth_endpoint.get_shape()[:]))


  if isTrainingFromScratch:
    return backbone_init_fn, depth_endpoint

  return depth_endpoint


def loss_fn(depth_logits , depth_batch):
  """Calculates the loss from the logits and the labels.
  Args:
    logits: Logits Tensor 
    annotation_batch: Traind-ID's corrected label Tensor
  Returns:
    loss: Loss tensor of type float.
  """

  with tf.variable_scope("Loss"):
    inverse_Huber_loss = berHu_loss(depth_batch, depth_logits)
  
    reg_loss_col = tf.GraphKeys.REGULARIZATION_LOSSES
    weight_loss = tf.add_n(tf.get_collection(reg_loss_col),name='reg_loss')


    tf.summary.scalar('berHu', inverse_Huber_loss)
    tf.summary.scalar('weight_loss', weight_loss) 


  return  inverse_Huber_loss +  weight_loss


def training_agg(loss, learning_rate):
  """Sets up the training Ops.
  Creates a summarizer to track the loss over time in TensorBoard.
  Creates an optimizer and applies the gradients to all trainable variables.
  The Ops returned by this function is what must be passed to the
  `sess.run()` call to cause the model to train.
  Args:
    loss: Loss tensor, from loss().
    learning_rate: The learning rate to use for gradient descent.
  Returns:
    zero_ops: The Op for resetting the accumulated gradients.
    accum_ops: The Op for accumulating gradients of current forwardpass.
    train_op: The Op for training (backprop over accumulated grads).
  """

  with tf.variable_scope("Optimizer"):
    # Add a scalar summary for the snapshot loss.
    
    cur_step = tf.Variable(0, name='cur_step', trainable=False)
    starter_learning_rate = learning_rate
    end_learning_rate = 1e-6
    decay_steps = 81364 
    learning_rate = tf.train.polynomial_decay(starter_learning_rate, cur_step, decay_steps, end_learning_rate, power=0.9)
    
    tf.summary.scalar('Learning_rate', learning_rate)
    tf.summary.scalar('loss', loss)

    opt = tf.train.AdamOptimizer(learning_rate)

    #set up helper vars for gradient accumulation 
    divisor = tf.Variable(0, trainable=False)
    div_fl = tf.to_float(divisor)
    reset_divisor = divisor.assign(0)
    inc_divisor = divisor.assign(divisor+1)
    inc_cur_step = cur_step.assign(cur_step+1)

    #get all trainable vars and gradients
    #setup var for accumulating gradients over multiple forward passes
    t_vars = tf.trainable_variables()
    grads, graph_vars = zip(*opt.compute_gradients(loss, t_vars))
    grads_proc =[grad if grad is not None else tf.zeros_like(var)for var, grad in zip(t_vars, grads)]
    accum_grads = [tf.Variable(tf.zeros_like(t_var.initialized_value()), trainable=False) for t_var in t_vars]

    #helper for resetting the accumulation after a training-step
    with tf.control_dependencies([reset_divisor]):
      zero_ops = [tv.assign(tf.zeros_like(tv)) for tv in accum_grads]

    #helper for accumulating the grads per forward pass
    with tf.control_dependencies([inc_divisor]):
      accum_ops = [accum_grad.assign_add(grad) for (accum_grad, grad) in zip(accum_grads, grads_proc)]

    #average the accum. grads for one trainning setp
    normalised_accum_grads = [accum_grad/div_fl for (accum_grad) in accum_grads]

    #update_ops are needed for batch_norm
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      with tf.control_dependencies([inc_cur_step]):
        train_op = opt.apply_gradients(zip(normalised_accum_grads, graph_vars))

  return zero_ops, accum_ops, update_ops, train_op  


def training_single(loss, learning_rate):
  with tf.variable_scope("Optimizer"):
    # Add a scalar summary for the snapshot loss.
    tf.summary.scalar('loss', loss)

    global_step = tf.Variable(0, trainable=False)
    starter_learning_rate = learning_rate
    end_learning_rate = 1e-6
    decay_steps = 180000 
    learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step, decay_steps, end_learning_rate, power=0.9)
    
    tf.summary.scalar('Learning_rate', learning_rate)


    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(update_ops):
      train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

  return 0, 0, update_ops, train_op  



def fill_queue_pipeline(tf_record_set):
  with tf.variable_scope("Queue"):
    with tf.device("/cpu:0"):
      filename_queue = tf.train.string_input_producer([tf_record_set])
      image, annotation, depth = ImageProcessing.read_and_decode(filename_queue, hasDisparity=True, constHeight=IMAGE_HEIGHT, constWidth=IMAGE_WIDTH) 
      
      image = ImageProcessing.preprocessImage(image, CROPPING_FRACTION)


      print("image : ", str(image.get_shape()[:]))
      print("annotation : ", str(annotation.get_shape()[:]))

      image = tf.image.resize_images(image, [299, 299]) 
      depth = tf.image.resize_images(depth, [256, 256], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) #network output size needed for loss-function
      depth = tf.cast(depth, tf.int32) 

      image, depth = ImageProcessing.flip_randomly_left_right_image_with_annotation(image, depth)

      print("image : ", str(image.get_shape()[:]))
      print("depth : ", str(depth.get_shape()[:]))



  return tf.train.shuffle_batch([image, depth], batch_size=BATCH_SIZE, capacity=QUEUE_CAPACITY, num_threads=QUEUE_THREADS, min_after_dequeue=MIN_DEQUEUE )


  
  
def read_and_decode_validation(filename_queue):
    IMAGE_HEIGHT = 1024
    IMAGE_WIDTH = 2048
    reader = tf.TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = tf.parse_single_example(
        serialized_example,
        features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'name': tf.FixedLenFeature([], tf.string)
        })


    image = tf.decode_raw(features['image_raw'], tf.uint8)
    image_shape = tf.stack([IMAGE_HEIGHT, IMAGE_WIDTH, 3])
    image = tf.reshape(image, image_shape)
    return image, features['name']
