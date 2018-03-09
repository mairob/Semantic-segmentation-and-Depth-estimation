#! /usr/bin/python3
from __future__ import absolute_import
import os

#mute error messages
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import datetime as dt
import time
import sys
sys.path.append(r'Path/To/MyUtils_Folder')
sys.path.append(r'Path/To/tf_slim/slim')
slim = tf.contrib.slim

#TF 1.4+ needed
#Scripts are part of tf_slim
from nets.nasnet import nasnet 
from MyUtils import LayerBuildingBlocks as LayerBlocks 
from MyUtils import ImageProcessing as ImageProcessing
from MyUtils import Network as Network


import json
with open(r'Path\To\MyUtils\Global_Variables.json', encoding='utf-8') as data_file:
  meta = json.loads(data_file.read())


#Calculate training meta-data
STEPS_TO_RUN = int(((meta["Hyperparams"]["NR_OF_TR_IMAGES"] * meta["Hyperparams"]["EPOCHS_TO_RUN"] ) // (meta["Hyperparams"]["BATCH_SIZE"] * meta["Hyperparams"]["PSEUDO_BATCH_SIZE"])) + 1) #+1 to compensate fractual division
STEP_FOR_LOGGING = int(1 + (meta["Hyperparams"]["NR_OF_TR_IMAGES"] // (meta["Hyperparams"]["BATCH_SIZE"] * meta["Hyperparams"]["PSEUDO_BATCH_SIZE"])) * 0.05)  #logging  ca. each n-th percent of an epoch



def PrintMetaInformation():
	"""Print preamble with informations about the current run
  """
  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~') 
  print('~~~~~~~~~~ METAINFORMATION OF TESTRUN ~~~~~~~~~~~')
  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
  print('')
  print('Started training at :  %s ' %(time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime(time.time() + 3600 ))))
  print('Backbone-Model : %s ' %meta["Model"]["BACKBONE_NAME"])
  print('Logging directory : %s ' %meta["Logging"]["LOGGING_DIR"])
  print('Checkpoint directory :  %s ' %meta["Model"]["BACKBONE_CKPT"])
  print('Using tfRecord: %s ' %meta["Hyperparams"]["TF_RECORD_FILE"])
  print('Nr. of classes : %i ' %meta["Hyperparams"]["NR_OF_CLASSES"])
  print('Nr. of images in training-set : %i ' %meta["Hyperparams"]["NR_OF_TR_IMAGES"])
  print('Epochs to run : %i ' %meta["Hyperparams"]["EPOCHS_TO_RUN"])
  print('Images for one backprop : %i via BS %i and pseudo-BS %i ' %((meta["Hyperparams"]["BATCH_SIZE"]*meta["Hyperparams"]["PSEUDO_BATCH_SIZE"]), meta["Hyperparams"]["BATCH_SIZE"], meta["Hyperparams"]["PSEUDO_BATCH_SIZE"]))
  print('Training-steps to run : %i ' %STEPS_TO_RUN)
  print('Learning Rate : %s ' %meta["Hyperparams"]["LEARN_RATE"])
  print('Logging of Metainformation after %i steps ' %STEP_FOR_LOGGING)
  print('Saving Checkpoint after %i steps ' %meta["Logging"]["STEP_FOR_CKPTSAVE"])
  print('')
  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
  print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
  print('')

def run_training(isTrainingRun = True, TrainFromScratch = True):
	"""Builds and trains the entire network. Metainformation, saving of checkpoints and logging included
    
    Args:
      isTrainingRun: boolean, sets training flag for batch normalization
      TrainFromScratch: boolean, whether to use already trained FCN-checkpoint or to start from scratch with the backbone only
    """
  with tf.Graph().as_default():

    images_placeholder, depth_batch = Network.fill_queue_pipeline(meta["Hyperparams"]["TF_RECORD_FILE"])


    if TrainFromScratch:
      backbone_init_fn, depth_logits = Network.inference(images_placeholder, isTrainingRun = isTrainingRun, isTrainingFromScratch=TrainFromScratch)
    else:
      depth_logits = Network.inference(images_placeholder, isTrainingRun = isTrainingRun, isTrainingFromScratch=TrainFromScratch)

    
    loss = Network.loss_fn(depth_logits, depth_batch)
    zero_ops, accum_ops, update_ops, train_op = Network.training_agg(loss,  meta["Hyperparams"]["LEARN_RATE"])
    #zero_ops, accum_ops, update_ops, train_op = Network.training_single(loss,  meta["Hyperparams"]["LEARN_RATE"])
    

    orig_image_summary = tf.summary.image('Current_Image', images_placeholder)
    depth_batch_summary = tf.py_func(ImageProcessing.plot_depthmap, [depth_batch], tf.float32) 


    depth_logits_summary = tf.py_func(ImageProcessing.plot_depthmap, [depth_logits], tf.float32)
    Depth_summary = tf.summary.image('Depth', tf.concat(axis=2, values=[tf.expand_dims(depth_batch_summary, axis=0), tf.expand_dims(depth_logits_summary, axis=0)]), max_outputs=meta["Hyperparams"]["MAX_SUMMARY_IMAGES"]) 

    summary = tf.summary.merge_all()
    saver = tf.train.Saver(tf.global_variables(), max_to_keep = 3)
    sess = tf.Session()

    summary_writer = tf.summary.FileWriter(meta["Logging"]["LOGGING_DIR"], sess.graph)

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())

    if TrainFromScratch:
      backbone_init_fn(sess)
      print('>>> Everything initialized')
    else: 
      saver.restore(sess, meta["Model"]["TRAINED_CKPT"])
      print('>>> Session Restored....continue training')


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    print(">>> Queue initialized")


    epoch = 0
    last_loss = 1e-3
    last_miou = 1e-3

    start = time.time()
    start_of_run = start
    start_of_logging_interval = start

    #aggregation -- comment out if not used
    sess.run(zero_ops)

    print(">>> Start Trainig")
    for i in range(STEPS_TO_RUN):
      if (i+1) % STEP_FOR_LOGGING == 0: 
        #aggregation -- comment out if not used
        for j in range(meta["Hyperparams"]["PSEUDO_BATCH_SIZE"] ):
           sess.run([accum_ops])
        #<---

        combined_loss, summary_string, bn_up, _ = sess.run([loss, summary , update_ops, train_op])

        summary_writer.add_summary(summary_string, i+1) 
        summary_writer.flush()

        _images_left = (STEPS_TO_RUN - i)* meta["Hyperparams"]["BATCH_SIZE"] * meta["Hyperparams"]["PSEUDO_BATCH_SIZE"] 
        _current_image_time = (time.time() - start_of_logging_interval) / (STEP_FOR_LOGGING * meta["Hyperparams"]["BATCH_SIZE"] * meta["Hyperparams"]["PSEUDO_BATCH_SIZE"] )
        _time_2_bridge = _images_left * _current_image_time

        print('Estimated finishing time:  %s ' %(time.strftime("%d-%m-%Y %H:%M:%S", time.gmtime(3600 + time.time() + _time_2_bridge))))
        print("Step %i : Current total_loss of %.3f " %((i+1), combined_loss))
        print("Loss changed by %.2f percent" %(np.abs((1 - (combined_loss/last_loss)) * 100)))
        print("Processing one image took %.4f seconds (averaged) " %_current_image_time)
        print("Elapsed time since last logging %i seconds " %((time.time() - start))) 
        print("Specified trainig completed by %.1f percent" % (((i+1)/STEPS_TO_RUN)*100))
        print('')

        start_of_logging_interval = time.time()			
        last_loss = combined_loss

        #aggregation -- comment out if not used
        sess.run(zero_ops)

      else:
        #aggregation -- comment out if not used
        for j in range(meta["Hyperparams"]["PSEUDO_BATCH_SIZE"] ):
          sess.run([accum_ops])
        #<--

        sess.run([update_ops, train_op])

        #aggregation -- comment out if not used
        sess.run(zero_ops)

        
      if (i+1) % meta["Logging"]["STEP_FOR_CKPTSAVE"] == 0:
        saver.save(sess, meta["Model"]["SAVING_PATH"] +  meta["Model"]["BACKBONE_NAME"] +'.ckpt', global_step=i)
        print("##############################################")
        print("######### Logging after %i steps ###########" %STEP_FOR_LOGGING)
        print("##############################################")
        start = time.time()

    saver.save(sess, meta["Model"]["SAVING_PATH"] +  meta["Model"]["BACKBONE_NAME"] +'.ckpt', global_step=STEPS_TO_RUN)
    print(">>> Finished Trainig")

def run_validation():
"""Builds and evaluates model - predicted images are saved on disk. Checkpoint must be defined in "C:\...\MyUtils\Global_Variables.json"

  """
  import time
  with tf.Graph().as_default():

    filename_queue = tf.train.string_input_producer([r'Path/To/validation.tfrecords'], num_epochs=1)
    image, name = Network.read_and_decode_validation(filename_queue)

    image, name = Network.read_and_decode_validation(filename_queue)

    image = ImageProcessing.preprocessImage(image)
    image = tf.image.resize_images(image, [299, 299]) #size must match training input (299-IncRes / 224-NASnet)

    depth_logits = Network.inference(tf.expand_dims(image, dim=0), isTrainingRun = False, isTrainingFromScratch=False)
    predicted_image = tf.image.resize_images(depth_logits, [1024, 2048], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR) 

    image_encode = tf.image.encode_png(tf.cast(tf.squeeze(predicted_image, 0), tf.uint16))
    write_op = tf.write_file(name, image_encode)

    saver = tf.train.Saver(tf.global_variables(), max_to_keep = 3)
    sess = tf.Session()

    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, meta["Model"]["TRAINED_CKPT"])


    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
  
    cnt=0
    start= time.time()
    end =  start
    try:
      while True:
        sess.run(write_op) 
    except tf.errors.OutOfRangeError:
      end = time.time()
      coord.request_stop()
    finally:
      end= time.time()
      coord.request_stop()
      coord.join(threads)


    print(end - start)
    print(">>> Finished Evaluation")

	
	
### Example usage ####
# PrintMetaInformation()
# run_training(isTrainingRun = True,  TrainFromScratch= True)

# run_validation()


