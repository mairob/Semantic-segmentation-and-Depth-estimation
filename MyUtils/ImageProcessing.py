#! /usr/bin/python3
#############################################################
### Helper File for TFRecords and Image manipulation ######## 
#############################################################
import tensorflow as tf
import numpy as np


## Label mapping for Cityscapes (34 classes)
Cityscapes34_ID_2_RGB = [(0,0,0), (0,0,0), (0,0,0), (0,0,0), (0,0,0) 
                # 0=unlabeled, ego vehicle,rectification border, oo roi, static
                ,(111,74,0),(81,0,81),(128,64,128),(244,35,232),(250,170,160)
                # 5=dynamic, 6=ground, 7=road, 8=sidewalk, 9=parking
                ,(230,150,140), (70,70,70), (102,102,156),(190,153,153),(180,165,180)
                # 10=rail track, 11=building, 12=wall, 13=fence, 14=guard rail
                ,(150,100,100),(150,120, 90),(153,153,153),(153,153,153),(250,170, 30)
                # 15= bridge, 16=tunnel, 17=pole, 18=polegroup, 19=traffic light
                ,(220,220,0),(107,142,35),(152,251,152),(70,130,180),(220,20,60)
                # 20=traffic sign 21=vegetation, 22=terrain, 23=sky, 24=person
                ,(255,0,0),(0,0,142),(0,0,70),(0,60,100),(0,0,90), (0,0,110), (0,80,100), (0,0,230),   (119, 11, 32)]
                # 25=rider, 26=car, 22=terrain, 27=truck, 28=bus, 29=caravan, 30= trailer, 31=train, 32=motorcyle ,33=bicycle

## Label mapping for Cityscapes (19 classes + '255'=wildcard)
Cityscapes20_ID_2_RGB = [(128,64,128),(244,35,232), (70,70,70), (102,102,156),(190,153,153)
                #0=road, 1=sidewalk, 2=building, 3=wall, 4=fence 
				,(153,153,153), (250,170, 30), (220,220,0),(107,142,35),(152,251,152),(70,130,180),(220,20,60)
                # 5= pole, 6=traffic light, 7= traffic sign, 8= vegetation,9= terrain, 10=sky, 11=person
                ,(255,0,0),(0,0,142),(0,0,70),(0,60,100), (0,80,100), (0,0,230), (119, 11, 32), (255,255,255)]
                # 12=rider, 13=car, 14=truck, 15=bus, 16=train, 17=motorcycle, 18=bicycle, #255 --cast via tf.minimum
				

Pred_2_ID = [7, 8, 11, 12, 13
                #0=road, 1=sidewalk, 2=building, 3=wall, 4=fence 
				,17 , 19, 20, 21, 22, 23, 24
                # 5= pole, 6=traffic light, 7= traffic sign, 8= vegetation,9= terrain, 10=sky, 11=person
                ,25 , 26, 27, 28, 31, 32, 33, -1]
                # 12=rider, 13=car, 14=truck, 15=bus, 16=train, 17=motorcycle, 18=bicycle, #255 --cast via tf.minimum
				


##################################################################################
################## Functions for Image Preprocessing #############################
##################################################################################

def read_and_decode(filename_queue, hasDisparity=False, constHeight=1024, constWidth=1024):
    """Decode images from TF-Records Bytestream. TF-Record must be compiled with the "make_tf_record.py"-script!
    
    Args:
    filename_queue: String representation of TF-Records (returned from tf.train.string_input_producer([TFRECORD_FILENAME])
	filename_queue: Boolean, needed for procession disparity maps
    constHeight, constWidth: Expected shapes of Images to decode
    Returns:
      Decoded image and mask 
    """

    with tf.name_scope("Input_Decoder"):
      reader = tf.TFRecordReader()
      _, serialized_example = reader.read(filename_queue)

      if not hasDisparity:
        features = tf.parse_single_example(
            serialized_example,
            features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string)
            })


        image = tf.decode_raw(features['image_raw'], tf.uint8)
        annotation = tf.decode_raw(features['mask_raw'], tf.uint8)

        image_shape = tf.stack([constHeight, constWidth, 3]) 
        annotation_shape = tf.stack([constHeight, constWidth, 1]) 

        image = tf.reshape(image, image_shape)
        annotation = tf.reshape(annotation, annotation_shape)

        return image, annotation
    
      else:
        features = tf.parse_single_example(
          serialized_example,
          features={
            'image_raw': tf.FixedLenFeature([], tf.string),
            'mask_raw': tf.FixedLenFeature([], tf.string),
            'disp_raw': tf.FixedLenFeature([], tf.string)
            })

        image = tf.decode_raw(features['image_raw'], tf.uint8)
        annotation = tf.decode_raw(features['mask_raw'], tf.uint8)
        disparity = tf.decode_raw(features['disp_raw'], tf.int16) #uint6

        image_shape = tf.stack([constHeight, constWidth, 3]) 
        masks_shape = tf.stack([constHeight, constWidth, 1])

        image = tf.reshape(image, image_shape)
        annotation = tf.reshape(annotation, masks_shape)
        disparity = tf.reshape(disparity, masks_shape)

      return image, annotation, disparity
    
def decode_labels(mask, num_images=1, num_classes=20, label=Cityscapes20_ID_2_RGB):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
	  label: List, which value to assign for different classes
    
    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    from PIL import Image
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w, 3), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('RGB', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label[k]
      outputs[i] = np.array(img)
    return outputs

def apply_with_random_selector(x, func, num_cases):
  from tensorflow.python.ops import control_flow_ops

  with tf.name_scope("Random_Selector"):
    sel = tf.random_uniform([], maxval=num_cases, dtype=tf.int32)
    # Pass the real x only to one of the func calls.
    return control_flow_ops.merge([
        func(control_flow_ops.switch(x, tf.equal(sel, case))[1], case)
        for case in range(num_cases)])[0]

def distort_color(image, color_ordering=0, fast_mode=True, scope=None):

  with tf.name_scope("Color_distortion"):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

  # The random_* ops do not necessarily clamp.
  return tf.clip_by_value(image, 0.0, 1.0)

from tensorflow.python.ops import control_flow_ops
def flip_randomly_left_right_image_with_annotation(image_tensor, annotation_tensor):
  """Flips an image randomly and applies the same to an annotation tensor.

  Args:
    image_tensor, annotation_tensor: 3-D-Tensors
  Returns:
    Flipped image and gt.
  """
    random_var = tf.random_uniform(maxval=2, dtype=tf.int32, shape=[])


    randomly_flipped_img = control_flow_ops.cond(pred=tf.equal(random_var, 0),
                                                 fn1=lambda: tf.image.flip_left_right(image_tensor),
                                                 fn2=lambda: image_tensor)

    randomly_flipped_annotation = control_flow_ops.cond(pred=tf.equal(random_var, 0),
                                                        fn1=lambda: tf.image.flip_left_right(annotation_tensor),
                                                        fn2=lambda: annotation_tensor)
    
    return randomly_flipped_img, randomly_flipped_annotation
  
  
def random_crop_and_pad_image_and_labels(image, sem_labels, dep_labels, size):
  """Randomly crops `image` together with `labels`.

  Args:
    image: A Tensor with shape [D_1, ..., D_K, N]
    labels: A Tensor with shape [D_1, ..., D_K, M]
    size: A Tensor with shape [K] indicating the crop size.
  Returns:
    A tuple of (cropped_image, cropped_label).
  """
  combined = tf.concat([image, sem_labels, dep_labels], axis=2)

  print("combined : ", str(combined.get_shape()[:]))

  combined_crop = tf.random_crop(combined, [size[0], size[1],5])

  print("combined_crop : ", str(combined_crop.get_shape()[:]))

  channels = tf.unstack(combined_crop, axis=-1)
  image =  tf.stack([channels[0],channels[1],channels[2]], axis=-1)
  sem_label = tf.expand_dims(channels[3], axis=2)
  dep_label = tf.expand_dims(channels[4], axis=2)

  return image, sem_label, dep_label

def preprocessImage(image, central_crop_fraction= 0.875):

  with tf.name_scope("Preprocessing"):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)

    distorted_image = apply_with_random_selector( image, lambda x, ordering: distort_color(x, ordering, fast_mode=True),num_cases=4)
    image = tf.subtract(distorted_image, 0.5)
    image = tf.multiply(image, 2.0)
    return image


##################################################################################
################## Functions for Image Postprocessing #############################
##################################################################################

def generate_prediction_Img(mask, num_images=1, num_classes= 20, label=Pred_2_ID):
    """Decode batch of segmentation masks.
    
    Args:
      mask: result of inference after taking argmax.
      num_images: number of images to decode from the batch.
      num_classes: number of classes to predict (including background).
	  label: List, which value to assign for different classes

    Returns:
      A batch with num_images RGB images of the same size as the input. 
    """
    from PIL import Image
    n, h, w, c = mask.shape
    assert(n >= num_images), 'Batch size %d should be greater or equal than number of images to save %d.' % (n, num_images)
    outputs = np.zeros((num_images, h, w), dtype=np.uint8)
    for i in range(num_images):
      img = Image.new('L', (len(mask[i, 0]), len(mask[i])))
      pixels = img.load()
      for j_, j in enumerate(mask[i, :, :, 0]):
          for k_, k in enumerate(j):
              if k < num_classes:
                  pixels[k_,j_] = label[k]
      outputs[i] = np.array(img)
    return outputs
    
def plot_depthmap(mask):
    """Network output as [w, h, 1]-Tensor is transformed to a heatmap for easier visual interpretation
    
    Args:
      mask: result of inference (depth = 1)
    Returns:
      A RGB-Image (representation of the depth prediction as heatmap 
    """
  import matplotlib.pyplot as plt

  cmap = plt.get_cmap('hot')
  gray = mask[0,:,:,0].astype(np.uint16)
  divisor = np.max(gray) - np.min(gray)

  if divisor != 0:
    normed = (gray - np.min(gray)) / divisor
  else:
    normed = (gray - np.min(gray))

  rgba_img = cmap(normed)
  rgb_img = np.delete(rgba_img, 3,2)

  return (65535 * rgb_img).astype(np.float32)