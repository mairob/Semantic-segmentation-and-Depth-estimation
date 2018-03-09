#! /usr/bin/python3
#############################################################
### Helper File to generate TFRecords ####################### 
#############################################################
import os

import numpy as np
import tensorflow as tf
import skimage.io as io
from PIL import Image
import cv2 #for disparity images

IMAGE_HEIGHT = 1024
IMAGE_WIDTH = 2048


#resize images if you want to save storage
RESIZE_HEIGHT = 512
RESIZE_WIDTH = 1024


def loadImages(imgDir, annoDir, dispDir=None):
    """Get all image paths from Cityscapes/.../images  
    
    Args:
    imgDir, annoDir, dispDir: Folderpath of "leftImg8bit", "gt...", and "disparity" folders
    Returns:
      Paths to all images contained within the  folders 
    """

    files = [os.path.relpath(os.path.join(dirpath, file), annoDir) for (dirpath, dirnames, filenames) in
             os.walk(annoDir) for file in filenames]
    Exclude = ['color', 'instance', 'label', 'polygon']
    filtered = [path for path in files if not any(unwanted in path for unwanted in Exclude)]

    if dispDir:
        new_Images_dep = [path.replace('_gtFine_TrainIds.png', '_disparity.png') for path in filtered]
        disparityPaths = [os.path.join(dispDir, path) for path in new_Images_dep]
   
    new_Images_seg = [path.replace('_gtFine_TrainIds.png', '_leftImg8bit.png') for path in filtered]
    annotationPaths = [os.path.join(annoDir, path) for path in filtered]
    imagePaths = [os.path.join(imgDir, path) for path in new_Images_seg]

    if dispDir:
        return imagePaths, annotationPaths, disparityPaths

    return imagePaths, annotationPaths


def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def _floats_feature(value):
  return tf.train.Feature(float_list=tf.train.FloatList (value=[value]))


def createRecords(tfrecords_filename, filename_pairs, containsDisp=False):
    """Create binary representation of training set (TFRecord)  
    
    Args:
    tfrecords_filename: path to tfrecord (or just the filename 'myRecord.tfrecords')
	filename_pairs: zipped list of image-paths to form dataset (returned from loadImages + zipped)
	containsDisp: boolean, whether or not disparity maps should be included into Record
    """

    writer = tf.python_io.TFRecordWriter(tfrecords_filename)
    cnt = 0
    if containsDisp:
        for img_path, annotation_path, disparity_path in filename_pairs:

			#resize needs to be nearest neighbor -> PIL default
			img = np.array(Image.open(img_path).resize((RESIZE_WIDTH, RESIZE_HEIGHT)))
			annotation = np.array(Image.open(annotation_path).resize((RESIZE_WIDTH, RESIZE_HEIGHT)))
			disparity = np.array(cv2.resize(cv2.imread(disparity_path,-1), (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_NEAREST))# second argument -1 == cv2.IMREAD_UNCHANGED

			
			if np.size(annotation) > (RESIZE_HEIGHT*RESIZE_WIDTH) or np.size(img) > (RESIZE_HEIGHT*RESIZE_WIDTH*3) or np.size(disparity) > (RESIZE_HEIGHT*RESIZE_WIDTH*1):
				print(img_path)
				print(annotation_path)
				print(disparity_path)
			else:
				img_raw = img.tostring()
				annotation_raw = annotation.tostring()
				disparity_raw = disparity.tostring()
				example = tf.train.Example(features=tf.train.Features(feature={
					'image_raw': _bytes_feature(img_raw),
					'mask_raw': _bytes_feature(annotation_raw),
					'disp_raw': _bytes_feature(disparity_raw)}))

				writer.write(example.SerializeToString())

    else:
        for img_path, annotation_path in filename_pairs:
			#resize needs to be nearest neighbor -> PIL default
			img = np.array(Image.open(img_path).resize((RESIZE_WIDTH, RESIZE_HEIGHT)))
			annotation = np.array(Image.open(annotation_path).resize((RESIZE_WIDTH, RESIZE_HEIGHT)))
			
			if np.size(annotation) > (RESIZE_HEIGHT*RESIZE_WIDTH) or np.size(img) > (RESIZE_HEIGHT*RESIZE_WIDTH*3):
				print(annotation_path)
				print(img_path)
			else:
				img_raw = img.tostring()
				annotation_raw = annotation.tostring()
				example = tf.train.Example(features=tf.train.Features(feature={
					'image_raw': _bytes_feature(img_raw),
					'mask_raw': _bytes_feature(annotation_raw)}))

				writer.write(example.SerializeToString())
            print(cnt)
            cnt += 1

    writer.close()



	
#######################################################
############### example usage #########################
#######################################################	
	
# #Set path to image folders	
# annotationsDir = r'C:\...\Cityscapes_fine\gtFine\train'
# imagesDir = r'C:\...\Cityscapes_fine\leftImg8bit\train'
# disparityDir = r'C:\...\Cityscapes_fine\disparity\train'

# #set name and location for tfrecord that will be created
# tfrecords_filename = 'City_TrainIDs_Fine_Train.tfrecords'

# #get all image-paths
# images, annotations, disparity = loadImages(imagesDir, annotationsDir, disparityDir)

# #create record
# merged = zip(images, annotations, disparity)
# createRecords(tfrecords_filename, merged, containsDisp=True)