def inference(images, isTrainingRun = True, isTrainingFromScratch=True):

  images = tf.to_float(images)
 

  with slim.arg_scope(nasnet.nasnet_mobile_arg_scope()):
    probabilities, end_points  = nasnet.build_nasnet_mobile(images, num_classes=1001, is_training=False)

  if isTrainingFromScratch:
    variables_to_restore = slim.get_variables_to_restore(exclude=['build_nasnet_mobile/AuxLogits', 'build_nasnet_mobile/Logits'])
    backbone_init_fn = slim.assign_from_checkpoint_fn(checkpoint_file, variables_to_restore)


  with tf.name_scope('COMMON_ENTRANCE'):
    with tf.variable_scope('COMMON_ENTRANCE'):
      CME_GCN_1 = LayerBlocks.global_conv_module(end_points['Cell_11'], end_points['Cell_11'].get_shape()[3], name='CME_GCN_1', k=5) 
      CME_Atrous_1 = LayerBlocks.atrousPyramid_small(end_points['Cell_11'], name= "CME_Atrous_1", allow_r16=False, training=isTrainingRun) 
     
      Mid_conv = tf.contrib.layers.conv2d(end_points['Cell_10'], 192, (2,2), padding='SAME', activation_fn = tf.nn.relu)
      Mid_dil = tf.contrib.layers.conv2d(Mid_conv, 64, kernel_size=3, rate=2, padding='SAME')

      High_conv = tf.contrib.layers.conv2d(end_points['Cell_9'], 192, (2,2), padding='SAME', activation_fn = tf.nn.relu)
      High_dil = tf.contrib.layers.conv2d(High_conv, 64, kernel_size=3, rate=2, padding='SAME') 


      CME_1 =  tf.concat([CME_Atrous_1, CME_GCN_1], 3)
      CME_2 =  tf.concat([Mid_conv, Mid_dil], 3)
      CME_3 =  tf.concat([High_conv, High_dil], 3)

      print("end_points['Cell_11']  : " + str(end_points['Cell_11'].get_shape()[:]))
      print("end_points['Cell_10']  : " + str(end_points['Cell_10'].get_shape()[:]))
      print("end_points['Cell_9']  : " + str(end_points['Cell_9'].get_shape()[:]))
      print("CME_GCN_1  : " + str(CME_GCN_1.get_shape()[:]))
      print("CME_Atrous_1  : " + str(CME_Atrous_1.get_shape()[:]))
      print("Mid_conv  : " + str(Mid_conv.get_shape()[:]))
      print("Mid_dil  : " + str(Mid_dil.get_shape()[:]))
      print("High_conv  : " + str(High_conv.get_shape()[:]))
      print("High_dil  : " + str(High_dil.get_shape()[:]))
      print("CME_1 : " + str(CME_1.get_shape()[:]))
      print("CME_2 : " + str(CME_2.get_shape()[:]))
      print("CME_3 : " + str(CME_3.get_shape()[:]))
	  
      CME = tf.concat([CME_1, CME_2, CME_3], 3)
      CME = tf.layers.conv2d(CME, 1024 , (1,1), activation= tf.nn.relu, padding='VALID')

  with tf.name_scope('NEUTRAL_Branch'):
    with tf.variable_scope('NEUTRAL_Branch'):
      n_up_1 = LayerBlocks.up_project(CME, size=[3, 3, 1024, 128], id = 'n_2x_a', stride = 1, training=isTrainingRun)
      n_up_2 = LayerBlocks.up_project(n_up_1, size=[3, 3, 128, 128], id = 'n_2x_b', stride = 1, training=isTrainingRun)
      n_up_3 = LayerBlocks.up_project(n_up_2, size=[3, 3, 128, 64], id = 'n_2x_c', stride = 1, training=isTrainingRun)
      n_up_4 = LayerBlocks.up_project(n_up_3, size=[3, 3, 64, 64], id = 'n_2x_d', stride = 1, training=isTrainingRun) 
      n_up_5 = LayerBlocks.up_project(n_up_4, size=[3, 3, 64, 16], id = 'n_2x_e', stride = 1, training=isTrainingRun) #16

      print("n_up_1  : " + str(n_up_1.get_shape()[:]))
      print("n_up_2 : " + str(n_up_2.get_shape()[:]))
      print("n_up_3 : " + str(n_up_3.get_shape()[:]))
      print("n_up_4 : " + str(n_up_4.get_shape()[:]))
      print("n_up_5 : " + str(n_up_5.get_shape()[:]))

    with tf.variable_scope('NEUTRAL_Branch_mid'):
      n_up_1_mid = LayerBlocks.up_project(CME_2, size=[3, 3, 256, 128], id = 'n_2x_a_mid', stride = 1, training=isTrainingRun)
      n_up_2_mid  = LayerBlocks.up_project(n_up_1_mid, size=[3, 3, 128, 64], id = 'n_2x_b_mid', stride = 1, training=isTrainingRun)
      n_up_3_mid  = LayerBlocks.up_project(n_up_2_mid, size=[3, 3, 64, 48], id = 'n_2x_c_mid', stride = 1, training=isTrainingRun)
      n_up_4_mid  = LayerBlocks.up_project(n_up_3_mid, size=[3, 3, 48, 16], id = 'n_2x_d_mid', stride = 1, training=isTrainingRun)
      n_up_5_mid  = LayerBlocks.up_project(n_up_4_mid, size=[3, 3, 16, 16], id = 'n_2x_e_mid', stride = 1, training=isTrainingRun)  


      print("n_up_1_mid  : " + str(n_up_1_mid.get_shape()[:]))
      print("n_up_2_mid  : " + str(n_up_2_mid.get_shape()[:]))
      print("n_up_3_mid  : " + str(n_up_3_mid.get_shape()[:]))
      print("n_up_4_mid  : " + str(n_up_4_mid.get_shape()[:]))
      print("n_up_5_mid  : " + str(n_up_5_mid.get_shape()[:]))

    with tf.variable_scope('NEUTRAL_Branch_high'):
      n_up_1_high = LayerBlocks.up_project(CME_3, size=[3, 3, 256, 128], id = 'n_2x_a_high', stride = 1, training=isTrainingRun)
      n_up_2_high = LayerBlocks.up_project(n_up_1_high , size=[3, 3, 128, 48], id = 'n_2x_b_high', stride = 1, training=isTrainingRun)
      n_up_3_high = LayerBlocks.up_project(n_up_2_high , size=[3, 3, 48, 32], id = 'n_2x_c_high', stride = 1, training=isTrainingRun)
      n_up_4_high = LayerBlocks.up_project(n_up_3_high , size=[3, 3, 32, 16], id = 'n_2x_d_high', stride = 1, training=isTrainingRun)
      n_up_5_high = LayerBlocks.up_project(n_up_4_high , size=[3, 3, 16, 16], id = 'n_2x_e_high', stride = 1, training=isTrainingRun)


      print("n_up_1_high   : " + str(n_up_1_high .get_shape()[:]))
      print("n_up_2_high  : " + str(n_up_2_high .get_shape()[:]))
      print("n_up_3_high  : " + str(n_up_3_high .get_shape()[:]))
      print("n_up_4_high  : " + str(n_up_4_high .get_shape()[:]))
      print("n_up_5_high  : " + str(n_up_5_high .get_shape()[:]))


  with tf.name_scope('DEPTH_Branch'):
    with tf.variable_scope('DEPTH_Branch'):
      s_up_1 = LayerBlocks.boundary_refine(CME, name='BR_d_up_1', training=isTrainingRun)    
      s_up_1 = slim.convolution2d_transpose(s_up_1, 96, [3,3], [4,4], activation_fn=tf.nn.relu) 

      dil_s_up_1 = tf.contrib.layers.conv2d(s_up_1, 96, kernel_size=3, rate=2)
      sem_mix_1 = tf.concat([s_up_1, n_up_2, n_up_2_mid, n_up_2_high, dil_s_up_1], 3)

      s_up_2= LayerBlocks.boundary_refine(sem_mix_1, name='BR_d_up_2', training=isTrainingRun)
      s_up_2 = slim.convolution2d_transpose(s_up_2, 64, [3,3], [2,2], activation_fn=tf.nn.relu) 

      dil_s_up_2 = tf.contrib.layers.conv2d(s_up_2, 48, kernel_size=3, rate=8)
      sem_mix_2 = tf.concat([s_up_2, n_up_3,  n_up_3_mid, n_up_3_high , dil_s_up_2], 3)


      s_up_3 = LayerBlocks.boundary_refine(sem_mix_2, name='BR_d_up_3' ,training=isTrainingRun)
      s_up_3 = slim.convolution2d_transpose(s_up_3, 64, [3,3], [2,2], activation_fn=tf.nn.relu)

      dil_s_up_3 = tf.contrib.layers.conv2d(s_up_3, 48, kernel_size=3, rate=16)
      sem_mix_3 = tf.concat([s_up_3, n_up_4, n_up_4_mid, n_up_4_high ,dil_s_up_3], 3)

      s_up_4 = LayerBlocks.boundary_refine(sem_mix_3, name='BR_d_up_4', training=isTrainingRun)
      s_up_4 = slim.convolution2d_transpose(s_up_4, 64, [3,3], [2,2], activation_fn=tf.nn.relu) 

      dil_s_up_4 = tf.contrib.layers.conv2d(s_up_4, 48, kernel_size=3, rate=16)
      sem_mix_4 = tf.concat([s_up_4, n_up_5, n_up_5_mid, n_up_5_high, dil_s_up_4], 3)


      depth_endpoint = tf.layers.conv2d(sem_mix_4, 1 , (1,1), activation=tf.nn.relu, padding='VALID')
      depth_endpoint = LayerBlocks.boundary_refine(depth_endpoint, name='BR_depth_endpoint' ,training=isTrainingRun)


      print("s_up_1  : " + str(s_up_1.get_shape()[:]))
      print("s_up_2 : " + str(s_up_2.get_shape()[:]))
      print("s_up_3 : " + str(s_up_3.get_shape()[:]))
      print("s_up_4 : " + str(s_up_4.get_shape()[:]))
      print("depth_endpoint before resize : " + str(depth_endpoint.get_shape()[:]))


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
