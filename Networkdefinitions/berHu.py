
def berHu_loss(y_true, y_pred):
    '''Calculates the inverse Huber-Loss. Tensors must have the same shape and size
    '''

    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    y_true = tf.reshape(y_true, [-1])
    y_pred = tf.reshape(y_pred, [-1])

    err = tf.subtract(y_pred, y_true)
    abs_err = tf.abs(err)


    #abs_err <= c
    #--> abs_err

    # abs_err > c
    #--> (abs_err**2 + c**2) / (2*tens)
    c = 0.2 * tf.reduce_max(abs_err)
    fraction =  tf.add(abs_err**2, c**2)
    fraction =  tf.divide(fraction, 2*c)


    return  tf.reduce_mean(tf.where(abs_err <= c, abs_err  , fraction ))  # if, then, else)