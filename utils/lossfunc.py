import tensorflow as tf
from ViT.ViT_hyperparameters import vitHyperParameters
import numpy as np

# indices = np.array([[0], [1], [2]])
# print(indices.shape)
# depth = 3
# print(tf.squeeze(tf.one_hot(indices, depth)))  # output: [3 x 3]
# # [[1., 0., 0.],
# #  [0., 1., 0.],
# #  [0., 0., 1.]]

# exit()
hyperparameters = vitHyperParameters()
input_shape = hyperparameters.input_shape
patch_size = hyperparameters.patch_size
num_patches = hyperparameters.num_patches


def l1Loss(y_true, y_pred):
    l1 = tf.math.reduce_mean(tf.math.abs(y_true - y_pred))

    return l1


def wingLoss(y_true, y_pred, omega=10, epsilon=0.5):
    '''
    y_true, y_pred = (theta, phi)
    '''
    diff = tf.reduce_sum(tf.math.abs(y_true-y_pred),axis=1)
    diff_under = tf.cast(diff[diff < omega], tf.float64)
    diff_over = tf.cast(diff[diff >= omega], tf.float64)

    loss_1 = omega * tf.math.log(1 + diff_under/epsilon)
    C = omega - omega * tf.math.log(1+omega/epsilon)
    C = tf.cast(C, tf.float64)
    loss_2 = tf.math.subtract(diff_over, C)

    loss_1 = tf.cast(loss_1,tf.float64)
    loss_2 = tf.cast(loss_2,tf.float64)
    num = tf.cast((len(loss_1)+len(loss_2)), tf.float64)

    return (tf.math.reduce_sum(loss_1) + tf.math.reduce_sum(loss_2)) / num

# 
def AWingLoss(y_true, y_pred, omega, epsilon, alpha, theta):
    diff = tf.reduce_sum(tf.math.abs(y_true-y_pred),axis=1)
    diff_under = tf.cast(diff[diff < theta], tf.float64)
    diff_over = tf.cast(diff[diff >= theta], tf.float64)

    term_1 = tf.math.pow((diff_under/epsilon),alpha-y_true)
    loss_1 = omega * tf.math.log(1 + term_1)
    term_2 = tf.math.pow(theta/epsilon, alpha-y_true)+1
    # A = omega* (1/)
    # loss_2 = 
    return 0


def CustomCCE(y_true, y_pred_label, input_shape=input_shape, patch_size=patch_size, num_patches=num_patches):
    '''
    categorical cross entropy
    y_true = (altitude, azimuth)    (batch,2)
    y_pred = patch_num (label)      (batch,num_patches)
    '''
    height = int(input_shape[0]//patch_size)
    width = int(input_shape[1]//patch_size)

    x = 2*y_true[:,0]*256                    # altitude
    y = y_true[:,1]*1024                     # azimuth

    x = tf.cast((x//32),tf.int64)
    y = tf.cast((y//32),tf.int64)

    x = tf.expand_dims(x, axis=1)
    y = tf.expand_dims(y, axis=1)
    
    y_true_label = y+x*width
    y_true_label = tf.one_hot(y_true_label,depth=num_patches)
    y_true_label = tf.squeeze(y_true_label)
    y_pred_label = tf.convert_to_tensor(y_pred_label)

    # y_pred_position = (y_pred_label%32, int(y_pred_label//32))
    # y_true_position = np.concatenate([x,y],axis=1)

    ### 일단 가장 쉬운 cross entropy 부터
    cce = tf.keras.losses.CategoricalCrossentropy(
        reduction=tf.keras.losses.Reduction.SUM
        )
    cce_loss = cce(y_true_label,y_pred_label)

    return cce_loss

def customMSE(y_true, y_pred, input_shape=input_shape, patch_size=patch_size, num_patches=num_patches):
    '''
    this function is for computing loss with multiple outputs
    unet gives two outputs (image embedding, reconstructed image)
    y_pred = [reconstructed image, image embedding]
           = [(batch, 256, 512, 3), (batch, 128)]
    '''

    reconstruction = y_pred[1]
    mse = tf.keras.losses.MeanSquaredError(name='reconstruction_mse')
    loss = mse(y_true, reconstruction)
    return loss
