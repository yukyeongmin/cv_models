
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

from utils.utils import pixel_to_startpoint, pixel_to_position, position_to_label

def plot_initial_state(pred, real, filename):
    fig, ax = plt.subplots(figsize=(8,8))
    pred = ax.scatter(pred[:,1], pred[:,0],s=5,c='tab:orange')
    gt = ax.scatter(real[:,1], real[:,0],s=5,c='tab:blue')
    ax.legend((gt, pred),('ground truth', 'pred'))
    plt.title(f'Prediction Visualization Keras Callback - init')
    plt.xlabel('altitude')
    plt.ylabel('azimuth')
    plt.savefig('./save/{}/plot_results/initial_state.png'.format(filename))


def plot_patches(image_patches, image_shape=(256,1024,3), patch_size=32, sunpose = (-1,-1)):
    """
    image_patches : tf.tensor type
    """
    width = int(image_shape[1]//patch_size)
    height = int(image_shape[0]//patch_size)
    if sunpose[0] == -1:
        plt.figure(figsize=(32,4))
        for i, patch in enumerate(image_patches):
            ax =plt.subplot(8, 32,i+1)
            patch_img = tf.reshape(patch, (patch_size,patch_size,3))
            plt.imshow(patch_img.numpy().astype("uint8"))
            plt.axis("off")
        plt.show()
    else :
        plt.figure(figsize=(32,4))
        for i, patch in enumerate(image_patches):
            ax =plt.subplot(height, width,i+1)
            patch_img = tf.reshape(patch, (patch_size,patch_size,3))
            if i == sunpose[0]+sunpose[1]*width :
                patch_img = np.zeros(shape=patch_img.shape)
                patch_img[:,:,2] = 255
                plt.imshow(patch_img.astype("uint8"))
            else:
                plt.imshow(patch_img.numpy().astype("uint8"))
            plt.axis("off")
        plt.show()

def plot_pred_class(image, position, image_shape, patch_size):
    fig, ax = plt.subplots(figsize=(8,4))
    plt.imshow(image)
    plt.scatter(position[0],position[1],s=5,c='red')
    start_point = pixel_to_startpoint(image_shape,patch_size,position)
    rect = patches.Rectangle(start_point,patch_size, patch_size, linewidth=1,edgecolor='b',facecolor='none')
    ax.add_patch(rect)
    plt.axis("off")
    plt.show()


def plot_image(image, x, y):
    plt.figure(figsize=(32,4))
    plt.scatter(x,y, s=5, c='blue')
    plt.imshow(image)
    plt.show()