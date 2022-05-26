# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb#scrollTo=GFyPlBWv1B5j

import datetime
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from load_tfrecords import load_tfrecords, get_batch, load_tfrecords_batch
from utils import plot_images
from pix2pix_model import Generator, Discriminator, generator_loss, discriminator_loss

TF_TRAIN_DIR = "./tf_records/synthetic/train/*.tfrecords"
TF_VAL_DIR = "./tf_records/synthetic/val/*.tfrecords"
TF_TEST_DIR = "./tf_records/synthetic/test/*.tfrecords"

NUM = 10
MODE = "WGAN"
CHECKPOINT_DIR = "./pix2pix/training_checkpoints"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, str(NUM))
GP_WEIGHT = 10
N_CRITIC = 3

BATCH_SIZE = 128
IMSHAPE = [32, 128, 3]

@tf.function
def interpolate(gen_output, target, discriminator):
    b = target.shape[0]
    alpha = tf.random.uniform(shape=[b,1,1,1])
    interpolated = tf.multiply(alpha, target) + tf.multiply((1-alpha),gen_output)
    
    prob_interpolated = discriminator([interpolated, target], training=False)
    gradients = tf.gradients(prob_interpolated ,interpolated)

    return gradients

def gradient_panelty(gen_output, target, discriminator):
    # https://github.com/EmilienDupont/wgan-gp/blob/ef82364f2a2ec452a52fbf4a739f95039ae76fe3/training.py#L73
    b = target.shape[0]
    gradients = interpolate(gen_output, target, discriminator)[0]
    gradients = tf.reshape(gradients,(b, -1)) # flatten
    gradients_norm = tf.math.sqrt(tf.math.reduce_sum(tf.math.square(gradients), axis=-1)+1e-12)

    return tf.math.reduce_mean(tf.math.square(gradients_norm-1))

@tf.function
def train_step(generator, discriminator, input_image, target, step, summary_writer):

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gp = gradient_panelty(gen_output, target, discriminator)

        gen_total_loss, gen_gan_loss, gen_l1_loss, em_distance = generator_loss(disc_generated_output, gen_output, target, mode=MODE)
        disc_loss, real_loss, generated_loss = discriminator_loss(disc_real_output, disc_generated_output, gen_output, target, mode=MODE)
        disc_loss += GP_WEIGHT*gp


    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss(gan+l1+em)', gen_total_loss, step=step)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step)
        tf.summary.scalar('wasserstein_distance', em_distance, step=step)
        tf.summary.scalar('disc_loss(real+generated)', disc_loss, step=step)
        tf.summary.scalar('disc_real_loss', real_loss, step=step)
        tf.summary.scalar('disc_generated_loss', generated_loss, step=step)
        tf.summary.scalar('gradient_panelty', gp, step=step)
    

def fit(generator, discrimiantor, train_ds, val_ds, steps, num_batch, checkpoint_prefix):
    ldr_input, hdr_target = next(iter(val_ds.take(1)))
    start = time.time()
    epoch = 0

    log_dir="logs/"

    summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    for step, (ldr_input, hdr_target) in train_ds.repeat().take(steps).enumerate():
        if step % num_batch == 0:
            epoch += 1
            if epoch % 5 == 0:
                plot_images(generator, ldr_input, hdr_target, epoch, save_dir= checkpoint_prefix) 
                checkpoint.save(file_prefix=checkpoint_prefix)    
        train_step(generator, discrimiantor, ldr_input, hdr_target, step, summary_writer)


if __name__== "__main__":

    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    #     except RuntimeError as e:
    #         print(e)
            
    generator = Generator()
    discriminator = Discriminator()

    train_ds = load_tfrecords_batch(TF_TRAIN_DIR)
    val_ds = load_tfrecords_batch(TF_VAL_DIR)
    num_batch = 307 #len(list(train_ds)) # 307

    if MODE == "BCE":
        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    elif MODE == "WGAN": 
        generator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1 = 0, beta_2 = 0.9)
        discriminator_optimizer = tf.keras.optimizers.Adam(1e-4, beta_1=0, beta_2 = 0.9)

    checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,

                                    discriminator_optimizer = discriminator_optimizer,
                                    generator = generator,
                                    discriminator = discriminator)

    os.makedirs(CHECKPOINT_PREFIX, exist_ok=True)


    fit(generator, discriminator, train_ds, val_ds, 30000, num_batch, checkpoint_prefix=CHECKPOINT_PREFIX)
    

