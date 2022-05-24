# https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/tutorials/generative/pix2pix.ipynb#scrollTo=GFyPlBWv1B5j

from configparser import Interpolation
import datetime
import tensorflow as tf
import tensorflow_addons as tfa
import numpy as np
import matplotlib.pyplot as plt
import os
import time

from load_tfrecords import load_tfrecords
from utils import tone_mapping, inverse_tone_mapping, plot_images
from pix2pix_model import Generator, Discriminator, generator_loss, discriminator_loss

TF_TRAIN_DIR = "./tf_records/synthetic/train/*.tfrecords"
TF_VAL_DIR = "./tf_records/synthetic/val/*.tfrecords"
TF_TEST_DIR = "./tf_records/synthetic/test/*.tfrecords"

NUM = 7
MODE = "WGAN"
CHECKPOINT_DIR = "./pix2pix/training_checkpoints"
CHECKPOINT_PREFIX = os.path.join(CHECKPOINT_DIR, str(NUM))

BATCH_SIZE = 128
IMSHAPE = [32, 128, 3]


#@tf.function
def train_step(generator, discriminator, input_image, target, step, summary_writer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        # disc, gen이 학습하는 개수를 맞추기 위해 input_sample, real_sample, fake_sample을 반개씩 고르는 함수 필요
        # input_sample, gen_output_sample, target_sample = get_sample(input_image, gen_output, target)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss, em_distance = generator_loss(disc_generated_output, gen_output, target, mode=MODE)
        disc_loss, real_loss, generated_loss = discriminator_loss(disc_real_output, disc_generated_output, mode=MODE)

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
                plot_images(generator, ldr_input, hdr_target, epoch, save_dir= checkpoint_prefix) # TODO val_ds 와 연결하기
                checkpoint.save(file_prefix=checkpoint_prefix)    
        train_step(generator, discrimiantor, ldr_input, hdr_target, step, summary_writer)


if __name__== "__main__":

    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
    # 텐서플로가 첫 번째 GPU만 사용하도록 제한
        try:
            tf.config.experimental.set_visible_devices(gpus[1], 'GPU')
        except RuntimeError as e:
            # 프로그램 시작시에 접근 가능한 장치가 설정되어야만 합니다
            print(e)
            
    
    generator = Generator()
    discriminator = Discriminator()

    train_ds = load_tfrecords(TF_TRAIN_DIR)
    val_ds = load_tfrecords(TF_VAL_DIR)
    num_batch = 307 #len(list(train_ds)) # 307

    if MODE == "BCE":
        generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)
        discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
    elif MODE == "WGAN": # TODO WGAN-GP로 weight 값 눌러주기
        generator_optimizer = tf.keras.optimizers.RMSprop(2e-4)
        discriminator_optimizer = tf.keras.optimizers.RMSprop(2e-4)

    checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,

                                    discriminator_optimizer = discriminator_optimizer,
                                    generator = generator,
                                    discriminator = discriminator)

    os.makedirs(CHECKPOINT_PREFIX, exist_ok=True)

    fit(generator, discriminator, train_ds, val_ds, 30000, num_batch, checkpoint_prefix=CHECKPOINT_PREFIX)
    

