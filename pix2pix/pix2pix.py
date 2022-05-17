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
from utils import tone_mapping

TF_TRAIN_DIR = "./tf_records/synthetic/train/*.tfrecords"
TF_VAL_DIR = "./tf_records/synthetic/val/*.tfrecords"
TF_TEST_DIR = "./tf_records/synthetic/test/*.tfrecords"

BATCH_SIZE = 128
IMSHAPE = [32, 128, 3]

NUM = 2

def downsampling(filters, size, apply_batchnorm=True):
    initializer = tf.random_normal_initializer(0., 0.02)

    result = tf.keras.Sequential()
    result.add(
        tf.keras.layers.Conv2D(filters, size, strides=2, padding="same", kernel_initializer=initializer, use_bias=False)
    )

    if apply_batchnorm:
        result.add(tfa.layers.InstanceNormalization())
    result.add(tf.keras.layers.LeakyReLU())

    return result

def upsampling(filters, size, apply_dropout=False):
    initializer = tf.random_normal_initializer(0.,0.02)

    result = tf.keras.Sequential()
    result.add(tf.keras.layers.UpSampling2D(size=2, interpolation="nearest"))
    result.add(tf.keras.layers.Conv2D(filters, size, strides=1, padding="same", kernel_initializer=initializer, use_bias=False ))

    if apply_dropout:
        result.add(tf.keras.layers.Dropout(0.5))

    result.add(tf.keras.layers.ReLU())

    return result

def Generator(): #~unet
    # TODO input_ldr_image에 noise삽입
    inputs = tf.keras.layers.Input(shape=[32,128,3])

    down_stack=[
        downsampling(64,4,apply_batchnorm=False),       # n,16,64,64
        downsampling(128,4),                            # n,8,32,128
        downsampling(256,4),                            # n,4,16,256
        downsampling(512,2),                            # n,2,8,512
        downsampling(512,2)                             # n,1,4,512
    ]
    up_stack=[
        upsampling(512,2,apply_dropout=True),           # n,2,8,512
        upsampling(256,2,apply_dropout=True),           # n,4,16,256
        upsampling(128,4),                              # n,8,32,128
        upsampling(64,4),                               # n,16,64,64
        upsampling(32,4)                                # n,32,128,32
    ]

    initializer = tf.random_normal_initializer(0.,0.02)
    last = tf.keras.layers.Conv2D(3, 3, strides=1, padding="same", kernel_initializer=initializer, activation="tanh") # n,32,128,3

    x = inputs
    # skip-connection 없음.
    for layer in down_stack:
        x = layer(x)

    for layer in up_stack:
        x = layer(x)

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


def generator_loss(disc_generated_output, gen_output, target):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    LAMBDA = 2
    gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)

    # mean absolute error
    l1_loss = tf.reduce_mean(tf.abs(target-gen_output))
    total_gan_loss = gan_loss + (LAMBDA*l1_loss)

    return total_gan_loss, gan_loss, l1_loss

def Discriminator():
    initializer = tf.random_normal_initializer(0.,0.02)

    input = tf.keras.layers.Input(shape=[32,128,3], name="input_ldr_image")
    target = tf.keras.layers.Input(shape=[32,128,3], name="target_hdr_iamge")

    x = tf.keras.layers.concatenate([input, target])    # n,32,128,6

    down1 = downsampling(64,4,apply_batchnorm=False)(x) # n,16,64,64
    down2 = downsampling(128,4)(down1)                  # n,8,32,128
    down3 = downsampling(256,4)(down2)                  # n,4,16,256

    conv = tf.keras.layers.Conv2D(512,4,strides=1, padding="same", kernel_initializer=initializer, use_bias=False)(down3) # n,4,16,512
    batchnorm1 = tf.keras.layers.BatchNormalization()(conv)
    leaky_relu = tf.keras.layers.LeakyReLU()(batchnorm1)
    last = tf.keras.layers.Conv2D(1,4,strides=1,kernel_initializer=initializer)(leaky_relu)                               # n,4,16,512

    return tf.keras.Model(inputs=[input, target], outputs=last)

def discriminator_loss(disc_real_output, disc_generated_output):
    loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)

    real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)

    generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

    total_disc_loss = real_loss + generated_loss

    return total_disc_loss, real_loss, generated_loss


def generate_images(model, test_input, target, epoch, save_dir):
    prediction = model(test_input, training=False)
    plt.figure(figsize = (15,15))

    display_list = [test_input[0], tone_mapping(target[0]), prediction[0]]
    title = ["Input Image", "Ground Truth", "Predicted Image"]

    for i in range(3):
        plt.subplot(1,3,i+1)
        plt.title(title[i])

        plt.imshow(display_list[i])
        plt.axis("off")

    #TODO generated image inverse tone mapping하여 exr 파일로 저장하여 직집 비교 해보기

    file_name = "generated_e{0:03d}".format(epoch)
    save_path = os.path.join(save_dir,file_name)
    plt.savefig(save_path)


def get_sample(ldr, hdr, target):
    n_sample = int(hdr.shape[0]/2)
    index = np.random.randint(0,hdr.shape[0],n_sample)

    selected_ldr = tf.gather(ldr, indices=index)
    selected_hdr = tf.gather(hdr, indices=index)
    selected_target = tf.gather(target, indices=index)

    return selected_ldr, selected_hdr, selected_target

#@tf.function
def train_step(generator, discriminator, input_image, target, step, summary_writer):
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        gen_output = generator(input_image, training=True)

        # disc, gen이 학습하는 개수를 맞추기 위해 input_sample, real_sample, fake_sample을 반개씩 고르는 함수 필요
        # input_sample, gen_output_sample, target_sample = get_sample(input_image, gen_output, target)

        # loss의 범위 조절을 위해 tonemapping / generator의 output 범위 (-1 ~ 1)
        target = tf.map_fn(tone_mapping, target)

        disc_real_output = discriminator([input_image, target], training=True)
        disc_generated_output = discriminator([input_image, gen_output], training=True)

        gen_total_loss, gen_gan_loss, gen_l1_loss = generator_loss(disc_generated_output, gen_output, target)
        disc_loss, real_loss, generated_loss = discriminator_loss(disc_real_output, disc_generated_output)

    generator_gradients = gen_tape.gradient(gen_total_loss, generator.trainable_variables)
    discriminator_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(generator_gradients, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(discriminator_gradients, discriminator.trainable_variables))

    with summary_writer.as_default():
        tf.summary.scalar('gen_total_loss(gan+l1)', gen_total_loss, step=step)
        tf.summary.scalar('gen_gan_loss', gen_gan_loss, step=step)
        tf.summary.scalar('gen_l1_loss', gen_l1_loss, step=step)
        tf.summary.scalar('disc_loss(real+generated)', disc_loss, step=step)
        tf.summary.scalar('disc_real_loss', real_loss, step=step)
        tf.summary.scalar('disc_generated_loss', generated_loss, step=step)
    

    # loss={}
    # loss["gen_total_loss"] = gen_total_loss
    # loss["gen_gan_loss"] = gen_gan_loss
    # loss["gen_l1_loss"] = gen_l1_loss
    # loss["disc_real_output"] = disc_real_output
    # loss["disc_generated_output"] = disc_generated_output
    # return loss

def fit(generator, discrimiantor, train_ds, val_ds, steps, num_batch):
    ldr_input, hdr_target = next(iter(val_ds.take(1)))
    start = time.time()
    epoch = 0

    log_dir="logs/"

    summary_writer = tf.summary.create_file_writer(log_dir + "fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    for step, (ldr_input, hdr_target) in train_ds.repeat().take(steps).enumerate():
        if step % num_batch == 0:
            epoch += 1
            if epoch % 5 == 0:
                generate_images(generator, ldr_input, hdr_target, epoch, save_dir= checkpoint_dir)
                checkpoint.save(file_prefix=checkpoint_prefix)    
        train_step(generator, discrimiantor, ldr_input, hdr_target, step, summary_writer)


if __name__== "__main__":
    generator = Generator()
    discriminator = Discriminator()

    train_ds = load_tfrecords(TF_TRAIN_DIR)
    val_ds = load_tfrecords(TF_VAL_DIR)
    num_batch = 307 #len(list(train_ds)) # 307

    generator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1 = 0.5)
    discriminator_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

    checkpoint_dir = "./pix2pix/training_checkpoints"

    checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(generator_optimizer = generator_optimizer,
                                    discriminator_optimizer = discriminator_optimizer,
                                    generator = generator,
                                    discriminator = discriminator)

    os.makedirs(checkpoint_prefix, exist_ok=True)

    fit(generator, discriminator, train_ds, val_ds, 30000, num_batch)
    

