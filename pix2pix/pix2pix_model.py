import tensorflow as tf
import tensorflow_addons as tfa
from utils import inverse_tone_mapping, compare_bright

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
    inputs = tf.keras.layers.Input(shape=[32,128,3])

    down_stack=[
        downsampling(64,4,apply_batchnorm=False),       # n,16,64,64
        downsampling(128,4),                            # n,8,32,128
        downsampling(256,4),                            # n,4,16,256
        downsampling(512,2),                            # n,2,8,512
        downsampling(512,2)                             # n,1,4,512
    ]
    up_stack=[
        upsampling(512,2,apply_dropout=True),           # n,2,8,256
        upsampling(256,4),                              # n,4,16,128
        upsampling(128,4),                               # n,8,32,64
        upsampling(64,4),                                # n,16,64,32
    ]

    initializer = tf.random_normal_initializer(0.,0.02)
    last = tf.keras.Sequential()
    last.add(tf.keras.layers.UpSampling2D(size=2, interpolation="nearest"))
    last.add(tf.keras.layers.Conv2D(3, 3, strides=1, padding="same", kernel_initializer=initializer, activation="tanh" )) # n,32,128,3

    x = inputs
   
    skips = []
   
    for layer in down_stack:
        x = layer(x)
        skips.append(x)

    skips = reversed(skips[:-1]) # 마지막 down_stack의 output제외

    for layer, skip in zip(up_stack, skips): # len(skips)=4
        x = layer(x)
        x = tf.keras.layers.Concatenate()([x, skip])

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)


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
    last = tf.keras.layers.Conv2D(1,4,strides=1,padding="same",activation="sigmoid", kernel_initializer=initializer)(leaky_relu)                 # n,4,16,512

    return tf.keras.Model(inputs=[input, target], outputs=last)


def generator_loss(disc_generated_output, gen_output, target, mode = "BCE"): 
    LAMBDA1 = 1
    LAMBDA2 = 0.01
    if mode == "BCE":
        # adversarial loss1(BCE)
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        gan_loss = loss_object(tf.ones_like(disc_generated_output), disc_generated_output)
    elif mode == "WGAN":
        # adversarial loss2(WGAN==mean)
        gan_loss = tf.reduce_mean(disc_generated_output)

    # mean absolute error
    target_ = inverse_tone_mapping(target)
    gen_output_ = inverse_tone_mapping(gen_output)
    l1_loss = tf.reduce_mean(tf.abs(target-gen_output))

    # wasserstein distance
    em_distance = compare_bright(gen_output_, target_)
    total_gan_loss = gan_loss + (LAMBDA1*l1_loss) + (LAMBDA2*em_distance)

    return total_gan_loss, gan_loss, l1_loss, em_distance


def discriminator_loss(disc_real_output, disc_generated_output, gen_output, target, mode="BCE"):

    if mode =="BCE":
        # adversarial loss(BCE)
        loss_object = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = loss_object(tf.ones_like(disc_real_output), disc_real_output)
        generated_loss = loss_object(tf.zeros_like(disc_generated_output), disc_generated_output)

        total_disc_loss = real_loss + generated_loss
    elif mode =="WGAN":
        # adversarial loss(WGAN==mean)
        real_loss = tf.reduce_mean(disc_real_output)
        generated_loss = tf.reduce_mean(disc_generated_output)

        total_disc_loss = -real_loss + generated_loss
    # elif mode == "MSE":
    #     # discriminator에 sigmoid acti쓰면안됨.
    #     total_disc_loss = tf.math.square(disc_real_output-disc_generated_output)

    return total_disc_loss, real_loss, generated_loss
